"""
Experiment — Optimal LLM Labeling Percentage (Incremental)

Figures out how much of the dataset needs to be labeled by the LLM
before DeBERTa performance plateaus. Tests 10%, 20%, 30%, 40%, 45%.

The key insight: at 30%, F1 hits 0.94 and starts declining after that
(the LLM introduces more noise than signal past that point).

Incremental approach — reuses labels from previous runs so you only
pay the Ollama cost for the *new* samples at each step.

Usage:
    python scripts/experiment_labeling_pct.py [--resume]
"""

import pandas as pd
import numpy as np
import json
import time
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from torch.utils.data import Dataset
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, PROCESSED_DIR, LABELED_DIR, RESULTS_DIR

# ── Config ──────────────────────────────────────────────────────────
PERCENTAGES = [0.10, 0.20, 0.30, 0.40, 0.45]
MODEL_NAME = "microsoft/deberta-v3-large"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"
CONFIDENCE_THRESHOLD = 0.7
DOWNSAMPLE_RATIO = 5
MAX_EPOCHS = 10
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_EVAL = 16
LEARNING_RATE = 2e-5
CLASSIFICATION_THRESHOLD = 0.8
SEED = 42

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LABELED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Paths
ACCUMULATED_LABELS_PATH = LABELED_DIR / "experiment_labels_accumulated.csv"
EXISTING_LABELS_PATH = LABELED_DIR / "active_learning" / "al_accumulated_labels.csv"
RESULTS_CSV_PATH = RESULTS_DIR / "labeling_experiment_results.csv"


# ── Privacy classification prompt (same as Stage 1) ──
PRIVACY_PROMPT = """You are a privacy expert analyzing Discord bot reviews.
Classify whether this review mentions ANY privacy-related concern or topic.

Privacy-related includes: data collection, data sharing, permissions,
tracking, surveillance, consent, data deletion, DMs being read,
security concerns, personal information, anonymity, or any other
privacy topic (positive or negative).

Review: "{review}"

Respond ONLY with JSON (no markdown, no explanation):
{{"label": "privacy" or "not_privacy", "confidence": 0.0-1.0}}"""


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


# =====================================================================
# STEP 1: Load existing labels
# =====================================================================
def load_existing_labels():
    """Load already-labeled reviews from Stage 1 / active learning."""
    candidates = [
        ACCUMULATED_LABELS_PATH,
        EXISTING_LABELS_PATH,
        LABELED_DIR / "llm_labels.csv",
    ]

    all_labels = []
    seen_ids = set()

    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            print(f"  Found existing labels: {path.name} ({len(df):,} rows)")

            # Normalize column names
            label_col = None
            for col in ["llm_label", "label", "privacy_label"]:
                if col in df.columns:
                    label_col = col
                    break

            conf_col = None
            for col in ["llm_confidence", "confidence"]:
                if col in df.columns:
                    conf_col = col
                    break

            id_col = None
            for col in ["review_id", "id", "index"]:
                if col in df.columns:
                    id_col = col
                    break

            if label_col is None:
                continue

            rename_map = {label_col: "llm_label"}
            if conf_col:
                rename_map[conf_col] = "llm_confidence"
            if id_col:
                rename_map[id_col] = "review_id"
            df = df.rename(columns=rename_map)

            # Deduplicate
            if "review_id" in df.columns:
                new = df[~df["review_id"].isin(seen_ids)]
                seen_ids.update(new["review_id"].tolist())
            else:
                new = df

            all_labels.append(new)

    if all_labels:
        combined = pd.concat(all_labels, ignore_index=True)
        print(f"  Total existing labels: {len(combined):,}")
        return combined
    else:
        print("  No existing labels found - starting from scratch")
        return pd.DataFrame()


# =====================================================================
# STEP 2: Label new samples with Ollama (incremental)
# =====================================================================
def label_with_ollama(text, max_retries=2):
    prompt = PRIVACY_PROMPT.format(review=str(text)[:500])

    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300}
            }, timeout=60)

            if resp.status_code == 200:
                raw = resp.json().get("response", "")
                try:
                    data = json.loads(raw.strip())
                except json.JSONDecodeError:
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    if start >= 0 and end > start:
                        data = json.loads(raw[start:end])
                    else:
                        continue

                label = data.get("label", "not_privacy")
                confidence = float(data.get("confidence", 0.5))
                return label, confidence

        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
            continue

    return "not_privacy", 0.5


def label_incremental(all_reviews, existing_labels, n_target):
    """
    Label up to n_target reviews total, reusing existing labels.
    Only calls Ollama for the GAP (n_target - n_existing).
    """
    n_existing = len(existing_labels)
    n_needed = max(0, n_target - n_existing)

    print(f"    Target: {n_target:,} | Already have: {n_existing:,} | Need to label: {n_needed:,}")

    if n_needed == 0:
        return existing_labels.sample(min(n_target, len(existing_labels)),
                                       random_state=SEED)

    text_col = "text_clean" if "text_clean" in all_reviews.columns else "text"

    if "review_id" in existing_labels.columns and "review_id" in all_reviews.columns:
        already_labeled_ids = set(existing_labels["review_id"].astype(str))
        unlabeled = all_reviews[~all_reviews["review_id"].astype(str).isin(already_labeled_ids)]
    else:
        already_labeled_idx = set(existing_labels.index) if len(existing_labels) > 0 else set()
        unlabeled = all_reviews[~all_reviews.index.isin(already_labeled_idx)]

    print(f"    Unlabeled pool: {len(unlabeled):,}")

    # Stratified sample
    if "stars" in unlabeled.columns and len(unlabeled) > n_needed:
        sampled = unlabeled.groupby("stars", group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), max(1, int(n_needed * len(x) / len(unlabeled)))),
                random_state=SEED
            )
        )
        if len(sampled) < n_needed:
            remaining = unlabeled[~unlabeled.index.isin(sampled.index)]
            extra = remaining.sample(min(n_needed - len(sampled), len(remaining)),
                                      random_state=SEED)
            sampled = pd.concat([sampled, extra])
        sampled = sampled.head(n_needed)
    else:
        sampled = unlabeled.sample(min(n_needed, len(unlabeled)), random_state=SEED)

    # Label with Ollama
    print(f"    Labeling {len(sampled):,} NEW reviews with Ollama...")
    labels = []
    confidences = []

    for i, (idx, row) in enumerate(tqdm(sampled.iterrows(), total=len(sampled),
                                         desc="    Labeling")):
        text = str(row.get(text_col, ""))
        label, conf = label_with_ollama(text)
        labels.append(label)
        confidences.append(conf)

        # Save checkpoint every 2000 reviews
        if (i + 1) % 2000 == 0:
            partial = sampled.iloc[:i+1].copy()
            partial["llm_label"] = labels
            partial["llm_confidence"] = confidences
            checkpoint = pd.concat([existing_labels, partial], ignore_index=True)
            checkpoint.to_csv(ACCUMULATED_LABELS_PATH, index=False)
            n_priv = sum(1 for l in labels if l == "privacy")
            print(f"      Checkpoint: {i+1:,}/{len(sampled):,} "
                  f"({n_priv} privacy) saved")

    sampled = sampled.copy()
    sampled["llm_label"] = labels
    sampled["llm_confidence"] = confidences

    n_priv_new = (sampled["llm_label"] == "privacy").sum()
    print(f"    New labels: {n_priv_new:,} privacy, {len(sampled)-n_priv_new:,} not_privacy")

    # Combine existing + new
    combined = pd.concat([existing_labels, sampled], ignore_index=True)
    combined.to_csv(ACCUMULATED_LABELS_PATH, index=False)
    print(f"    Saved: {ACCUMULATED_LABELS_PATH} ({len(combined):,} total labels)")

    return combined.head(n_target)


# =====================================================================
# STEP 3: Prepare training data
# =====================================================================
def prepare_training_data(labeled_df):
    high_conf = labeled_df[labeled_df["llm_confidence"] >= CONFIDENCE_THRESHOLD].copy()
    high_conf["label_int"] = (high_conf["llm_label"] == "privacy").astype(int)

    privacy = high_conf[high_conf["label_int"] == 1]
    not_privacy = high_conf[high_conf["label_int"] == 0]

    print(f"    High-conf: {len(high_conf):,} ({len(privacy):,} privacy, "
          f"{len(not_privacy):,} not_privacy)")

    max_not_priv = len(privacy) * DOWNSAMPLE_RATIO
    if len(not_privacy) > max_not_priv:
        not_privacy = not_privacy.sample(max_not_priv, random_state=SEED)

    balanced = pd.concat([privacy, not_privacy]).sample(frac=1, random_state=SEED)

    train_df, temp_df = train_test_split(balanced, test_size=0.2,
                                          stratify=balanced["label_int"],
                                          random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5,
                                        stratify=temp_df["label_int"],
                                        random_state=SEED)

    return train_df, val_df, test_df


# =====================================================================
# STEP 4: Train DeBERTa
# =====================================================================
def train_deberta(train_df, val_df, test_df, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    text_col = "text_clean" if "text_clean" in train_df.columns else "text"

    train_enc = tokenizer(train_df[text_col].fillna("").tolist(), truncation=True,
                          padding=True, max_length=512)
    val_enc = tokenizer(val_df[text_col].fillna("").tolist(), truncation=True,
                        padding=True, max_length=512)
    test_enc = tokenizer(test_df[text_col].fillna("").tolist(), truncation=True,
                         padding=True, max_length=512)

    train_dataset = TextDataset(train_enc, train_df["label_int"].tolist())
    val_dataset = TextDataset(val_enc, val_df["label_int"].tolist())
    test_dataset = TextDataset(test_enc, test_df["label_int"].tolist())

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=MAX_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        preds = (probs >= CLASSIFICATION_THRESHOLD).astype(int)
        return {
            "f1": f1_score(labels, preds, pos_label=1, zero_division=0),
            "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
            "recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    test_results = trainer.evaluate(test_dataset)

    # Full classification report
    test_preds = trainer.predict(test_dataset)
    probs = torch.softmax(torch.tensor(test_preds.predictions), dim=1)[:, 1].numpy()
    pred_labels = (probs >= CLASSIFICATION_THRESHOLD).astype(int)
    true_labels = test_df["label_int"].values

    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\n    Classification Report:")
    print(classification_report(true_labels, pred_labels,
                                 target_names=["not_privacy", "privacy"]))
    print(f"    Confusion Matrix:\n    {confusion_matrix(true_labels, pred_labels)}")

    return {
        "f1": test_results.get("eval_f1", 0),
        "precision": test_results.get("eval_precision", 0),
        "recall": test_results.get("eval_recall", 0),
        "loss": test_results.get("eval_loss", 0),
    }


# =====================================================================
# STEP 5: Plot results
# =====================================================================
def plot_results(results_df):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "figure.dpi": 600,
        "savefig.dpi": 600,
    })

    fig, (ax, ax_tbl) = plt.subplots(
        2, 1, figsize=(6.5, 5.8),
        gridspec_kw={"height_ratios": [3, 1.2]}
    )
    fig.subplots_adjust(top=0.93, bottom=0.05, hspace=0.30)

    pcts = results_df["pct_label"] * 100

    # ── Line chart ──
    ax.plot(pcts, results_df["f1"], "o-", color="#2563EB", lw=2.2,
            markersize=9, label="F1 Score", zorder=3)
    ax.plot(pcts, results_df["precision"], "s--", color="#DC2626", lw=1.5,
            markersize=7, label="Precision", zorder=3)
    ax.plot(pcts, results_df["recall"], "^--", color="#16A34A", lw=1.5,
            markersize=7, label="Recall", zorder=3)

    # Annotate F1 at each point
    for _, row in results_df.iterrows():
        pct_x = row["pct_label"] * 100
        f1 = row["f1"]
        prec = row["precision"]
        rec = row["recall"]

        # F1 above the point (always show)
        ax.annotate(f"F1={f1:.2f}",
                     (pct_x, f1),
                     textcoords="offset points", xytext=(0, 14),
                     ha="center", fontsize=7.5, fontweight="bold", color="#2563EB")

        # Precision label (only if far enough from F1)
        if abs(prec - f1) > 0.04:
            y_off = 8 if prec > f1 else -12
            ax.annotate(f"{prec:.2f}",
                         (pct_x, prec),
                         textcoords="offset points", xytext=(14, y_off),
                         ha="left", fontsize=6.5, color="#DC2626")

        # Recall label (only if far enough from F1 and Precision)
        if abs(rec - f1) > 0.04 and abs(rec - prec) > 0.04:
            y_off = 8 if rec > prec else -12
            ax.annotate(f"{rec:.2f}",
                         (pct_x, rec),
                         textcoords="offset points", xytext=(14, y_off),
                         ha="left", fontsize=6.5, color="#16A34A")

    # Mark optimal point
    best_idx = results_df["f1"].idxmax()
    best_row = results_df.loc[best_idx]
    best_pct = best_row["pct_label"] * 100
    ax.axvline(x=best_pct, color="#D97706", linestyle=":", lw=1.8, alpha=0.8)
    ax.annotate(f"Optimal {best_pct:.0f}%",
                 (best_pct, best_row["f1"]),
                 textcoords="offset points", xytext=(35, -18),
                 ha="center", fontsize=8, fontweight="bold", color="#D97706",
                 arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.2))

    # Shade diminishing returns
    ax.axvspan(best_pct, 47, alpha=0.06, color="#DC2626")

    ax.set_xlabel("Percentage of Dataset Labeled by LLM (%)")
    ax.set_ylabel("Score")
    ax.set_title("Classifier Performance vs. LLM Labeling Percentage",
                 fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    ax.set_ylim(0.4, 1.10)
    ax.set_xlim(7, 48)
    ax.set_xticks([10, 20, 30, 40, 45])
    ax.grid(True, alpha=0.3)

    # ── Results table below chart ──
    ax_tbl.axis("off")

    col_labels = ["Labeled %", "Samples", "Privacy\nLabels", "F1", "Precision", "Recall"]
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            f"{row['pct_label']*100:.0f}%",
            f"{row.get('n_samples', 0):,.0f}",
            f"{row.get('n_privacy', 0):,.0f}",
            f"{row['f1']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
        ])

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colColours=["#E5E7EB"] * 6,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.5)

    # Style header row
    for col in range(6):
        tbl[0, col].set_text_props(fontweight="bold")

    # Highlight optimal row in blue
    best_row_idx = int(results_df["f1"].values.argmax())
    for col in range(6):
        tbl[best_row_idx + 1, col].set_facecolor("#DBEAFE")
        tbl[best_row_idx + 1, col].set_text_props(fontweight="bold")

    path = FIGURES_DIR / "labeling_pct_experiment.png"
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"\n  Figure saved: {path}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip percentages that already have results")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXPERIMENT: OPTIMAL LLM LABELING PERCENTAGE (INCREMENTAL)")
    print("=" * 70)

    # Load all reviews
    all_reviews_path = PROCESSED_DIR / "all_reviews_classified.csv"
    if not all_reviews_path.exists():
        all_reviews_path = RAW_DIR / "bot_reviews_cleaned.csv"
    if not all_reviews_path.exists():
        all_reviews_path = RAW_DIR / "reviews.csv"

    all_reviews = pd.read_csv(all_reviews_path, low_memory=False)
    total_reviews = len(all_reviews)
    print(f"\n  Total reviews: {total_reviews:,}")

    # Load existing labels
    print(f"\n  Loading existing labels...")
    existing_labels = load_existing_labels()

    # Load previous results if resuming
    prev_results = []
    completed_pcts = set()
    if args.resume and RESULTS_CSV_PATH.exists():
        prev_df = pd.read_csv(RESULTS_CSV_PATH)
        prev_results = prev_df.to_dict("records")
        completed_pcts = set(prev_df["pct_label"].tolist())
        print(f"  Resuming: {len(completed_pcts)} percentages already done")

    results = list(prev_results)

    for pct in PERCENTAGES:
        if pct in completed_pcts:
            print(f"\n  [SKIP] {pct*100:.0f}% already completed (--resume)")
            continue

        n_target = int(total_reviews * pct)
        print(f"\n{'=' * 60}")
        print(f"  TESTING {pct*100:.0f}% = {n_target:,} reviews")
        print(f"{'=' * 60}")

        # Step 1: Get labels (reuse existing + label new)
        labeled = label_incremental(all_reviews, existing_labels, n_target)

        # Update existing_labels for next iteration
        if ACCUMULATED_LABELS_PATH.exists():
            existing_labels = pd.read_csv(ACCUMULATED_LABELS_PATH, low_memory=False)

        n_privacy = (labeled["llm_label"] == "privacy").sum()
        n_not = len(labeled) - n_privacy
        print(f"    Labels: {n_privacy:,} privacy, {n_not:,} not_privacy")

        # Step 2: Prepare training data
        try:
            train_df, val_df, test_df = prepare_training_data(labeled)
        except ValueError as e:
            print(f"    [ERROR] Could not prepare data: {e}")
            results.append({
                "pct_label": pct, "n_samples": n_target,
                "n_privacy": n_privacy, "n_not_privacy": n_not,
                "f1": 0, "precision": 0, "recall": 0, "loss": 0,
                "train_size": 0, "val_size": 0, "test_size": 0,
            })
            continue

        print(f"    Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
        print(f"    Train privacy %: {train_df['label_int'].mean()*100:.1f}%")

        # Step 3: Train DeBERTa
        output_dir = Path(f"models/experiment_pct_{int(pct*100)}")
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = train_deberta(train_df, val_df, test_df, output_dir)

        print(f"\n  RESULTS at {pct*100:.0f}%:")
        print(f"    F1:        {metrics['f1']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")

        results.append({
            "pct_label": pct,
            "n_samples": len(labeled),
            "n_privacy": n_privacy,
            "n_not_privacy": n_not,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "loss": metrics["loss"],
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
        })

        # Save after each run (crash-safe)
        results_df = pd.DataFrame(results)
        results_df.to_csv(RESULTS_CSV_PATH, index=False)
        print(f"    Results saved: {RESULTS_CSV_PATH}")

    # ── Final summary ──
    results_df = pd.DataFrame(results).sort_values("pct_label")
    results_df.to_csv(RESULTS_CSV_PATH, index=False)

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n  {'Pct':>5s}  {'Samples':>8s}  {'Privacy':>8s}  {'Train':>7s}  "
          f"{'F1':>7s}  {'Prec':>7s}  {'Recall':>7s}")
    print(f"  {'-'*58}")
    for _, row in results_df.iterrows():
        print(f"  {row['pct_label']*100:5.0f}%  {row.get('n_samples',0):8,.0f}  "
              f"{row.get('n_privacy',0):8,.0f}  {row.get('train_size',0):7,.0f}  "
              f"{row.get('f1',0):7.4f}  {row.get('precision',0):7.4f}  "
              f"{row.get('recall',0):7.4f}")

    valid = results_df[results_df["f1"] > 0]
    if len(valid) > 0:
        best = valid.loc[valid["f1"].idxmax()]
        print(f"\n  OPTIMAL: {best['pct_label']*100:.0f}% -> F1 = {best['f1']:.4f}")

        for i in range(1, len(valid)):
            if valid.iloc[i]["f1"] < valid.iloc[i-1]["f1"]:
                decline_pct = valid.iloc[i]["pct_label"] * 100
                peak_pct = valid.iloc[i-1]["pct_label"] * 100
                delta = valid.iloc[i-1]["f1"] - valid.iloc[i]["f1"]
                print(f"  PEAK at {peak_pct:.0f}%, then F1 DROPS by {delta:.4f} at {decline_pct:.0f}%")
                print(f"  Adding more LLM labels beyond {peak_pct:.0f}% hurts performance")
                break

    if len(valid) >= 2:
        plot_results(valid)

    print(f"\n  Results: {RESULTS_CSV_PATH}")
    print(f"  Labels:  {ACCUMULATED_LABELS_PATH}")


if __name__ == "__main__":
    main()