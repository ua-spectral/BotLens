"""
Stage 1b — Active Learning Loop for Privacy Classification

Iteratively improves the DeBERTa classifier by finding the reviews
it's most uncertain about, getting fresh LLM labels for those, and
retraining. Inspired by Cui et al. (PoPETs 2025) Section 4.1.

Key ideas:
  - Soft labels: each sample gets labeled 3 times at temp=0.7 so we
    get a probability distribution, not just a hard yes/no
  - Uncertainty sampling: pick reviews where model confidence is
    closest to 0.5 (i.e. the ones it's struggling with)
  - Minority class boost: in early rounds, bias toward potential
    privacy reviews the model is unsure about
  - Auto-deploy: if the new model beats the old one, swap it in

Usage:
    python scripts/stage1b_active_learning.py
    python scripts/stage1b_active_learning.py --rounds 5 --samples-per-round 500
    python scripts/stage1b_active_learning.py --resume
"""

import argparse
import json
import shutil
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.taxonomy import get_taxonomy_prompt_text
from config.settings import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    LABELED_DIR, PROCESSED_DIR, MODELS_DIR, RAW_DIR,
    MODEL_NAME, MAX_SEQ_LENGTH, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    LEARNING_RATE, NUM_EPOCHS, WARMUP_RATIO, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, PREDICTION_THRESHOLD,
    LABELING_CONFIDENCE_THRESHOLD
)

# =============================================================================
# ACTIVE LEARNING CONFIG
# =============================================================================
AL_DIR = LABELED_DIR / "active_learning"
AL_ROUNDS = 5
AL_SAMPLES_PER_ROUND = 500
AL_SOFT_LABEL_RUNS = 3
AL_SOFT_LABEL_TEMP = 0.7
AL_UNCERTAINTY_NOISE_STD = 0.15
AL_MINORITY_BOOST_ROUNDS = 2
AL_CONVERGENCE_THRESHOLD = 0.005
AL_DOWNSAMPLE_RATIO = 5.0

# Production model path (where Stage 4 reads from)
PROD_MODEL_PATH = MODELS_DIR / "deberta-privacy-classifier" / "best_model"
BACKUP_MODEL_PATH = MODELS_DIR / "deberta-privacy-classifier" / "best_model_backup"


# =============================================================================
# DATASET CLASS
# =============================================================================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts.tolist(), truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


# =============================================================================
# OLLAMA: SOFT LABELING
# =============================================================================
def build_classification_prompt(review_text: str) -> str:
    taxonomy_text = get_taxonomy_prompt_text()
    prompt = f"""You are a privacy expert analyzing Discord bot user reviews.

Your task: Determine if the following review discusses ANY privacy-related topic.

A review is PRIVACY-RELEVANT if it mentions ANY of these concepts:
{taxonomy_text}

A review is NOT-PRIVACY if it only discusses:
- General functionality (bot features, commands, bugs)
- Performance issues (speed, uptime, lag)
- UI/UX preferences
- Pricing/payment (unless about data monetization)
- General praise/complaints unrelated to privacy

REVIEW: "{review_text}"

Respond with ONLY a JSON object, nothing else:
{{
    "label": "privacy" or "not_privacy",
    "confidence": 0.0 to 1.0,
    "reason": "brief one-sentence explanation"
}}"""
    return prompt


def query_ollama(prompt: str, temperature: float = 0.1) -> dict:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                "options": {"temperature": temperature, "num_predict": 300}
            },
            timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        text = response.json()["response"].strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
        return None
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return None


def get_soft_label(review_text: str, n_runs: int = AL_SOFT_LABEL_RUNS) -> dict:
    prompt = build_classification_prompt(review_text)
    privacy_votes = 0
    not_privacy_votes = 0
    confidences = []
    reasons = []
    for _ in range(n_runs):
        result = query_ollama(prompt, temperature=AL_SOFT_LABEL_TEMP)
        if result and "label" in result:
            label = result["label"].strip().lower()
            conf = float(result.get("confidence", 0.5))
            if label == "privacy": privacy_votes += 1
            elif label == "not_privacy": not_privacy_votes += 1
            confidences.append(conf)
            reasons.append(result.get("reason", ""))
    total_votes = privacy_votes + not_privacy_votes
    if total_votes == 0:
        return None
    privacy_prob = privacy_votes / total_votes
    avg_confidence = np.mean(confidences) if confidences else 0.5
    hard_label = "privacy" if privacy_prob > 0.5 else "not_privacy"
    agreement = max(privacy_votes, not_privacy_votes) / total_votes
    combined_confidence = agreement * avg_confidence
    return {
        "label": hard_label, "privacy_prob": round(privacy_prob, 3),
        "confidence": round(combined_confidence, 3), "agreement": round(agreement, 3),
        "privacy_votes": privacy_votes, "not_privacy_votes": not_privacy_votes,
        "n_runs": total_votes, "reasons": reasons,
    }


# =============================================================================
# UNCERTAINTY SAMPLING (Cui et al. Section 4.1)
# =============================================================================
def select_uncertain_samples(df_unlabeled, privacy_scores, n_samples, al_round):
    df = df_unlabeled.copy()
    df["_privacy_score"] = privacy_scores
    minority_boost = al_round < AL_MINORITY_BOOST_ROUNDS
    if minority_boost:
        privacy_leaning = df[(df["_privacy_score"] > 0.3) & (df["_privacy_score"] < 0.8)]
        if len(privacy_leaning) >= n_samples // 2:
            n_minority = n_samples // 2
            n_general = n_samples - n_minority
            pl = privacy_leaning.copy()
            eps = np.random.normal(0, AL_UNCERTAINTY_NOISE_STD, len(pl))
            pl["_unc"] = np.abs(pl["_privacy_score"] - (0.5 + eps[:len(pl)]))
            minority_sel = pl.nsmallest(n_minority, "_unc")
            remaining = df[~df.index.isin(minority_sel.index)].copy()
            eps_g = np.random.normal(0, AL_UNCERTAINTY_NOISE_STD, len(remaining))
            remaining["_unc"] = np.abs(remaining["_privacy_score"] - (0.5 + eps_g[:len(remaining)]))
            general_sel = remaining.nsmallest(n_general, "_unc")
            selected = pd.concat([minority_sel, general_sel])
            print(f"    Minority boost: {n_minority} privacy-leaning + {n_general} general")
            return selected.drop(columns=["_privacy_score", "_unc"], errors="ignore")
    eps = np.random.normal(0, AL_UNCERTAINTY_NOISE_STD, len(df))
    df["_unc"] = np.abs(df["_privacy_score"] - (0.5 + eps[:len(df)]))
    selected = df.nsmallest(n_samples, "_unc")
    return selected.drop(columns=["_privacy_score", "_unc"], errors="ignore")


# =============================================================================
# MODEL INFERENCE
# =============================================================================
def score_unlabeled(model, tokenizer, texts, device, batch_size=64):
    model.eval()
    all_probs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring unlabeled"):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(
            batch, truncation=True, padding=True,
            max_length=MAX_SEQ_LENGTH, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
    return np.array(all_probs)


# =============================================================================
# MODEL TRAINING
# =============================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= PREDICTION_THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0,
    }


def prepare_balanced_splits(df):
    df = df.copy()
    privacy_df = df[df["label_int"] == 1]
    not_privacy_df = df[df["label_int"] == 0]
    ratio = len(not_privacy_df) / max(len(privacy_df), 1)
    if ratio > AL_DOWNSAMPLE_RATIO:
        target = int(len(privacy_df) * AL_DOWNSAMPLE_RATIO)
        not_privacy_df = not_privacy_df.sample(n=target, random_state=42)
        df = pd.concat([privacy_df, not_privacy_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_int"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label_int"])
    return train_df, val_df, test_df


def train_model(train_df, val_df, tokenizer, round_num, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, problem_type="single_label_classification"
    ).to(device)
    train_dataset = ReviewDataset(train_df["text"], train_df["label_int"], tokenizer, MAX_SEQ_LENGTH)
    val_dataset = ReviewDataset(val_df["text"], val_df["label_int"], tokenizer, MAX_SEQ_LENGTH)
    round_dir = output_dir / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(round_dir), num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE, warmup_ratio=WARMUP_RATIO, weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        logging_steps=50, fp16=torch.cuda.is_available(), dataloader_num_workers=4, report_to="none",
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    trainer.train()
    return trainer, model


def evaluate_model(trainer, test_df, tokenizer):
    test_dataset = ReviewDataset(test_df["text"], test_df["label_int"], tokenizer, MAX_SEQ_LENGTH)
    preds_output = trainer.predict(test_dataset)
    probs = torch.softmax(torch.tensor(preds_output.predictions), dim=-1)[:, 1].numpy()
    preds = (probs >= PREDICTION_THRESHOLD).astype(int)
    report = classification_report(test_df["label_int"], preds, target_names=["not_privacy", "privacy"], output_dict=True)
    report_str = classification_report(test_df["label_int"], preds, target_names=["not_privacy", "privacy"])
    cm = confusion_matrix(test_df["label_int"], preds)
    print(f"\n  Classification Report:")
    print(report_str)
    print(f"  Confusion Matrix:\n    {cm}")
    return {
        "accuracy": report["accuracy"],
        "privacy_precision": report["privacy"]["precision"],
        "privacy_recall": report["privacy"]["recall"],
        "privacy_f1": report["privacy"]["f1-score"],
        "not_privacy_f1": report["not_privacy"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
    }


# =============================================================================
# AUTO-DEPLOY BEST MODEL
# =============================================================================
def auto_deploy_best_model(best_f1, baseline_f1):
    """
    Automatically deploy the best AL model to production if it improved.
    Stage 4 reads from PROD_MODEL_PATH so the switch is seamless.
    """
    al_best = AL_DIR / "best_model"
    if not al_best.exists():
        print("\n  [AUTO-DEPLOY] No AL best model found. Skipping.")
        return False

    if best_f1 <= baseline_f1:
        print(f"\n  [AUTO-DEPLOY] AL did NOT improve F1 ({baseline_f1:.4f} -> {best_f1:.4f})")
        print(f"  Keeping original model. No deployment needed.")
        return False

    print(f"\n  [AUTO-DEPLOY] AL improved F1: {baseline_f1:.4f} -> {best_f1:.4f}")
    print(f"  Deploying best model to production...")

    # Backup current production model
    if PROD_MODEL_PATH.exists():
        if BACKUP_MODEL_PATH.exists():
            shutil.rmtree(BACKUP_MODEL_PATH)
        shutil.copytree(PROD_MODEL_PATH, BACKUP_MODEL_PATH)
        print(f"  Backed up current model to: {BACKUP_MODEL_PATH}")

    # Deploy best AL model to production
    if PROD_MODEL_PATH.exists():
        shutil.rmtree(PROD_MODEL_PATH)
    shutil.copytree(al_best, PROD_MODEL_PATH)

    print(f"  Deployed to: {PROD_MODEL_PATH}")
    print(f"  Stage 4 will now use the improved model automatically.")
    print(f"  To rollback: copy {BACKUP_MODEL_PATH} back to {PROD_MODEL_PATH}")
    return True


# =============================================================================
# MAIN ACTIVE LEARNING LOOP
# =============================================================================
def run_active_learning(n_rounds=AL_ROUNDS, samples_per_round=AL_SAMPLES_PER_ROUND, resume=False):
    AL_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'=' * 70}")
    print(f"  ACTIVE LEARNING PIPELINE")
    print(f"  Inspired by Cui et al. (PoPETs 2025)")
    print(f"{'=' * 70}")
    print(f"  Device:             {device}")
    if device == "cuda":
        print(f"  GPU:                {torch.cuda.get_device_name(0)}")
    print(f"  Base model:         {MODEL_NAME}")
    print(f"  LLM:                {OLLAMA_MODEL}")
    print(f"  Max rounds:         {n_rounds}")
    print(f"  Samples per round:  {samples_per_round}")
    print(f"  Soft label runs:    {AL_SOFT_LABEL_RUNS} per sample (temp={AL_SOFT_LABEL_TEMP})")
    print(f"  Convergence:        stop if F1 improvement < {AL_CONVERGENCE_THRESHOLD}")
    print(f"  Auto-deploy:        enabled (best model -> Stage 4 production path)")
    print()

    # Load Stage 1 labels
    print("[AL] Loading existing labeled data from Stage 1...")
    phase1_path = LABELED_DIR / "phase1_binary_labels.csv"
    if not phase1_path.exists():
        print(f"  ERROR: {phase1_path} not found. Run Stage 1 first.")
        return
    labeled_df = pd.read_csv(phase1_path)
    labeled_df = labeled_df[labeled_df["label"].isin(["privacy", "not_privacy"])].copy()
    labeled_df = labeled_df[labeled_df["confidence"] >= LABELING_CONFIDENCE_THRESHOLD].copy()
    labeled_df["label_int"] = (labeled_df["label"] == "privacy").astype(int)
    labeled_df["al_source"] = "stage1"
    privacy_n = labeled_df["label_int"].sum()
    total_n = len(labeled_df)
    print(f"  Loaded: {total_n:,} labeled reviews ({privacy_n:,} privacy, {total_n - privacy_n:,} not_privacy)")

    # Load all reviews
    print("\n[AL] Loading all reviews...")
    all_reviews_path = RAW_DIR / "reviews.csv"
    if not all_reviews_path.exists():
        print(f"  ERROR: {all_reviews_path} not found. Run Stage 0 first.")
        return
    all_df = pd.read_csv(all_reviews_path)
    all_df = all_df[all_df["text"].notna() & (all_df["text"].str.len() >= 10)].copy()
    print(f"  All reviews: {len(all_df):,}")

    # Resume check
    al_labels_path = AL_DIR / "al_accumulated_labels.csv"
    start_round = 0
    if resume and al_labels_path.exists():
        al_prev = pd.read_csv(al_labels_path)
        al_prev["label_int"] = (al_prev["label"] == "privacy").astype(int)
        labeled_df = pd.concat([labeled_df, al_prev], ignore_index=True)
        start_round = int(al_prev["al_round"].max()) + 1 if "al_round" in al_prev.columns else 1
        print(f"  Resumed: loaded {len(al_prev):,} labels from previous AL rounds")
        print(f"  Starting from round {start_round}")

    labeled_texts = set(labeled_df["text"].tolist())
    print("\n[AL] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Active Learning Rounds
    history = []
    al_new_labels = []
    best_f1 = 0.0
    baseline_f1 = 0.0

    for round_num in range(start_round, start_round + n_rounds):
        round_start = time.time()
        print(f"\n{'=' * 70}")
        print(f"  ROUND {round_num}")
        print(f"{'=' * 70}")

        # Train
        print(f"\n[Round {round_num}] Preparing training data...")
        print(f"  Total labeled: {len(labeled_df):,} ({labeled_df['label_int'].sum():,} privacy, {(labeled_df['label_int'] == 0).sum():,} not_privacy)")
        train_df, val_df, test_df = prepare_balanced_splits(labeled_df)
        print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
        print(f"  Train privacy ratio: {train_df['label_int'].mean() * 100:.1f}%")

        print(f"\n[Round {round_num}] Training DeBERTa...")
        output_dir = AL_DIR / "models"
        trainer, model = train_model(train_df, val_df, tokenizer, round_num, output_dir)

        # Evaluate
        print(f"\n[Round {round_num}] Evaluating...")
        metrics = evaluate_model(trainer, test_df, tokenizer)
        current_f1 = metrics["privacy_f1"]
        improvement = current_f1 - best_f1

        if round_num == start_round:
            baseline_f1 = current_f1
            print(f"\n  Baseline F1 (Round {round_num}): {baseline_f1:.4f}")

        print(f"\n  Privacy F1:        {current_f1:.4f}  (previous best: {best_f1:.4f}, delta={improvement:+.4f})")
        print(f"  Privacy Recall:    {metrics['privacy_recall']:.4f}")
        print(f"  Privacy Precision: {metrics['privacy_precision']:.4f}")

        history.append({
            "round": round_num, "labeled_total": len(labeled_df),
            "labeled_privacy": int(labeled_df["label_int"].sum()),
            "train_size": len(train_df), **metrics,
        })

        # Save best model
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_path = AL_DIR / "best_model"
            trainer.save_model(str(best_model_path))
            tokenizer.save_pretrained(str(best_model_path))
            print(f"  >>> New best model saved: {best_model_path}")

        # Convergence check
        if round_num > start_round and improvement < AL_CONVERGENCE_THRESHOLD:
            print(f"\n  Converged: F1 improvement ({improvement:.4f}) < threshold ({AL_CONVERGENCE_THRESHOLD})")
            print(f"  Stopping after round {round_num}.")
            break

        # Score unlabeled
        print(f"\n[Round {round_num}] Scoring unlabeled reviews...")
        unlabeled_mask = ~all_df["text"].isin(labeled_texts)
        unlabeled_df = all_df[unlabeled_mask].copy()
        print(f"  Unlabeled reviews: {len(unlabeled_df):,}")
        if len(unlabeled_df) < samples_per_round:
            print(f"  Not enough unlabeled reviews. Stopping.")
            break

        unlabeled_texts = unlabeled_df["text"].tolist()
        privacy_scores = score_unlabeled(model, tokenizer, unlabeled_texts, device)

        print(f"  Score distribution:")
        for lo, hi, label in [(0.0,0.2,"confident not_privacy"),(0.2,0.4,"leaning not_privacy"),(0.4,0.6,"uncertain"),(0.6,0.8,"leaning privacy"),(0.8,1.0,"confident privacy")]:
            count = ((privacy_scores >= lo) & (privacy_scores < hi)).sum()
            print(f"    [{lo:.1f}-{hi:.1f}) {label}: {count:,}")

        # Select uncertain samples
        print(f"\n[Round {round_num}] Selecting {samples_per_round} uncertain samples...")
        selected_df = select_uncertain_samples(unlabeled_df, privacy_scores, samples_per_round, round_num)
        sel_indices = unlabeled_df.index.get_indexer(selected_df.index)
        sel_scores = privacy_scores[sel_indices]
        print(f"  Selected score range: [{sel_scores.min():.3f}, {sel_scores.max():.3f}]")
        print(f"  Selected score mean:  {sel_scores.mean():.3f}")

        # Label with Ollama
        print(f"\n[Round {round_num}] Labeling {len(selected_df)} samples with Ollama ({AL_SOFT_LABEL_RUNS} runs each, temp={AL_SOFT_LABEL_TEMP})...")
        est_time = len(selected_df) * AL_SOFT_LABEL_RUNS * 2 / 60
        print(f"  Estimated time: ~{est_time:.0f} minutes")

        new_labels = []
        failed = 0
        for idx, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc=f"Round {round_num} labeling"):
            text = str(row["text"])
            result = get_soft_label(text, n_runs=AL_SOFT_LABEL_RUNS)
            if result:
                new_labels.append({
                    "text": text, "bot_id": row.get("bot_id", ""),
                    "stars": row.get("stars", 0),
                    "date": row.get("date", ""), "label": result["label"],
                    "confidence": result["confidence"], "privacy_prob": result["privacy_prob"],
                    "agreement": result["agreement"], "privacy_votes": result["privacy_votes"],
                    "not_privacy_votes": result["not_privacy_votes"],
                    "al_round": round_num, "al_source": "active_learning",
                })
            else:
                failed += 1

        print(f"  Labeled: {len(new_labels):,}  Failed: {failed}")
        if not new_labels:
            print("  No new labels obtained. Stopping.")
            break

        new_df = pd.DataFrame(new_labels)
        new_df = new_df[new_df["confidence"] >= LABELING_CONFIDENCE_THRESHOLD].copy()
        new_df["label_int"] = (new_df["label"] == "privacy").astype(int)
        new_privacy = new_df["label_int"].sum()
        new_total = len(new_df)
        print(f"  After confidence filter: {new_total:,} ({new_privacy:,} privacy, {new_total - new_privacy:,} not_privacy)")

        labeled_df = pd.concat([labeled_df, new_df], ignore_index=True)
        labeled_texts.update(new_df["text"].tolist())
        al_new_labels.append(new_df)

        round_path = AL_DIR / f"round_{round_num}_labels.csv"
        new_df.to_csv(round_path, index=False)
        round_elapsed = (time.time() - round_start) / 60
        print(f"\n  Round {round_num} completed in {round_elapsed:.1f} minutes")
        print(f"  Round labels saved: {round_path}")

    # -------------------------------------------------------------------------
    # Save final outputs
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  ACTIVE LEARNING COMPLETE")
    print(f"{'=' * 70}")

    if al_new_labels:
        al_accumulated = pd.concat(al_new_labels, ignore_index=True)
        al_accumulated.to_csv(al_labels_path, index=False)
        print(f"\n  AL labels saved: {al_labels_path} ({len(al_accumulated):,} new labels)")

    history_df = pd.DataFrame(history)
    history_path = AL_DIR / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"  Training history: {history_path}")

    print(f"\n  Round  Labeled    Privacy    F1         Recall     Precision")
    print(f"  {'-' * 62}")
    for _, row in history_df.iterrows():
        print(f"  {int(row['round']):<7} {int(row['labeled_total']):<10} "
              f"{int(row['labeled_privacy']):<10} {row['privacy_f1']:<10.4f} "
              f"{row['privacy_recall']:<10.4f} {row['privacy_precision']:<10.4f}")

    # -------------------------------------------------------------------------
    # AUTO-DEPLOY: Replace production model if AL improved F1
    # -------------------------------------------------------------------------
    deployed = auto_deploy_best_model(best_f1, baseline_f1)

    if deployed:
        print(f"\n  >>> Best model auto-deployed to production.")
        print(f"  >>> Run Stage 4 directly — no manual copy needed:")
        print(f"      python scripts/stage4_classify_all.py --input data/raw/reviews.csv")
    else:
        best_model_path = AL_DIR / "best_model"
        if best_model_path.exists():
            print(f"\n  Best model at: {best_model_path}")
            print(f"  To manually deploy:")
            print(f"    cp -r {best_model_path}/* {PROD_MODEL_PATH}/")

    final_labels_path = AL_DIR / "all_labels_combined.csv"
    labeled_df.to_csv(final_labels_path, index=False)
    print(f"\n  Combined labels (Stage 1 + AL): {final_labels_path} ({len(labeled_df):,} total)")

    return history_df


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1b: Active Learning (Cui et al. PoPETs 2025)")
    parser.add_argument("--rounds", type=int, default=AL_ROUNDS, help=f"Max AL rounds (default: {AL_ROUNDS})")
    parser.add_argument("--samples-per-round", type=int, default=AL_SAMPLES_PER_ROUND)
    parser.add_argument("--resume", action="store_true", help="Resume from previous AL run")
    parser.add_argument("--soft-runs", type=int, default=AL_SOFT_LABEL_RUNS)
    args = parser.parse_args()
    if args.soft_runs != AL_SOFT_LABEL_RUNS:
        AL_SOFT_LABEL_RUNS = args.soft_runs
    run_active_learning(n_rounds=args.rounds, samples_per_round=args.samples_per_round, resume=args.resume)