"""
Convert the 30% experiment labels into the format that
stage1b_active_learning.py expects (phase1_binary_labels.csv).
Backs up the original 10% labels before overwriting.

Usage:
    python scripts/prep_30pct_for_al.py
"""

# Windows consoles default to cp1252; bot names and review text are not.
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


import pandas as pd
from pathlib import Path

LABELED_DIR = Path("data/labeled")
EXPERIMENT_LABELS = LABELED_DIR / "experiment_labels_accumulated.csv"
PHASE1_BACKUP = LABELED_DIR / "phase1_binary_labels_10pct_backup.csv"
PHASE1_PATH = LABELED_DIR / "phase1_binary_labels.csv"

def main():
    # Backup original 10% labels
    if PHASE1_PATH.exists() and not PHASE1_BACKUP.exists():
        orig = pd.read_csv(PHASE1_PATH)
        orig.to_csv(PHASE1_BACKUP, index=False)
        print(f"Backed up original labels: {PHASE1_BACKUP} ({len(orig):,} rows)")

    # Load 30% experiment labels
    exp = pd.read_csv(EXPERIMENT_LABELS, low_memory=False)
    print(f"Loaded experiment labels: {len(exp):,} rows")

    # Map column names to what stage1b expects
    col_map = {}

    # Label column
    for col in ["llm_label", "label", "privacy_label"]:
        if col in exp.columns:
            col_map[col] = "label"
            break

    # Confidence column
    for col in ["llm_confidence", "confidence"]:
        if col in exp.columns:
            col_map[col] = "confidence"
            break

    # Text column
    for col in ["text_clean", "text"]:
        if col in exp.columns:
            if col != "text":
                col_map[col] = "text"
            break

    exp = exp.rename(columns=col_map)

    # Ensure required columns exist
    if "label" not in exp.columns:
        print("ERROR: No label column found")
        return
    if "confidence" not in exp.columns:
        exp["confidence"] = 0.9  # default high confidence
    if "text" not in exp.columns:
        print("ERROR: No text column found")
        return

    # Filter valid labels
    exp = exp[exp["label"].isin(["privacy", "not_privacy"])].copy()

    # Save as phase1 format
    exp.to_csv(PHASE1_PATH, index=False)

    n_priv = (exp["label"] == "privacy").sum()
    n_not = (exp["label"] == "not_privacy").sum()
    print(f"\nSaved: {PHASE1_PATH}")
    print(f"  Total: {len(exp):,}")
    print(f"  Privacy: {n_priv:,}")
    print(f"  Not privacy: {n_not:,}")
    print(f"\nNow run: python scripts/stage1b_active_learning.py")

if __name__ == "__main__":
    main()