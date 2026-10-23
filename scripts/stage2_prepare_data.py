"""
Stage 2 — Prepare Training Data for DeBERTa

Takes the LLM-labeled reviews from Stage 1, filters out low-confidence
labels, downsamples the majority class to a 5:1 ratio (since not_privacy
massively outnumbers privacy), and creates stratified train/val/test
splits for fine-tuning.

Usage:
    python scripts/stage2_prepare_data.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    LABELED_DIR, PROCESSED_DIR,
    LABELING_CONFIDENCE_THRESHOLD,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
)


def prepare_training_data():
    """Prepare and split labeled data for fine-tuning."""
    
    print("[Stage 2] Loading LLM-labeled data")
    df = pd.read_csv(LABELED_DIR / "phase1_binary_labels.csv")
    print(f"  Total labeled: {len(df):,}")
    
    # Filter out failed labels
    df = df[df["label"] != "failed"].copy()
    print(f"  After removing failures: {len(df):,}")
    
    # Filter by confidence threshold
    df = df[df["confidence"] >= LABELING_CONFIDENCE_THRESHOLD].copy()
    print(f"  After confidence filter (>={LABELING_CONFIDENCE_THRESHOLD}): {len(df):,}")
    
    # Binary label encoding
    df["label_int"] = (df["label"] == "privacy").astype(int)
    
    # Check class distribution
    privacy_count = df["label_int"].sum()
    not_privacy_count = len(df) - privacy_count
    print(f"\n  Class distribution:")
    print(f"    Privacy:     {privacy_count:,} ({privacy_count/len(df)*100:.1f}%)")
    print(f"    Not-privacy: {not_privacy_count:,} ({not_privacy_count/len(df)*100:.1f}%)")
    
    # Balance classes — downsample majority to 5:1 ratio
    ratio = not_privacy_count / max(privacy_count, 1)
    print(f"\n  Imbalance ratio: {ratio:.1f}:1")
    
    if ratio > 5.0:
        print(f"  Downsampling not-privacy class to 5:1 ratio...")
        privacy_df = df[df["label_int"] == 1]
        not_privacy_df = df[df["label_int"] == 0]
        
        target_not_privacy = len(privacy_df) * 5  # 5:1 ratio
        not_privacy_df = not_privacy_df.sample(n=target_not_privacy, random_state=42)
        
        df = pd.concat([privacy_df, not_privacy_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"  After balancing: {len(df):,}")
        print(f"    Privacy:     {df['label_int'].sum():,} ({df['label_int'].mean()*100:.1f}%)")
        print(f"    Not-privacy: {(df['label_int'] == 0).sum():,} ({(1-df['label_int'].mean())*100:.1f}%)")
    
    # Split: train / val / test
    train_df, temp_df = train_test_split(
        df, test_size=(VAL_SPLIT + TEST_SPLIT), 
        random_state=42, stratify=df["label_int"]
    )
    
    relative_test = TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        random_state=42, stratify=temp_df["label_int"]
    )
    
    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = PROCESSED_DIR / f"{name}.csv"
        split_df[["text", "label_int"]].to_csv(path, index=False)
        privacy_pct = split_df["label_int"].mean() * 100
        print(f"\n  {name}: {len(split_df):,} reviews ({privacy_pct:.1f}% privacy) -> {path}")
    
    # Save label mapping
    label_map = {"not_privacy": 0, "privacy": 1}
    with open(PROCESSED_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f)
    
    print(f"\n[Stage 2] COMPLETE.")


if __name__ == "__main__":
    prepare_training_data()