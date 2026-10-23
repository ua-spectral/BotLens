"""
Stage 4 — Classify All Reviews with Fine-Tuned DeBERTa

Runs the trained DeBERTa model across every review in the dataset
(183K+) in batches. Each review gets a privacy_score (0-1) and a
binary is_privacy label based on the threshold. This is the step
that produces the final SPRR numbers for the paper.

Usage:
    python scripts/stage4_classify_all.py --input data/raw/reviews.csv
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    MODELS_DIR, PROCESSED_DIR, MAX_SEQ_LENGTH, PREDICTION_THRESHOLD
)


def classify_all_reviews(input_path: str, batch_size: int = 64):
    """Run DeBERTa on all reviews for binary privacy classification."""
    
    model_path = MODELS_DIR / "deberta-privacy-classifier" / "best_model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[Stage 4] Loading model from {model_path}")
    print(f"  Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    
    # Load all reviews
    print(f"\n[Stage 4] Loading reviews from {input_path}")
    try:
        df = pd.read_csv(input_path, encoding="utf-8")
    except UnicodeDecodeError:
        print("  UTF-8 failed, trying latin-1 encoding...")
        df = pd.read_csv(input_path, encoding="latin-1")
    print(f"  Total reviews: {len(df):,}")
    
    # Filter empty reviews
    df = df[df["text"].notna() & (df["text"].str.len() >= 5)].copy()
    print(f"  After filtering empty/short: {len(df):,}")
    
    # Classify in batches
    print(f"\n[Stage 4] Classifying all reviews (batch_size={batch_size})")
    all_probs = []
    
    texts = df["text"].tolist()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying"):
        batch_texts = texts[i:i + batch_size]
        
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        
        all_probs.extend(probs.tolist())
    
    # Add predictions to dataframe
    df["privacy_score"] = all_probs
    df["is_privacy"] = (df["privacy_score"] >= PREDICTION_THRESHOLD).astype(int)
    
    # Stats
    privacy_count = df["is_privacy"].sum()
    total = len(df)
    sprr = privacy_count / total * 100
    
    print(f"\n[Stage 4] Classification Results:")
    print(f"  Total reviews:   {total:,}")
    print(f"  Privacy reviews: {privacy_count:,} ({sprr:.2f}% SPRR)")
    print(f"  Not-privacy:     {total - privacy_count:,}")
    
    # Save all classified reviews
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_path = PROCESSED_DIR / "all_reviews_classified.csv"
    df.to_csv(all_path, index=False)
    print(f"\n  All reviews saved: {all_path}")
    
    # Save privacy-only subset
    privacy_df = df[df["is_privacy"] == 1].copy()
    privacy_path = PROCESSED_DIR / "privacy_reviews.csv"
    privacy_df.to_csv(privacy_path, index=False)
    print(f"  Privacy subset saved: {privacy_path}")
    
    # SPRR by bot category (if tags available)
    if "tags" in df.columns:
        print(f"\n  SPRR by Bot Category (top 20):")
        # Explode tags
        df_tags = df.copy()
        df_tags["tag_list"] = df_tags["tags"].str.split(",")
        df_exploded = df_tags.explode("tag_list")
        df_exploded["tag_list"] = df_exploded["tag_list"].str.strip()
        
        tag_stats = df_exploded.groupby("tag_list").agg(
            total_reviews=("is_privacy", "count"),
            privacy_reviews=("is_privacy", "sum")
        ).assign(
            sprr=lambda x: x["privacy_reviews"] / x["total_reviews"] * 100
        ).sort_values("sprr", ascending=False)
        
        for tag, row in tag_stats.head(20).iterrows():
            print(f"    {tag}: {row['sprr']:.2f}% SPRR ({row['privacy_reviews']:,.0f}/{row['total_reviews']:,.0f})")
    
    print(f"\n[Stage 4] COMPLETE.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Classify all reviews")
    parser.add_argument("--input", required=True, help="Path to all reviews CSV")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    classify_all_reviews(args.input, args.batch_size)