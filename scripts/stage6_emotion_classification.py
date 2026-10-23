"""
Stage 6 — Emotion Classification with GoEmotions

Runs a pre-trained GoEmotions model (RoBERTa-base) over all privacy
reviews and maps the 28 fine-grained emotions to 8 Ekman-based groups.
This lets us characterize *how* users feel about privacy issues, not
just *what* they mention.

Usage:
    python scripts/stage6_emotion_classification.py
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
from config.taxonomy import EMOTION_GROUPS
from config.settings import EMOTION_MODEL, PROCESSED_DIR


# GoEmotions label mapping (28 emotions)
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]


def map_to_ekman_group(emotion: str) -> str:
    """Map a fine-grained emotion to Ekman-based group."""
    for group, emotions in EMOTION_GROUPS.items():
        if emotion in emotions:
            return group
    return "Neutral"


def classify_emotions(batch_size: int = 64):
    """Run emotion classification on all privacy-themed reviews."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[Stage 6] Emotion Classification")
    print(f"  Model: {EMOTION_MODEL}")
    print(f"  Device: {device}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)
    model.to(device)
    model.eval()
    
    # Load themed privacy reviews
    input_path = PROCESSED_DIR / "privacy_reviews_themed.csv"
    print(f"\n[Stage 6] Loading from {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Reviews: {len(df):,}")
    
    # Classify in batches
    print(f"\n[Stage 6] Classifying emotions (batch_size={batch_size})")
    
    all_emotions = []
    all_emotion_groups = []
    all_emotion_scores = []
    
    texts = df["text"].fillna("").tolist()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Emotion classification"):
        batch_texts = texts[i:i + batch_size]
        
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**encodings)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
        
        for j in range(len(batch_texts)):
            # Get top emotion
            top_idx = np.argmax(probs[j])
            top_emotion = GOEMOTIONS_LABELS[top_idx]
            top_score = probs[j][top_idx]
            
            # Get emotion group
            emotion_group = map_to_ekman_group(top_emotion)
            
            # Get all emotions above threshold (multi-label)
            threshold = 0.3
            active_emotions = [
                GOEMOTIONS_LABELS[k] 
                for k in range(len(GOEMOTIONS_LABELS)) 
                if probs[j][k] >= threshold
            ]
            
            all_emotions.append(top_emotion)
            all_emotion_groups.append(emotion_group)
            all_emotion_scores.append(json.dumps({
                "top_emotion": top_emotion,
                "top_score": float(top_score),
                "active_emotions": active_emotions,
            }))
    
    # Add to dataframe
    df["emotion"] = all_emotions
    df["emotion_group"] = all_emotion_groups
    df["emotion_details"] = all_emotion_scores
    
    # Save
    output_path = PROCESSED_DIR / "privacy_reviews_full.csv"
    df.to_csv(output_path, index=False)
    
    # Stats
    print(f"\n[Stage 6] Emotion Distribution (Ekman groups):")
    group_counts = df["emotion_group"].value_counts()
    for group, count in group_counts.items():
        print(f"  {group}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print(f"\n  Top Fine-Grained Emotions:")
    emotion_counts = df["emotion"].value_counts()
    for emotion, count in emotion_counts.head(15).items():
        print(f"  {emotion}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Emotion by theme
    print(f"\n  Dominant Emotion by Theme:")
    theme_emotion = df.groupby("primary_theme")["emotion_group"].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else "Unknown"
    )
    for theme, emotion in theme_emotion.items():
        count = len(df[df["primary_theme"] == theme])
        if count >= 50:
            print(f"  {theme} ({count:,} reviews): {emotion}")
    
    print(f"\n[Stage 6] COMPLETE.")
    print(f"  Output: {output_path}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6: Emotion classification")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    classify_emotions(args.batch_size)
