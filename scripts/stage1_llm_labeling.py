"""
Stage 1 — Binary Privacy Labeling via Local LLM (Ollama)

Classifies a stratified sample of reviews as privacy-relevant or not,
using Ollama with Llama 3 8B running locally. The labels serve as
training data for the DeBERTa classifier in Stage 3.

This is essentially knowledge distillation — we transfer the LLM's
privacy understanding into a much smaller, faster model that can
classify the full 183K dataset in minutes instead of days.

Usage:
    python scripts/stage1_llm_labeling.py --input data/raw/reviews.csv --sample-size 5000
"""

import argparse
import json
import time
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.taxonomy import get_taxonomy_prompt_text, PRIVACY_TAXONOMY
from config.settings import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    LABELING_SAMPLE_SIZE, LABELED_DIR
)


def build_classification_prompt(review_text: str) -> str:
    """Build the prompt for binary privacy classification."""
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
    "reason": "brief one-sentence explanation",
    "privacy_concepts": ["concept1", "concept2"] or []
}}"""
    
    return prompt


def build_theme_prompt(review_text: str) -> str:
    """Build prompt for fine-grained privacy theme assignment."""
    concepts = list(PRIVACY_TAXONOMY.keys())
    concepts_str = "\n".join(f"- {c}: {PRIVACY_TAXONOMY[c]['description']}" for c in concepts)
    
    prompt = f"""You are a privacy expert. This Discord bot review has been identified as privacy-relevant.

Assign the most relevant privacy theme(s) from this taxonomy:
{concepts_str}

REVIEW: "{review_text}"

Respond with ONLY a JSON object:
{{
    "primary_theme": "most relevant theme name",
    "secondary_themes": ["other relevant themes"] or [],
    "specific_issue": "brief description of the specific privacy issue raised",
    "sentiment_toward_privacy": "positive", "negative", or "neutral"
}}"""
    
    return prompt


def query_ollama(prompt: str, model: str = None) -> dict:
    """Send a prompt to Ollama and parse JSON response."""
    model = model or OLLAMA_MODEL
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temp for consistent labeling
                    "num_predict": 300,
                }
            },
            timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        
        text = response.json()["response"].strip()
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
       # Find JSON object in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            # Try parsing, if it fails try finding just the first complete JSON object
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Find the first balanced {} pair
                depth = 0
                for i, char in enumerate(text[start:], start):
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            return json.loads(text[start:i+1])
        
        return None
        
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"  Ollama error: {e}")
        return None


def create_stratified_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Create a stratified sample across star ratings and text lengths."""
    df = df.copy()
    
    # Filter out empty/very short reviews
    df = df[df["text"].str.len() >= 10].copy()
    
    # Stratify by star rating
    df["star_bin"] = pd.cut(df["stars"], bins=[0, 1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"])
    
    # Stratify by review length (short/medium/long)
    df["len_bin"] = pd.cut(
        df["text"].str.len(), 
        bins=[0, 50, 200, float("inf")], 
        labels=["short", "medium", "long"]
    )
    
    # Sample proportionally from each stratum
    sampled = df.groupby(["star_bin", "len_bin"], group_keys=False, observed=True).apply(
        lambda x: x.sample(n=min(len(x), max(1, sample_size // 15)), random_state=42),
        include_groups=False
    )
    
    # If we need more, sample remaining randomly
    if len(sampled) < sample_size:
        remaining = df[~df.index.isin(sampled.index)]
        extra = remaining.sample(n=min(len(remaining), sample_size - len(sampled)), random_state=42)
        sampled = pd.concat([sampled, extra])
    
    return sampled.head(sample_size).drop(columns=["star_bin", "len_bin"])


def run_labeling(input_path: str, sample_size: int = None):
    """Main labeling pipeline."""
    sample_size = sample_size or LABELING_SAMPLE_SIZE
    
    print(f"[Stage 1] Loading reviews from {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Total reviews: {len(df):,}")
    
    # Ensure required columns
    assert "text" in df.columns, "CSV must have a 'text' column"
    assert "stars" in df.columns, "CSV must have a 'stars' column"
    
    # Create stratified sample
    print(f"\n[Stage 1] Creating stratified sample of {sample_size:,} reviews")
    sample_df = create_stratified_sample(df, sample_size)
    print(f"  Sampled: {len(sample_df):,} reviews")
    
    # Phase 1: Binary classification (privacy / not_privacy)
    print(f"\n[Stage 1] Phase 1: Binary privacy classification via Ollama ({OLLAMA_MODEL})")
    print(f"  Estimated time: ~{len(sample_df) * 3 / 60:.0f} minutes")
    
    results = []
    failed = 0
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Classifying"):
        prompt = build_classification_prompt(row["text"])
        result = query_ollama(prompt)
        
        if result and "label" in result:
            results.append({
                "original_index": idx,
                "bot_id": row.get("bot_id", ""),
                "stars": row["stars"],
                "text": row["text"],
                "date": row.get("date", ""),
                "label": result["label"],
                "confidence": result.get("confidence", 0.5),
                "reason": result.get("reason", ""),
                "privacy_concepts": json.dumps(result.get("privacy_concepts", [])),
            })
        else:
            failed += 1
            results.append({
                "original_index": idx,
                "bot_id": row.get("bot_id", ""),
                "stars": row["stars"],
                "text": row["text"],
                "date": row.get("date", ""),
                "label": "failed",
                "confidence": 0.0,
                "reason": "LLM failed to respond",
                "privacy_concepts": "[]",
            })
    
    labeled_df = pd.DataFrame(results)
    
    # Stats
    total_valid = labeled_df[labeled_df["label"] != "failed"]
    privacy_count = (total_valid["label"] == "privacy").sum()
    not_privacy_count = (total_valid["label"] == "not_privacy").sum()
    
    print(f"\n[Stage 1] Phase 1 Results:")
    print(f"  Privacy:     {privacy_count:,} ({privacy_count/len(total_valid)*100:.1f}%)")
    print(f"  Not-privacy: {not_privacy_count:,} ({not_privacy_count/len(total_valid)*100:.1f}%)")
    print(f"  Failed:      {failed:,}")
    
    # Save Phase 1 results
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    phase1_path = LABELED_DIR / "phase1_binary_labels.csv"
    labeled_df.to_csv(phase1_path, index=False)
    print(f"  Saved: {phase1_path}")
    
    # Phase 2: Theme assignment for privacy reviews
    privacy_reviews = labeled_df[
        (labeled_df["label"] == "privacy") & 
        (labeled_df["confidence"] >= 0.7)
    ].copy()
    
    print(f"\n[Stage 1] Phase 2: Theme assignment for {len(privacy_reviews):,} privacy reviews")
    
    theme_results = []
    
    for idx, row in tqdm(privacy_reviews.iterrows(), total=len(privacy_reviews), desc="Assigning themes"):
        prompt = build_theme_prompt(row["text"])
        result = query_ollama(prompt)
        
        if result and "primary_theme" in result:
            theme_results.append({
                **row.to_dict(),
                "primary_theme": result["primary_theme"],
                "secondary_themes": json.dumps(result.get("secondary_themes", [])),
                "specific_issue": result.get("specific_issue", ""),
                "sentiment_toward_privacy": result.get("sentiment_toward_privacy", "neutral"),
            })
        else:
            theme_results.append({
                **row.to_dict(),
                "primary_theme": "unassigned",
                "secondary_themes": "[]",
                "specific_issue": "",
                "sentiment_toward_privacy": "neutral",
            })
    
    themed_df = pd.DataFrame(theme_results)
    
    phase2_path = LABELED_DIR / "phase2_themed_labels.csv"
    themed_df.to_csv(phase2_path, index=False)
    print(f"  Saved: {phase2_path}")
    
    # Print theme distribution
    print(f"\n[Stage 1] Theme Distribution:")
    theme_counts = themed_df["primary_theme"].value_counts()
    for theme, count in theme_counts.head(20).items():
        print(f"  {theme}: {count} ({count/len(themed_df)*100:.1f}%)")
    
    print(f"\n[Stage 1] COMPLETE. Outputs:")
    print(f"  Binary labels: {phase1_path}")
    print(f"  Themed labels: {phase2_path}")
    
    return labeled_df, themed_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: LLM-based review labeling")
    parser.add_argument("--input", required=True, help="Path to reviews CSV")
    parser.add_argument("--sample-size", type=int, default=LABELING_SAMPLE_SIZE)
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")
    args = parser.parse_args()
    
    if args.model != OLLAMA_MODEL:
        OLLAMA_MODEL = args.model
    
    run_labeling(args.input, args.sample_size)
