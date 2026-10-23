"""
Stage 5 — Privacy Theme Assignment via LLM

For each review flagged as privacy-relevant in Stage 4, we ask Ollama
to assign one of the 33 taxonomy concepts (22 from Akgul et al. + 11
Discord-specific). Processes reviews one at a time — slower but more
reliable than batching with smaller LLMs.

Usage:
    python scripts/stage5_theme_assignment.py
    python scripts/stage5_theme_assignment.py --limit 50   # test run
"""

import argparse
import json
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.taxonomy import PRIVACY_TAXONOMY, get_theme_list_prompt
from config.settings import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    PROCESSED_DIR
)


def query_ollama_single(prompt: str) -> dict:
    """Send a prompt to Ollama and parse JSON response with robust parsing."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 200,
                }
            },
            timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()

        text = response.json()["response"].strip()

        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find first complete JSON object using bracket matching
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])

        return None

    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        return None


def assign_theme(review_text: str) -> dict:
    """Assign privacy theme to a single review."""
    themes_str = get_theme_list_prompt()

    prompt = f"""You are a privacy expert analyzing a Discord bot user review.

Assign the most relevant privacy theme from this list:
{themes_str}

REVIEW: "{review_text[:500]}"

Respond with ONLY a JSON object (no extra text):
{{"primary_theme": "theme name", "secondary_themes": [], "specific_issue": "one sentence description", "sentiment": "positive or negative or neutral"}}"""

    return query_ollama_single(prompt)


def run_theme_assignment(limit: int = None):
    """Assign themes to all privacy reviews."""

    privacy_path = PROCESSED_DIR / "privacy_reviews.csv"
    print(f"[Stage 5] Loading privacy reviews from {privacy_path}")
    df = pd.read_csv(privacy_path)

    if limit:
        df = df.head(limit)

    print(f"  Reviews to process: {len(df):,}")
    est_minutes = len(df) * 2 / 60
    print(f"  Estimated time: ~{est_minutes:.0f} minutes")

    results = []
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Assigning themes"):
        text = str(row.get("text", ""))
        if len(text) < 5:
            failed += 1
            results.append({
                **row.to_dict(),
                "primary_theme": "unassigned",
                "secondary_themes": "[]",
                "specific_issue": "",
                "privacy_sentiment": "neutral",
            })
            continue

        result = assign_theme(text)

        if result and "primary_theme" in result and result["primary_theme"]:
            # Validate theme name against taxonomy
            theme = str(result["primary_theme"])
            valid_themes = list(PRIVACY_TAXONOMY.keys())
            if theme not in valid_themes:
                # Try to find closest match
                theme_lower = theme.lower()
                matched = False
                for vt in valid_themes:
                    if vt.lower() in theme_lower or theme_lower in vt.lower():
                        theme = vt
                        matched = True
                        break
                if not matched:
                    theme = "unassigned"

            results.append({
                **row.to_dict(),
                "primary_theme": theme,
                "secondary_themes": json.dumps(result.get("secondary_themes", [])),
                "specific_issue": result.get("specific_issue", ""),
                "privacy_sentiment": result.get("sentiment", "neutral"),
            })
        else:
            failed += 1
            results.append({
                **row.to_dict(),
                "primary_theme": "unassigned",
                "secondary_themes": "[]",
                "specific_issue": "",
                "privacy_sentiment": "neutral",
            })

    themed_df = pd.DataFrame(results)

    # Save
    output_path = PROCESSED_DIR / "privacy_reviews_themed.csv"
    themed_df.to_csv(output_path, index=False)

    # Stats
    assigned = themed_df[themed_df["primary_theme"] != "unassigned"]
    print(f"\n[Stage 5] Theme Assignment Results:")
    print(f"  Processed:  {len(themed_df):,}")
    print(f"  Assigned:   {len(assigned):,}")
    print(f"  Unassigned: {failed:,}")

    print(f"\n  Theme Distribution (TPR):")
    theme_counts = themed_df["primary_theme"].value_counts()
    total_assigned = len(assigned)

    for theme, count in theme_counts.head(30).items():
        tpr = count / total_assigned * 100 if total_assigned > 0 else 0
        print(f"    {theme}: {count:,} ({tpr:.1f}%)")

    print(f"\n  Privacy Sentiment:")
    sent_counts = themed_df["privacy_sentiment"].value_counts()
    for sent, count in sent_counts.items():
        print(f"    {sent}: {count:,} ({count / len(themed_df) * 100:.1f}%)")

    # Category-level breakdown
    from config.taxonomy import CATEGORIES
    print(f"\n  By Category:")
    for cat, themes in CATEGORIES.items():
        cat_count = themed_df[themed_df["primary_theme"].isin(themes)].shape[0]
        cat_pct = cat_count / total_assigned * 100 if total_assigned > 0 else 0
        print(f"    {cat}: {cat_count:,} ({cat_pct:.1f}%)")

    print(f"\n[Stage 5] COMPLETE.")
    print(f"  Output: {output_path}")

    return themed_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5: Theme assignment")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of reviews")
    args = parser.parse_args()
    run_theme_assignment(args.limit)