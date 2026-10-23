"""
Stage 7 — Analysis & Metrics

Computes the headline numbers of Sec. IV-A (Table I): overall SPRR, SPRR by bot
tag and by bot, the theme distribution (TPR), and affective expression by theme.

The SPRR-over-time trend (Fig. 2, R^2 = 0.35) is fitted in
stage8_visualizations.py; the permission and metadata analysis of RQ3
(Sec. IV-D) is in rq3_permissions_metadata.py.

Usage:
    python scripts/stage7_analysis.py
"""

import json
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    PROCESSED_DIR, RESULTS_DIR,
    MIN_REVIEWS_PER_THEME, TOP_N_THEMES
)


def compute_sprr(df_all: pd.DataFrame, df_privacy: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
    """
    Compute Security and Privacy Review Rate (SPRR).
    SPRR = (privacy_reviews / total_reviews) * 100
    """
    if group_col:
        total = df_all.groupby(group_col).size().reset_index(name="total_reviews")
        privacy = df_privacy.groupby(group_col).size().reset_index(name="privacy_reviews")
        merged = total.merge(privacy, on=group_col, how="left").fillna(0)
        merged["sprr"] = merged["privacy_reviews"] / merged["total_reviews"] * 100
        return merged.sort_values("sprr", ascending=False)
    else:
        ppr = len(df_privacy) / len(df_all) * 100
        return pd.DataFrame([{"total_reviews": len(df_all), "privacy_reviews": len(df_privacy), "sprr": ppr}])


def compute_tpr(df_privacy: pd.DataFrame, theme_col: str = "primary_theme") -> pd.DataFrame:
    """
    Compute Percentage of Theme Privacy Reviews (TPR).
    TPR = (reviews_in_theme / total_privacy_reviews) * 100
    """
    theme_counts = df_privacy[theme_col].value_counts().reset_index()
    theme_counts.columns = [theme_col, "count"]
    theme_counts["tpr"] = theme_counts["count"] / len(df_privacy) * 100
    return theme_counts


def run_analysis():
    """Run full analysis pipeline."""
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("[Stage 7] Loading data")
    all_classified = pd.read_csv(PROCESSED_DIR / "all_reviews_classified.csv")
    privacy_full = pd.read_csv(PROCESSED_DIR / "privacy_reviews_full.csv")
    
    print(f"  All reviews: {len(all_classified):,}")
    print(f"  Privacy reviews: {len(privacy_full):,}")
    
    # =========================================================================
    # 1. Overall SPRR
    # =========================================================================
    print(f"\n{'='*60}")
    print("1. OVERALL SPRR (Percentage of Privacy Reviews)")
    print(f"{'='*60}")
    
    overall_sprr = len(privacy_full) / len(all_classified) * 100
    print(f"  SPRR(all) = {overall_sprr:.2f}%")
    
    # =========================================================================
    # 2. SPRR by Bot Category
    # =========================================================================
    if "tags" in all_classified.columns:
        print(f"\n{'='*60}")
        print("2. SPRR BY BOT CATEGORY")
        print(f"{'='*60}")
        
        # Explode tags for both dataframes
        def explode_tags(df):
            df = df.copy()
            df["tag"] = df["tags"].fillna("").str.split(",")
            return df.explode("tag").assign(tag=lambda x: x["tag"].str.strip())
        
        all_exploded = explode_tags(all_classified)
        privacy_exploded = explode_tags(privacy_full)
        
        sprr_by_tag = compute_sprr(all_exploded, privacy_exploded, "tag")
        sprr_by_tag = sprr_by_tag[sprr_by_tag["total_reviews"] >= 100]
        
        print(f"\n  Top 20 categories by SPRR:")
        for _, row in sprr_by_tag.head(20).iterrows():
            print(f"    {row['tag']}: {row['sprr']:.2f}% ({row['privacy_reviews']:,.0f}/{row['total_reviews']:,.0f})")
        
        sprr_by_tag.to_csv(RESULTS_DIR / "sprr_by_category.csv", index=False)
    
    # =========================================================================
    # 3. TPR (Theme Distribution)
    # =========================================================================
    print(f"\n{'='*60}")
    print("3. PRIVACY THEME DISTRIBUTION (TPR)")
    print(f"{'='*60}")
    
    tpr = compute_tpr(privacy_full)
    tpr = tpr[tpr["count"] >= MIN_REVIEWS_PER_THEME]
    
    print(f"\n  Top {TOP_N_THEMES} themes:")
    for _, row in tpr.head(TOP_N_THEMES).iterrows():
        print(f"    {row['primary_theme']}: {row['count']:,} reviews ({row['tpr']:.1f}% TPR)")
    
    tpr.to_csv(RESULTS_DIR / "theme_distribution.csv", index=False)
    
    # =========================================================================
    # 4. Emotion Analysis
    # =========================================================================
    print(f"\n{'='*60}")
    print("4. EMOTION ANALYSIS")
    print(f"{'='*60}")
    
    if "emotion_group" in privacy_full.columns:
        # Overall emotion distribution
        emotion_dist = privacy_full["emotion_group"].value_counts()
        print(f"\n  Overall Emotion Distribution:")
        for emotion, count in emotion_dist.items():
            print(f"    {emotion}: {count:,} ({count/len(privacy_full)*100:.1f}%)")
        
        # Emotion by theme (like Figure 5 in the paper)
        print(f"\n  Emotion Distribution by Theme (top themes):")
        top_themes = tpr.head(15)["primary_theme"].tolist()
        
        emotion_by_theme = {}
        for theme in top_themes:
            theme_df = privacy_full[privacy_full["primary_theme"] == theme]
            if len(theme_df) >= MIN_REVIEWS_PER_THEME:
                emotion_pcts = (theme_df["emotion_group"].value_counts() / len(theme_df) * 100).to_dict()
                emotion_by_theme[theme] = emotion_pcts
                
                # Negative emotion percentage (like paper's Figure 9)
                neg_pct = sum(emotion_pcts.get(e, 0) for e in ["Anger", "Disgust", "Fear", "Sadness"])
                joy_pct = emotion_pcts.get("Joy", 0)
                print(f"    {theme}: {neg_pct:.0f}% negative, {joy_pct:.0f}% joy")
        
        pd.DataFrame(emotion_by_theme).T.to_csv(RESULTS_DIR / "emotion_by_theme.csv")
    
    # =========================================================================
    # 5. SPRR by Bot (which bots get the most privacy complaints?)
    # =========================================================================
    print(f"\n{'='*60}")
    print("5. SPRR BY BOT")
    print(f"{'='*60}")
    
    if "bot_id" in all_classified.columns:
        sprr_by_bot = compute_sprr(all_classified, privacy_full, "bot_id")
        sprr_by_bot = sprr_by_bot[sprr_by_bot["total_reviews"] >= 10]
        
        print(f"\n  Top 20 bots by SPRR (min 10 reviews):")
        for _, row in sprr_by_bot.head(20).iterrows():
            # Get bot name if available
            bot_name = all_classified[all_classified["bot_id"] == row["bot_id"]].get("name", pd.Series(["Unknown"])).iloc[0] if "name" in all_classified.columns else row["bot_id"]
            print(f"    {bot_name}: {row['sprr']:.1f}% ({row['privacy_reviews']:,.0f}/{row['total_reviews']:,.0f})")
        
        sprr_by_bot.to_csv(RESULTS_DIR / "sprr_by_bot.csv", index=False)

    # =========================================================================
    # Summary
    # =========================================================================
    summary = {
        "total_reviews": len(all_classified),
        "privacy_reviews": len(privacy_full),
        "sprr_overall": round(overall_sprr, 2),
        "num_themes": len(tpr),
        "top_theme": tpr.iloc[0]["primary_theme"] if len(tpr) > 0 else "N/A",
        "top_theme_tpr": round(tpr.iloc[0]["tpr"], 1) if len(tpr) > 0 else 0,
    }
    
    with open(RESULTS_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Results saved to: {RESULTS_DIR}")
    
    return summary


if __name__ == "__main__":
    run_analysis()
