"""
Stage 8 — Publication-Ready Visualizations

Generates paper Figs. 2-9 (files fig02_* .. fig09_*), sized for IEEE 2-column format
(3.5 in single-column width). Uses the Wong (2011) colorblind-safe
palette throughout, with white bar edges and data labels on every bar.

Outputs both PDF and PNG for each figure.

Usage:
    python scripts/stage8_visualizations.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats
from matplotlib.patches import Patch

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.taxonomy import CATEGORIES

# =============================================================================
# IEEE 2-Column Paper Configuration
# =============================================================================
SINGLE_COL_W = 3.5    # IEEE single column width (inches)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =============================================================================
# Wong (2011) Colorblind-Safe Palette — Nature Methods 8, 441
# =============================================================================
WONG = {
    "blue":      "#0072B2",
    "orange":    "#E69F00",
    "green":     "#009E73",
    "vermilion": "#D55E00",
    "sky_blue":  "#56B4E9",
    "purple":    "#CC79A7",
    "yellow":    "#F0E442",
    "black":     "#000000",
}

# Ordered list for cycling through categories/series
WONG_CYCLE = [
    WONG["blue"], WONG["orange"], WONG["green"], WONG["vermilion"],
    WONG["sky_blue"], WONG["purple"], WONG["yellow"], WONG["black"],
]

# --- Emotion colors (mapped to Wong palette) ---
# Negative emotions use warmer/darker Wong colors
# Positive emotions use cooler Wong colors
# Neutral uses accessibility gray
EMOTION_COLORS = {
    "Anger":     WONG["vermilion"],   # #D55E00
    "Disgust":   WONG["purple"],      # #CC79A7
    "Fear":      WONG["yellow"],      # #F0E442
    "Sadness":   WONG["sky_blue"],    # #56B4E9
    "Confusion": WONG["orange"],      # #E69F00
    "Neutral":   "#BBBBBB",           # Gray (accessibility neutral)
    "Joy":       WONG["green"],       # #009E73
    "Surprise":  WONG["blue"],        # #0072B2
}

# 4 taxonomy category colors (Wong palette)
CATEGORY_COLORS = {
    "Data Practices":         WONG["blue"],       # #0072B2
    "User Rights & Controls": WONG["green"],      # #009E73
    "Security & Safety":      WONG["orange"],     # #E69F00
    "Discord-Specific Risks": WONG["vermilion"],  # #D55E00
}

# =============================================================================
# CORRECT 33-THEME TAXONOMY MAPPING
# Must match the actual theme names output by Stage 5 (Llama 3)
# =============================================================================
# Theme -> category mapping is derived from config/taxonomy.py, which is the single
# source of truth for the taxonomy. Deriving it rather than restating it here keeps
# theme names and group assignments identical to Table V of the paper.
THEME_TO_CATEGORY = {
    theme: category
    for category, themes in CATEGORIES.items()
    for theme in themes
}


def save_fig(fig, fig_dir, name):
    """Save figure as PDF + PNG."""
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {name}")


def get_main_tag(tags_value):
    """Extract first tag from tags column (handles string lists)."""
    if pd.isna(tags_value):
        return "other"
    if isinstance(tags_value, str):
        first = tags_value.split(",")[0].strip().strip("[]'\"")
        return first if first else "other"
    return "other"


# =============================================================================
# FIGURE 1: SPRR Trend Over Time (line + regression)
# =============================================================================
def fig02_sprr_over_time(all_reviews, privacy_reviews, fig_dir):
    ar = all_reviews.dropna(subset=["date"]).copy()
    pr = privacy_reviews.dropna(subset=["date"]).copy()

    all_monthly = ar.set_index("date").resample("30D").size()
    priv_monthly = pr.set_index("date").resample("30D").size()
    combined = pd.DataFrame({"all": all_monthly, "priv": priv_monthly}).fillna(0)
    combined = combined[combined["all"] > 0]
    combined["sprr"] = combined["priv"] / combined["all"] * 100

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.2))

    ax.plot(combined.index, combined["sprr"], color=WONG["blue"], lw=1.0,
            label="SPRR", zorder=3)

    x_num = np.arange(len(combined))
    valid = ~np.isnan(combined["sprr"].values)
    if valid.sum() > 2:
        slope, intercept, r_val, _, _ = stats.linregress(
            x_num[valid], combined["sprr"].values[valid]
        )
        ax.plot(combined.index, slope * x_num + intercept,
                color=WONG["vermilion"], ls="--", lw=0.8,
                label=f"Trend ($R^2$={r_val**2:.2f})")

    ax.set_ylabel("SPRR (%)")
    ax.set_xlabel("Time")
    ax.set_title("Security and Privacy Review Rate (SPRR) Over Time", fontweight="bold", pad=6)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)

    ax.legend(loc="upper left", fontsize=6, framealpha=0.9, edgecolor="none")

    fig.tight_layout()
    save_fig(fig, fig_dir, "fig02_sprr_over_time")


# =============================================================================
# FIGURE 2: Theme Distribution (TPR) — Horizontal Bar
# =============================================================================
def fig05_theme_distribution(privacy_reviews, fig_dir):
    theme_counts = privacy_reviews["primary_theme"].value_counts()
    total = len(privacy_reviews)

    # Exclude "unassigned"
    theme_counts = theme_counts[theme_counts.index != "unassigned"]
    ptpr = (theme_counts / total * 100).head(12).sort_values(ascending=True)

    # Color each bar by its taxonomy category
    colors = [CATEGORY_COLORS.get(THEME_TO_CATEGORY.get(t, ""), "#BBBBBB") for t in ptpr.index]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 3.0))
    bars = ax.barh(range(len(ptpr)), ptpr.values, color=colors,
                   edgecolor="white", lw=0.5, height=0.7)

    ax.set_yticks(range(len(ptpr)))
    ax.set_yticklabels(ptpr.index, fontsize=6)
    ax.set_xlabel("TPR (%)")
    ax.set_title("Top Security and Privacy Themes (TPR)", fontweight="bold", pad=6)

    # Value labels to the right of bars
    for bar, val in zip(bars, ptpr.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=5.5)

    ax.set_xlim(0, ptpr.max() * 1.18)

    # Category legend
    legend_handles = [Patch(facecolor=CATEGORY_COLORS[c], edgecolor="white",
                            linewidth=0.5, label=c) for c in CATEGORY_COLORS]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=5,
              framealpha=0.9, edgecolor="none", handlelength=1.0, handletextpad=0.3)

    fig.tight_layout(rect=[0.02, 0, 1, 1])
    save_fig(fig, fig_dir, "fig05_theme_distribution")


# =============================================================================
# FIGURE 3: Emotions by Theme — Stacked Horizontal Bar
# =============================================================================
def fig08_affect_by_theme(privacy_reviews, fig_dir):
    top_themes = privacy_reviews["primary_theme"].value_counts()
    top_themes = top_themes[top_themes.index != "unassigned"].head(8).index.tolist()

    emotion_order = ["Anger", "Disgust", "Fear", "Sadness", "Confusion",
                     "Neutral", "Joy", "Surprise"]

    data = {}
    for theme in top_themes:
        subset = privacy_reviews[privacy_reviews["primary_theme"] == theme]
        emo_dist = subset["emotion_group"].value_counts(normalize=True) * 100
        data[theme] = {e: emo_dist.get(e, 0) for e in emotion_order}

    df = pd.DataFrame(data).T[emotion_order]

    # Drawn as Fig. 8 appears in the paper: sans-serif, dotted vertical grid,
    # one-row legend below the axis.
    with plt.style.context("default"):
        plt.rcParams.update({"font.family": "sans-serif",
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(figsize=(7.2, 5.4))

        left = np.zeros(len(df))
        for emotion in emotion_order:
            ax.barh(range(len(df)), df[emotion].values, left=left,
                    color=EMOTION_COLORS[emotion], label=emotion, height=0.62)
            left += df[emotion].values

        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.index, fontsize=11)
        ax.tick_params(axis="x", labelsize=11)
        ax.set_xlabel("Percentage (%)", fontsize=13)
        ax.set_xlim(0, 100)
        ax.grid(axis="x", linestyle=":", color="#BBBBBB", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title("Affective-Expression Distribution by Theme",
                     fontweight="bold", fontsize=16, pad=10)

        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.11),
                  ncol=8, frameon=False, fontsize=9,
                  handlelength=1.6, handletextpad=0.5, columnspacing=1.6)

        fig.tight_layout()
        save_fig(fig, fig_dir, "fig08_affect_by_theme")


# =============================================================================
# FIGURE 4: SPRR by Bot Tag — Horizontal Bar
# =============================================================================
def fig06_sprr_by_tag(all_reviews, fig_dir, min_reviews=100):
    df = all_reviews.copy()

    df["tags_list"] = df["tags"].apply(
        lambda x: [t.strip().strip("[]'\"") for t in str(x).split(",")]
        if pd.notna(x) else []
    )
    df = df.explode("tags_list")
    df = df[df["tags_list"].str.len() > 0]

    tag_stats = df.groupby("tags_list").agg(
        total=("is_privacy", "count"),
        privacy=("is_privacy", "sum"),
    )
    tag_stats["sprr"] = tag_stats["privacy"] / tag_stats["total"] * 100
    tag_stats = tag_stats[tag_stats["total"] >= min_reviews].nlargest(15, "sprr")
    tag_stats = tag_stats.sort_values("sprr", ascending=True)
    overall_sprr = all_reviews["is_privacy"].mean() * 100

    # Drawn as Fig. 6 appears in the paper: sans-serif, with a dashed line at
    # the ecosystem-wide SPRR (4.82%).
    with plt.style.context("default"):
        plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        bars = ax.barh(range(len(tag_stats)), tag_stats["sprr"].values,
                       color=WONG["vermilion"], height=0.65)
        ax.axvline(overall_sprr, color="black", linestyle="--", lw=1.2)

        ax.set_yticks(range(len(tag_stats)))
        ax.set_yticklabels(tag_stats.index, fontsize=11)
        ax.tick_params(axis="x", labelsize=11)
        ax.set_xlabel("SPRR (%)", fontsize=12)
        ax.set_title(f"SPRR by Bot Tag (min. {min_reviews} reviews)", fontsize=13)
        ax.grid(axis="x", color="#DDDDDD", lw=0.8)
        ax.set_axisbelow(True)

        for bar, (_, row) in zip(bars, tag_stats.iterrows()):
            label = f"{row['sprr']:.1f}% ({int(row['privacy'])}/{int(row['total'])})"
            ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=10, color="#333333")

        ax.set_xlim(0, 72)
        fig.tight_layout()
        save_fig(fig, fig_dir, "fig06_sprr_by_tag")


# =============================================================================
# FIGURE 5: Themes per Main Bot Category — Stacked Horizontal Bar
# =============================================================================
def fig07_themes_by_category(privacy_reviews, fig_dir):
    pr = privacy_reviews.copy()
    pr["main_tag"] = pr["tags"].apply(get_main_tag)

    top_themes = pr["primary_theme"].value_counts()
    top_themes = top_themes[top_themes.index != "unassigned"].head(8).index.tolist()
    top_cats = pr["main_tag"].value_counts().head(10).index.tolist()

    matrix = []
    for cat in top_cats:
        subset = pr[pr["main_tag"] == cat]
        total = len(subset)
        row = {}
        for theme in top_themes:
            row[theme] = (subset["primary_theme"] == theme).sum() / total * 100 if total > 0 else 0
        matrix.append(row)

    df = pd.DataFrame(matrix, index=top_cats)[top_themes]

    # Use Wong cycle for theme colors
    theme_colors = WONG_CYCLE[:len(top_themes)]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 4.2))

    left = np.zeros(len(df))
    for i, theme in enumerate(top_themes):
        ax.barh(range(len(df)), df[theme].values, left=left,
                color=theme_colors[i], label=theme,
                edgecolor="white", lw=0.3, height=0.65)
        left += df[theme].values

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=6)
    ax.set_xlabel("Percentage of Security and Privacy Reviews (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Top Security and Privacy Themes by Bot Category", fontweight="bold", pad=6)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, frameon=False, fontsize=5,
              handlelength=1.0, handletextpad=0.3, columnspacing=0.6)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    save_fig(fig, fig_dir, "fig07_themes_by_category")


# =============================================================================
# FIGURE 6: Emotions by Bot Category — Stacked Horizontal Bar
# =============================================================================
def fig09_affect_by_category(privacy_reviews, fig_dir):
    pr = privacy_reviews.copy()
    pr["main_tag"] = pr["tags"].apply(get_main_tag)

    emotion_order = ["Anger", "Disgust", "Fear", "Sadness", "Confusion",
                     "Neutral", "Joy", "Surprise"]
    neg_emotions = ["Anger", "Disgust", "Fear", "Sadness"]

    top_cats = pr["main_tag"].value_counts().head(15).index.tolist()

    cat_neg = {}
    data = {}
    for cat in top_cats:
        subset = pr[pr["main_tag"] == cat]
        if len(subset) < 10:
            continue
        emo_dist = subset["emotion_group"].value_counts(normalize=True) * 100
        row = {e: emo_dist.get(e, 0) for e in emotion_order}
        data[cat] = row
        cat_neg[cat] = sum(row[e] for e in neg_emotions)

    sorted_cats = sorted(data.keys(), key=lambda c: cat_neg[c])
    df = pd.DataFrame(data).T.loc[sorted_cats][emotion_order]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 4.2))

    left = np.zeros(len(df))
    for emotion in emotion_order:
        ax.barh(range(len(df)), df[emotion].values, left=left,
                color=EMOTION_COLORS[emotion], label=emotion,
                edgecolor="white", lw=0.3, height=0.65)
        left += df[emotion].values

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=5.5)
    ax.set_xlabel("Percentage (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Affective-Expression Distribution by Bot Category",
                 fontweight="bold", fontsize=9, pad=12)
    ax.text(0.5, 1.02, "(ordered by negative %)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6, style="italic")

    # Legend order as printed in Fig. 9: Anger, Fear, Confusion, Joy, Neutral /
    # Surprise, Disgust, Sadness (matplotlib fills legend columns top to bottom).
    handles = dict(zip(*ax.get_legend_handles_labels()[::-1]))
    legend_order = ["Anger", "Surprise", "Fear", "Disgust", "Confusion",
                    "Sadness", "Joy", "Neutral"]
    ax.legend([handles[e] for e in legend_order], legend_order,
              loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=5, frameon=False, fontsize=5.5,
              handlelength=1.0, handletextpad=0.3, columnspacing=0.8)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    save_fig(fig, fig_dir, "fig09_affect_by_category")


# =============================================================================
# FIGURE 7: Taxonomy Category Breakdown — Horizontal Bar
# =============================================================================
def fig04_theme_groups(privacy_reviews, fig_dir):
    pr = privacy_reviews.copy()
    pr["tax_cat"] = pr["primary_theme"].map(THEME_TO_CATEGORY).fillna("Other")

    cat_counts = pr["tax_cat"].value_counts()
    total = len(pr)
    cat_pct = (cat_counts / total * 100)

    # Ensure all 4 categories appear even if count is 0
    all_cats = ["Data Practices", "Security & Safety",
                "User Rights & Controls", "Discord-Specific Risks"]
    # Add Other only if it exists and is > 0
    if "Other" in cat_pct and cat_pct["Other"] > 0.5:
        all_cats.append("Other")

    cat_pct = cat_pct.reindex(all_cats, fill_value=0).sort_values(ascending=True)

    color_map = {**CATEGORY_COLORS, "Other": "#BBBBBB"}
    colors = [color_map.get(c, "#BBBBBB") for c in cat_pct.index]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.2))
    bars = ax.barh(range(len(cat_pct)), cat_pct.values, color=colors,
                   edgecolor="white", lw=0.5, height=0.6)

    ax.set_yticks(range(len(cat_pct)))
    ax.set_yticklabels(cat_pct.index, fontsize=6.5)
    ax.set_xlabel("Percentage of Security and Privacy Reviews (%)")
    ax.set_title("Security and Privacy Reviews by Theme Group", fontweight="bold", pad=6)

    # Labels: inside only for very large bars (>25%), outside for everything else
    for i, (bar, (cat, pct)) in enumerate(zip(bars, cat_pct.items())):
        n = cat_counts.get(cat, 0)
        label = f"{pct:.1f}% (n={n:,})"
        if pct > 25:
            ax.text(bar.get_width() * 0.5, bar.get_y() + bar.get_height() / 2,
                    label, ha="center", va="center", fontsize=6, fontweight="bold",
                    color="white")
        else:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    label, ha="left", va="center", fontsize=5.5, fontweight="bold",
                    color="#334155")

    ax.set_xlim(0, cat_pct.max() * 1.25)
    fig.tight_layout()
    save_fig(fig, fig_dir, "fig04_theme_groups")


# =============================================================================
# FIGURE 8: Star Rating Comparison
# =============================================================================
def fig03_star_rating(all_reviews, privacy_reviews, fig_dir):
    ratings = [1, 2, 3, 4, 5]

    def star_pct(df):
        """Percentage of 1-5 star reviews, normalised to sum to 100.

        "stars" arrives as an object column because a small number of rows carry
        malformed values, so it must be coerced before counting; otherwise a
        label lookup on the resulting index silently falls back to positional
        lookup. Ratings outside 1-5 (including 0-star and unparseable values)
        are excluded so each series sums to 100.
        """
        s = pd.to_numeric(df["stars"], errors="coerce")
        s = s[s.between(1, 5)].astype(int)
        return s.value_counts(normalize=True).reindex(ratings, fill_value=0) * 100

    p_stars = star_pct(privacy_reviews)
    n_stars = star_pct(all_reviews[all_reviews["is_privacy"] == 0])
    x = np.arange(len(ratings))
    width = 0.35

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.2))

    non_priv_vals = [float(n_stars[r]) for r in ratings]
    priv_vals = [float(p_stars[r]) for r in ratings]

    bars_np = ax.bar(x - width / 2, non_priv_vals, width,
                     label="Non-S&P", color=WONG["sky_blue"],
                     edgecolor="white", lw=0.5)
    bars_p = ax.bar(x + width / 2, priv_vals, width,
                    label="S&P", color=WONG["vermilion"],
                    edgecolor="white", lw=0.5)

    # Data labels on all bars
    for bar in bars_np:
        h = bar.get_height()
        if h > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=4.5)
    for bar in bars_p:
        h = bar.get_height()
        if h > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=4.5)

    ax.set_ylabel("Reviews within group (%)")
    ax.set_xlabel("Star Rating")
    ax.set_xticks(x)
    ax.set_xticklabels(["1-star", "2-star", "3-star", "4-star", "5-star"], fontsize=6.5)
    ax.set_title("Star Rating Distribution: S&P vs. Non-S&P",
                 fontweight="bold", pad=6)

    ax.legend(loc="upper center", fontsize=6, framealpha=0.9, edgecolor="none")

    fig.tight_layout()
    save_fig(fig, fig_dir, "fig03_star_rating")


# =============================================================================
# DATE PARSING HELPER
# =============================================================================
EXCEL_EPOCH = pd.Timestamp("1899-12-30")

def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse dates that may be Excel serial numbers or standard date strings."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid_numeric = numeric.dropna()
    if len(valid_numeric) > 0 and valid_numeric.median() > 20000:
        return EXCEL_EPOCH + pd.to_timedelta(numeric, unit="D")
    return pd.to_datetime(series, errors="coerce")


# =============================================================================
# MAIN
# =============================================================================
def main():
    data_dir = Path("data")
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_reviews = pd.read_csv(
        data_dir / "processed" / "all_reviews_classified.csv",
        low_memory=False,
    )
    all_reviews["date"] = _parse_dates(all_reviews["date"])

    privacy_reviews = pd.read_csv(
        data_dir / "processed" / "privacy_reviews_full.csv",
        low_memory=False,
    )
    privacy_reviews["date"] = _parse_dates(privacy_reviews["date"])

    print(f"Loaded {len(all_reviews):,} total reviews, {len(privacy_reviews):,} privacy reviews\n")

    # Print theme→category mapping check
    mapped = privacy_reviews["primary_theme"].map(THEME_TO_CATEGORY)
    n_mapped = mapped.notna().sum()
    n_total = len(privacy_reviews)
    print(f"Theme mapping check: {n_mapped:,}/{n_total:,} reviews mapped "
          f"({n_mapped/n_total*100:.1f}%), {n_total - n_mapped:,} unmapped → 'Other'\n")

    # Show unmapped themes so we can fix them
    unmapped_themes = privacy_reviews.loc[mapped.isna(), "primary_theme"].value_counts()
    if len(unmapped_themes) > 0:
        print("  Unmapped themes (going to 'Other'):")
        for theme, count in unmapped_themes.head(10).items():
            print(f"    {theme}: {count}")
        print()

    print("Fig. 2: SPRR Over Time")
    fig02_sprr_over_time(all_reviews, privacy_reviews, fig_dir)

    print("Fig. 3: Star Rating Comparison")
    fig03_star_rating(all_reviews, privacy_reviews, fig_dir)

    print("Fig. 4: Theme Group Breakdown")
    fig04_theme_groups(privacy_reviews, fig_dir)

    print("Fig. 5: Theme Distribution (TPR)")
    fig05_theme_distribution(privacy_reviews, fig_dir)

    print("Fig. 6: SPRR by Bot Tag")
    fig06_sprr_by_tag(all_reviews, fig_dir, min_reviews=100)

    print("Fig. 7: Themes by Bot Category")
    fig07_themes_by_category(privacy_reviews, fig_dir)

    print("Fig. 8: Affective Expressions by Theme")
    fig08_affect_by_theme(privacy_reviews, fig_dir)

    print("Fig. 9: Affective Expressions by Bot Category")
    fig09_affect_by_category(privacy_reviews, fig_dir)

    print(f"\nAll 8 figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
