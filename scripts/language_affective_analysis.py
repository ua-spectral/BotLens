"""
Affective-Expression Distribution Across Languages
===================================================

Uses privacy_reviews_full.csv to analyze affective-expression
patterns across detected languages for S&P-relevant reviews.

Only languages with at least 100 S&P reviews are included
in the final figure.

Outputs (paper Appendix B, Fig. 17):
    figures/fig17_affect_by_language.png
    data/results/language_affective_distribution.csv
    data/results/language_affective_counts.csv
    data/results/language_affective_raw_counts.csv
"""

# Windows consoles default to cp1252; bot names and review text are not.
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_PATH = Path("data/processed/privacy_reviews_full.csv")

FIGURE_DIR = Path("figures")
RESULTS_DIR = Path("data/results")

FIGURE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------

MIN_SP_REVIEWS = 100

# Stack order matches the affective-expression figures for
# themes and bot categories.
EMOTION_ORDER = [
    "Anger",
    "Disgust",
    "Fear",
    "Sadness",
    "Confusion",
    "Neutral",
    "Joy",
    "Surprise",
]

EMOTION_COLORS = {
    "Neutral": "#BBBBBB",
    "Anger": "#D55E00",
    "Disgust": "#CC79A7",
    "Sadness": "#0072B2",
    "Fear": "#E69F00",
    "Surprise": "#F0E442",
    "Joy": "#009E73",
    "Confusion": "#56B4E9",
}

LANGUAGE_NAMES = {
    "en": "English",
    "tr": "Turkish",
    "es": "Spanish",
    "pt": "Portuguese",
    "so": "Somali",
    "af": "Afrikaans",
    "fr": "French",
    "de": "German",
    "ar": "Arabic",
    "ro": "Romanian",
    "nl": "Dutch",
    "it": "Italian",
    "cy": "Welsh",
    "tl": "Filipino",
    "sl": "Slovenian",
    "ca": "Catalan",
    "pl": "Polish",
    "no": "Norwegian",
    "sw": "Swahili",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "da": "Danish",
    "hr": "Croatian",
    "cs": "Czech",
    "et": "Estonian",
    "ru": "Russian",
}


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

def load_data():
    print(f"Loading: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False
    )

    required_columns = [
        "detected_lang",
        "is_privacy",
        "emotion_group",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(f"Total rows: {len(df):,}")
    print(
        f"Detected languages: "
        f"{df['detected_lang'].nunique(dropna=True)}"
    )

    return df


# ---------------------------------------------------------------------
# Prepare S&P reviews
# ---------------------------------------------------------------------

def prepare_sp_reviews(df):
    priv = df[
        df["is_privacy"] == 1
    ].copy()

    print(
        f"\nS&P-relevant reviews: "
        f"{len(priv):,}"
    )

    priv = priv[
        priv["detected_lang"].notna()
        & priv["emotion_group"].notna()
    ].copy()

    priv["detected_lang"] = (
        priv["detected_lang"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    priv["emotion_group"] = (
        priv["emotion_group"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    print(
        "S&P reviews with language and "
        f"affective-expression labels: "
        f"{len(priv):,}"
    )

    return priv


# ---------------------------------------------------------------------
# Language counts
# ---------------------------------------------------------------------

def analyze_language_distribution(priv):
    counts = (
        priv["detected_lang"]
        .value_counts()
        .rename_axis("language_code")
        .reset_index(name="count")
    )

    counts["percentage"] = (
        counts["count"]
        / len(priv)
        * 100
    )

    counts["language"] = (
        counts["language_code"]
        .map(LANGUAGE_NAMES)
        .fillna(
            counts["language_code"]
            .str.upper()
        )
    )

    counts = counts[
        [
            "language_code",
            "language",
            "count",
            "percentage",
        ]
    ]

    output = (
        RESULTS_DIR
        / "language_affective_counts.csv"
    )

    counts.to_csv(
        output,
        index=False
    )

    print(
        "\nLanguages among S&P reviews:"
    )

    print(
        counts.head(20).to_string(
            index=False,
            formatters={
                "percentage":
                    "{:.2f}%".format
            },
        )
    )

    print(f"\nSaved: {output}")

    return counts


# ---------------------------------------------------------------------
# Affective-expression analysis by language
# ---------------------------------------------------------------------

def analyze_affective_distribution(
    priv,
    language_counts
):
    eligible_codes = (
        language_counts[
            language_counts["count"]
            >= MIN_SP_REVIEWS
        ]["language_code"]
        .tolist()
    )

    print(
        f"\nLanguages with at least "
        f"{MIN_SP_REVIEWS} S&P reviews:"
    )

    eligible_table = (
        language_counts[
            language_counts["language_code"]
            .isin(eligible_codes)
        ]
    )

    print(
        eligible_table[
            [
                "language",
                "count",
                "percentage",
            ]
        ].to_string(
            index=False,
            formatters={
                "percentage":
                    "{:.2f}%".format
            },
        )
    )

    top_priv = priv[
        priv["detected_lang"]
        .isin(eligible_codes)
    ].copy()

    raw_counts = (
        top_priv
        .groupby(
            [
                "detected_lang",
                "emotion_group",
            ]
        )
        .size()
        .unstack(fill_value=0)
    )

    for emotion in EMOTION_ORDER:
        if emotion not in raw_counts.columns:
            raw_counts[emotion] = 0

    raw_counts = raw_counts[
        EMOTION_ORDER
    ]

    language_order = (
        top_priv["detected_lang"]
        .value_counts()
        .index
        .tolist()
    )

    raw_counts = raw_counts.loc[
        language_order
    ]

    percentages = (
        raw_counts
        .div(
            raw_counts.sum(axis=1),
            axis=0
        )
        * 100
    )

    language_labels = [
        LANGUAGE_NAMES.get(
            code,
            code.upper()
        )
        for code in percentages.index
    ]

    percentages.index = (
        language_labels
    )

    raw_counts.index = (
        language_labels
    )

    percentage_output = (
        RESULTS_DIR
        / "language_affective_distribution.csv"
    )

    percentages.round(3).to_csv(
        percentage_output
    )

    raw_output = (
        RESULTS_DIR
        / "language_affective_raw_counts.csv"
    )

    raw_counts.to_csv(
        raw_output
    )

    print(
        "\nAffective-expression "
        "distribution by language (%):"
    )

    print(
        percentages
        .round(1)
        .to_string()
    )

    print(
        f"\nSaved: "
        f"{percentage_output}"
    )

    print(
        f"Saved: "
        f"{raw_output}"
    )

    # -----------------------------------------------------------------
    # Coverage reporting for the paper text and caption
    # -----------------------------------------------------------------

    covered = int(
        raw_counts.sum().sum()
    )

    total_sp = len(priv)

    excluded = total_sp - covered

    print(
        f"\nCoverage: {covered:,} of "
        f"{total_sp:,} S&P reviews with "
        f"affective-expression labels "
        f"({covered / total_sp * 100:.1f}%) "
        f"across {len(raw_counts)} languages "
        f"with at least {MIN_SP_REVIEWS} "
        f"S&P reviews."
    )

    print(
        f"Excluded: {excluded:,} reviews "
        f"({excluded / total_sp * 100:.1f}%)."
    )

    return (
        percentages,
        raw_counts
    )


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

def create_figure(
    percentages,
    raw_counts
):
    languages = (
        percentages.index.tolist()
    )

    positions = np.arange(
        len(languages)
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 4.2)
    )

    lefts = np.zeros(
        len(languages)
    )

    for emotion in EMOTION_ORDER:
        values = (
            percentages[
                emotion
            ].values
        )

        ax.barh(
            positions,
            values,
            left=lefts,
            color=EMOTION_COLORS[
                emotion
            ],
            edgecolor="white",
            linewidth=0.6,
            height=0.65,
            label=emotion,
        )

        lefts += values

    labels = []

    for language in languages:
        n = int(
            raw_counts
            .loc[language]
            .sum()
        )

        labels.append(
            f"{language}\n"
            f"(n={n:,})"
        )

    ax.set_yticks(positions)

    ax.set_yticklabels(
        labels,
        fontsize=8,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Percentage (%)"
    )

    ax.set_ylabel("")

    ax.set_title(
        "Affective-Expression "
        "Distribution by Detected "
        "Language"
    )

    ax.set_xlim(
        0,
        100
    )

    ax.xaxis.set_major_formatter(
        mticker.FormatStrFormatter(
            "%g"
        )
    )

    handles = [
        Patch(
            facecolor=
                EMOTION_COLORS[e],
            label=e,
        )
        for e in EMOTION_ORDER
    ]

    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.16
        ),
        ncol=8,
        fontsize=7,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
    )

    fig.tight_layout()

    fig.subplots_adjust(
        bottom=0.22,
        top=0.90,
    )

    output = (
        FIGURE_DIR
        / "fig17_affect_by_language.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved figure: "
        f"{output}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    df = load_data()

    priv = (
        prepare_sp_reviews(df)
    )

    language_counts = (
        analyze_language_distribution(
            priv
        )
    )

    percentages, raw_counts = (
        analyze_affective_distribution(
            priv,
            language_counts,
        )
    )

    create_figure(
        percentages,
        raw_counts,
    )

    print(
        "\nAnalysis complete."
    )


if __name__ == "__main__":
    main()