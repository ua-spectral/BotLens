


"""
Stage 0 — Data Preprocessing, Merge & Translation

Takes the raw review CSV and bot metadata CSV, merges them on bot_id,
runs language detection, and translates non-English reviews to English
using Argos Translate (local, no API key needed). Output is a single
combined CSV ready for the rest of the pipeline.

Usage:
    python scripts/stage0_preprocess.py \
        --reviews data/raw/bot_reviews_cleaned.csv \
        --metadata data/raw/bot_metadata_cleaned.csv

Deps: pip install langdetect argostranslate
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, PROCESSED_DIR


def detect_language(text: str) -> str:
    """Detect language of text. Returns ISO 639-1 code (e.g., 'en', 'vi', 'tr')."""
    from langdetect import detect, LangDetectException
    try:
        if not text or len(text.strip()) < 3:
            return "unknown"
        return detect(text)
    except LangDetectException:
        return "unknown"


def setup_argos_translate(source_langs: list):
    """Download and install Argos Translate language packages."""
    import argostranslate.package
    import argostranslate.translate

    print("  Checking available Argos Translate packages...")
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    installed_langs = set()
    for lang in source_langs:
        if lang == "en":
            continue
        # Find package for lang -> en
        matching = [
            p for p in available_packages
            if p.from_code == lang and p.to_code == "en"
        ]
        if matching:
            pkg = matching[0]
            # Check if already installed
            installed = argostranslate.package.get_installed_packages()
            already_installed = any(
                p.from_code == lang and p.to_code == "en" for p in installed
            )
            if not already_installed:
                print(f"    Installing: {lang} -> en")
                pkg.install()
            installed_langs.add(lang)
        else:
            print(f"    WARNING: No package available for {lang} -> en")

    return installed_langs


def translate_text(text: str, source_lang: str) -> str:
    """Translate text from source_lang to English using Argos Translate."""
    import argostranslate.translate

    if not text or source_lang == "en":
        return text

    try:
        translated = argostranslate.translate.translate(text, source_lang, "en")
        return translated if translated else text
    except Exception:
        return text


def preprocess(reviews_path: str, metadata_path: str, skip_translation: bool = False):
    """Merge reviews with bot metadata, detect language, translate, and prepare for pipeline."""

    print("[Stage 0] Loading data")
    reviews = pd.read_csv(reviews_path)
    metadata = pd.read_csv(metadata_path)

    print(f"  Reviews:  {len(reviews):,} rows, {len(reviews.columns)} columns")
    print(f"  Metadata: {len(metadata):,} rows, {len(metadata.columns)} columns")

    # =========================================================================
    # 1. Clean reviews
    # =========================================================================
    print("\n[Stage 0] Cleaning reviews")

    # Rename columns to pipeline format
    # NOTE: reviewer_name is intentionally NOT carried into the pipeline.
    # We analyze review text and bot metadata only; reviewer identifiers
    # are not used as analysis variables and are not retained.
    reviews = reviews.rename(columns={
        "content": "text",
        "rating": "stars",
        "calculated_date": "date",
        "bot_name": "name",
    })

    # Drop reviewer identifier columns if present in the source file
    for _id_col in ["reviewer_name", "username"]:
        if _id_col in reviews.columns:
            reviews = reviews.drop(columns=[_id_col])

    # Drop rows with empty review text
    before = len(reviews)
    reviews = reviews[reviews["text"].notna() & (reviews["text"].str.strip() != "")].copy()
    print(f"  Removed {before - len(reviews):,} empty reviews ({len(reviews):,} remaining)")

    # Clean text: strip whitespace, remove excessive newlines
    reviews["text"] = reviews["text"].str.strip().str.replace(r"\s+", " ", regex=True)

    # Filter very short reviews (< 5 chars) — too ambiguous to classify
    before = len(reviews)
    reviews = reviews[reviews["text"].str.len() >= 5].copy()
    print(f"  Removed {before - len(reviews):,} very short reviews (<5 chars)")

    # Ensure stars is numeric
    reviews["stars"] = pd.to_numeric(reviews["stars"], errors="coerce")

    # Parse dates — handle Excel serial numbers (e.g., 44971 = 2023-02-14)
    # Excel serial dates count days since 1899-12-30
    date_col = pd.to_numeric(reviews["date"], errors="coerce")
    excel_mask = date_col.notna() & (date_col > 20000) & (date_col < 60000)

    if excel_mask.sum() > len(reviews) * 0.5:
        # Majority are Excel serial numbers
        excel_epoch = pd.Timestamp("1899-12-30")
        reviews["date"] = excel_epoch + pd.to_timedelta(date_col, unit="D")
        print(f"  Dates: converted {excel_mask.sum():,} Excel serial numbers to datetime")
    else:
        # Standard date strings
        reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")

    valid_dates = reviews["date"].notna().sum()
    print(f"  Valid dates: {valid_dates:,}/{len(reviews):,}")
    if valid_dates > 0:
        print(f"  Date range: {reviews['date'].min().date()} to {reviews['date'].max().date()}")

    # Ensure bot_id is string
    reviews["bot_id"] = reviews["bot_id"].astype(str)

    # =========================================================================
    # 2. Language detection & translation
    # =========================================================================
    if not skip_translation:
        print("\n[Stage 0] Detecting languages")

        tqdm.pandas(desc="Detecting language")
        reviews["detected_lang"] = reviews["text"].progress_apply(detect_language)

        lang_dist = reviews["detected_lang"].value_counts()
        print(f"\n  Language distribution (top 15):")
        for lang, count in lang_dist.head(15).items():
            print(f"    {lang}: {count:,} ({count / len(reviews) * 100:.1f}%)")

        en_count = (reviews["detected_lang"] == "en").sum()
        non_en_count = (reviews["detected_lang"] != "en").sum()
        print(f"\n  English:     {en_count:,} ({en_count / len(reviews) * 100:.1f}%)")
        print(f"  Non-English: {non_en_count:,} ({non_en_count / len(reviews) * 100:.1f}%)")

        # Translate non-English reviews
        if non_en_count > 0:
            print(f"\n[Stage 0] Translating {non_en_count:,} non-English reviews with Argos Translate")

            # Get unique non-English languages (excluding unknown and very rare ones)
            non_en_langs = (
                reviews[~reviews["detected_lang"].isin(["en", "unknown"])]
                ["detected_lang"].value_counts()
            )
            # Only translate languages with >= 10 reviews (worth installing the package)
            langs_to_translate = non_en_langs[non_en_langs >= 10].index.tolist()
            print(f"  Languages to translate: {langs_to_translate}")

            # Install Argos packages
            installed_langs = setup_argos_translate(langs_to_translate)
            print(f"  Installed translation packages: {installed_langs}")

            # Store original text
            reviews["original_text"] = reviews["text"]

            # Translate
            non_en_mask = (
                reviews["detected_lang"].isin(installed_langs) &
                (reviews["detected_lang"] != "en")
            )
            non_en_reviews = reviews[non_en_mask]

            print(f"  Translating {len(non_en_reviews):,} reviews...")
            translated_texts = []
            for idx, row in tqdm(non_en_reviews.iterrows(), total=len(non_en_reviews), desc="Translating"):
                translated = translate_text(row["text"], row["detected_lang"])
                translated_texts.append(translated)

            reviews.loc[non_en_mask, "text"] = translated_texts
            reviews["was_translated"] = non_en_mask

            translated_count = non_en_mask.sum()
            print(f"  Translated: {translated_count:,} reviews")

            # Reviews in unsupported languages — keep original
            unsupported_mask = (
                ~reviews["detected_lang"].isin(installed_langs) &
                (reviews["detected_lang"] != "en") &
                (reviews["detected_lang"] != "unknown")
            )
            unsupported_count = unsupported_mask.sum()
            if unsupported_count > 0:
                print(f"  Unsupported languages (kept as-is): {unsupported_count:,}")
                reviews.loc[unsupported_mask, "was_translated"] = False
        else:
            reviews["original_text"] = reviews["text"]
            reviews["was_translated"] = False
    else:
        print("\n[Stage 0] Skipping translation (--skip-translation flag)")
        reviews["detected_lang"] = "unknown"
        reviews["original_text"] = reviews["text"]
        reviews["was_translated"] = False

    # =========================================================================
    # 3. Clean metadata
    # =========================================================================
    print("\n[Stage 0] Cleaning metadata")

    metadata["bot_id"] = metadata["bot_id"].astype(str)

    # Select useful metadata columns for merge
    meta_cols = [
        "bot_id", "short_description", "prefix", "server_count", "votes",
        "average_rating", "total_reviews", "tags", "languages", "creators",
        "long_description", "invite_url", "permissions", "permissions_decoded",
        "github_url", "github_url_extracted", "commands"
    ]
    meta_cols = [c for c in meta_cols if c in metadata.columns]
    metadata_slim = metadata[meta_cols].copy()

    # Deduplicate metadata by bot_id (keep first)
    metadata_slim = metadata_slim.drop_duplicates(subset="bot_id", keep="first")
    print(f"  Unique bots in metadata: {len(metadata_slim):,}")

    # =========================================================================
    # 4. Merge reviews with metadata
    # =========================================================================
    print("\n[Stage 0] Merging reviews with metadata")

    merged = reviews.merge(metadata_slim, on="bot_id", how="left")

    matched = merged["tags"].notna().sum()
    print(f"  Reviews matched with metadata: {matched:,}/{len(merged):,} ({matched / len(merged) * 100:.1f}%)")

    # =========================================================================
    # 5. Add derived columns
    # =========================================================================
    print("\n[Stage 0] Adding derived columns")

    # Review text length
    merged["text_length"] = merged["text"].str.len()

    # Word count
    merged["word_count"] = merged["text"].str.split().str.len()

    # Number of permissions
    if "permissions_decoded" in merged.columns:
        merged["num_permissions"] = (
            merged["permissions_decoded"]
            .fillna("")
            .str.split(",")
            .apply(lambda x: len([p for p in x if p.strip()]))
        )

    # Has GitHub (transparency signal)
    if "github_url_extracted" in merged.columns:
        merged["has_github"] = merged["github_url_extracted"].notna() & (merged["github_url_extracted"] != "")

    # =========================================================================
    # 6. Summary stats
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"  FINAL DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total reviews:     {len(merged):,}")
    print(f"  Unique bots:       {merged['bot_id'].nunique():,}")
    print(f"  Unique bots:       {merged['bot_id'].nunique():,}")
    print(f"  Date range:        {merged['date'].min()} to {merged['date'].max()}")

    print(f"\n  Star distribution:")
    star_dist = merged["stars"].value_counts().sort_index()
    for star, count in star_dist.items():
        print(f"    {star}★: {count:,} ({count / len(merged) * 100:.1f}%)")

    print(f"\n  Text length stats:")
    print(f"    Mean:   {merged['text_length'].mean():.0f} chars")
    print(f"    Median: {merged['text_length'].median():.0f} chars")

    if "detected_lang" in merged.columns:
        print(f"\n  Language stats:")
        print(f"    English:       {(merged['detected_lang'] == 'en').sum():,}")
        print(f"    Translated:    {merged['was_translated'].sum():,}")
        untranslated = ((merged['detected_lang'] != 'en') & (~merged['was_translated'])).sum()
        print(f"    Not translated: {untranslated:,}")

    if "tags" in merged.columns:
        print(f"\n  Top 10 bot categories:")
        tag_counts = (
            merged["tags"].fillna("")
            .str.split(",")
            .explode()
            .str.strip()
            .value_counts()
        )
        for tag, count in tag_counts.head(10).items():
            if tag:
                print(f"    {tag}: {count:,}")

    if "num_permissions" in merged.columns:
        print(f"\n  Permissions stats:")
        bot_perms = merged.groupby("bot_id")["num_permissions"].first()
        print(f"    Mean permissions per bot:   {bot_perms.mean():.1f}")
        print(f"    Median permissions per bot: {bot_perms.median():.0f}")

    # =========================================================================
    # 7. Save
    # =========================================================================
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / "reviews.csv"
    merged.to_csv(output_path, index=False)
    print(f"\n[Stage 0] COMPLETE.")
    print(f"  Merged dataset saved: {output_path}")
    print(f"  Total columns: {len(merged.columns)}")
    print(f"  Columns: {list(merged.columns)}")
    print(f"\n  Next step: python scripts/stage1_llm_labeling.py --input {output_path}")

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0: Preprocess, merge, and translate")
    parser.add_argument("--reviews", required=True, help="Path to bot_reviews_cleaned.csv")
    parser.add_argument("--metadata", required=True, help="Path to bot_metadata_cleaned.csv")
    parser.add_argument("--skip-translation", action="store_true", help="Skip language detection and translation")
    args = parser.parse_args()
    preprocess(args.reviews, args.metadata, args.skip_translation)