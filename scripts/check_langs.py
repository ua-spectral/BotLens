# Windows consoles default to cp1252; bot names and review text are not.
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

import pandas as pd

LANG_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian",
    "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
    "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "sq": "Albanian", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "tl": "Filipino/Tagalog", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "vi": "Vietnamese", "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)", "unknown": "Unknown",
}

df = pd.read_csv("data/raw/reviews.csv", low_memory=False, encoding="latin-1")

print("=" * 70)
print("LANGUAGE ANALYSIS OF DISCORD BOT REVIEWS")
print("=" * 70)
print(f"\nTotal reviews: {len(df):,}")
print(f"Unique languages detected: {df['detected_lang'].nunique()}")

en = (df["detected_lang"] == "en").sum()
non_en = len(df) - en
print(f"\nEnglish:     {en:,} ({round(en/len(df)*100, 1)}%)")
print(f"Non-English: {non_en:,} ({round(non_en/len(df)*100, 1)}%)")

print(f"\n{'Rank':<6} {'Code':<8} {'Language':<25} {'Count':>8} {'Pct':>8}  {'Med. Chars':>10}")
print("-" * 70)
for i, (lang, count) in enumerate(df["detected_lang"].value_counts().items(), 1):
    pct = round(count / len(df) * 100, 2)
    name = LANG_NAMES.get(lang, lang)
    median_len = df[df["detected_lang"] == lang]["text"].str.len().median()
    print(f"{i:<6} {lang:<8} {name:<25} {count:>8,} {pct:>7}%  {median_len:>10.0f}")