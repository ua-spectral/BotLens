"""
BotLens MCP Server
==================
A security and privacy intelligence tool for Discord server administrators
and researchers. Provides natural-language access to the BotLens dataset of
183,142 reviews across 23,000+ Discord bots scraped from Top.gg.

The server exposes the six tools described in the paper:
    1. get_bot_signals
    2. get_taxonomy_breakdown
    3. get_review_samples
    4. compare_bots
    5. list_high_salience_bots
    6. scan_server_bots

The tools follow Algorithm 1 (paper Appendix A): theme counts come from the
31-theme labels assigned to each S&P review (TaxonomyFlags), and a bot with
no reviews falls back to a metadata-only permission audit (PermissionAudit).
Every per-bot signal is precomputed when the server starts, so a tool call is
a dictionary lookup rather than a scan of the review table.

Additional internal helpers (get_permissions_audit, search_bots,
filter_bots_by_theme, get_dataset_stats, diagnose, get_timing_stats) are kept
in this file for development and benchmarking but are NOT registered as MCP
tools.

Usage:
    Set BOTLENS_DATA_DIR in claude_desktop_config.json to the directory that
    contains data/processed/. Data files read at startup:
        data/processed/all_reviews_classified.csv   (all reviews, S&P label)
        data/processed/privacy_reviews_full.csv     (S&P reviews, themes, sentiment)
        data/processed/bot_metadata_cleaned.csv     (bot metadata, for the fallback)
"""

# Windows consoles default to cp1252; bot names and review text are not.
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


# ---- Suppress warnings BEFORE any other imports ----
# FutureWarnings from pandas pollute stdout and corrupt MCP JSON-RPC transport
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ast
import os
import sys
import time
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from mcp.server.fastmcp import FastMCP

# config/taxonomy.py is the single source of truth for the 31 themes
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.taxonomy import ALL_THEMES, CATEGORIES  # noqa: E402

# Ensure stdout/stderr encoding is correct for MCP transport
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Suppress pandas silent downcasting warnings globally
pd.set_option('future.no_silent_downcasting', True)

# ---------------------------------------------------------------------------
# Server init & data loading
# ---------------------------------------------------------------------------

mcp = FastMCP("BotLens")

DATA_DIR = os.environ.get("BOTLENS_DATA_DIR", ".")
DATA_PATH = Path(DATA_DIR) / "data" / "processed" / "all_reviews_classified.csv"
SENTIMENT_PATH = Path(DATA_DIR) / "data" / "processed" / "privacy_reviews_full.csv"
METADATA_PATH = Path(DATA_DIR) / "data" / "processed" / "bot_metadata_cleaned.csv"

SENTIMENT_GROUPS = ("positive", "negative", "neutral")
SENTIMENT_EMPTY_MESSAGE = "No S&P-relevant reviews available for sentiment analysis."
SPRR_SALIENCE_NOTE = (
    "SPRR measures the salience of S&P discussion in user reviews; "
    "it is not a verified measure of bot safety or risk."
)

# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------

# In-memory store: tool_name -> list of elapsed seconds
_TIMING_LOG: dict = defaultdict(list)


def _record_time(tool_name: str, elapsed: float, arg_preview: str = "") -> None:
    """Log a single tool call duration to stderr and the in-memory store."""
    _TIMING_LOG[tool_name].append(elapsed)
    print(
        f"[BOTLENS TIMING] {tool_name}({arg_preview}): {elapsed:.4f}s",
        file=sys.stderr,
        flush=True,
    )


# ---------------------------------------------------------------------------
# Taxonomy & permission constants
# ---------------------------------------------------------------------------

THEME_CATEGORY = {theme: cat for cat, themes in CATEGORIES.items() for theme in themes}
_THEME_SET = set(ALL_THEMES)

# Permission risk explanations for admin-friendly output
PERMISSION_EXPLANATIONS = {
    "ADMINISTRATOR": "⛔ Full admin access — can do anything in the server",
    "MANAGE_GUILD": "⚠️ Can change server settings and name",
    "MANAGE_ROLES": "⚠️ Can create, edit, and assign roles to members",
    "MANAGE_CHANNELS": "⚠️ Can create, edit, and delete channels",
    "MANAGE_MESSAGES": "⚠️ Can delete or pin any message in any channel",
    "MANAGE_WEBHOOKS": "⚠️ Can create webhooks that send messages as other users",
    "KICK_MEMBERS": "⚠️ Can remove members from the server",
    "BAN_MEMBERS": "⚠️ Can permanently ban members",
    "MENTION_EVERYONE": "⚠️ Can ping @everyone and @here",
    "READ_MESSAGE_HISTORY": "📖 Can read all past messages in channels it has access to",
    "VIEW_CHANNEL": "👁️ Can view channels and their messages",
    "SEND_MESSAGES": "💬 Can send messages in channels",
    "VIEW_AUDIT_LOG": "🔍 Can view the full server audit log (all admin actions)",
    "VIEW_GUILD_INSIGHTS": "📊 Can view server analytics and member activity",
    "MANAGE_NICKNAMES": "✏️ Can change any member's nickname",
    "CONNECT": "🔊 Can join voice channels",
    "SPEAK": "🎙️ Can speak in voice channels",
    "MUTE_MEMBERS": "🔇 Can mute members in voice channels",
    "DEAFEN_MEMBERS": "🔇 Can deafen members in voice channels",
    "MOVE_MEMBERS": "🔀 Can move members between voice channels",
    "USE_APPLICATION_COMMANDS": "🤖 Can use slash commands",
    "SEND_MESSAGES_IN_THREADS": "💬 Can send messages in threads",
    "EMBED_LINKS": "🔗 Can embed links and rich content",
    "ATTACH_FILES": "📎 Can attach files to messages",
    "ADD_REACTIONS": "😀 Can add reactions to messages",
    "EXTERNAL_EMOJIS": "😎 Can use emojis from other servers",
}


def _salience_label(sprr: float) -> str:
    if sprr >= 10.0:
        return "🔴 HIGH SALIENCE"
    elif sprr >= 5.0:
        return "🟡 MEDIUM SALIENCE"
    elif sprr >= 2.0:
        return "🟢 LOW SALIENCE"
    else:
        return "✅ VERY LOW SALIENCE"


def _load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Check that BOTLENS_DATA_DIR is set correctly in claude_desktop_config.json."
        )
    df = pd.read_csv(DATA_PATH, low_memory=False, encoding="utf-8")
    df["is_privacy"] = pd.to_numeric(df["is_privacy"], errors="coerce").fillna(0).astype(int)

    # Defensively coerce numeric columns — silently fix bad types from CSV
    for col in ["stars", "server_count", "num_permissions", "votes", "total_reviews",
                "average_rating", "text_length", "word_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure has_github is boolean-safe (fixed: no FutureWarning)
    if "has_github" in df.columns:
        df["has_github"] = df["has_github"].fillna(False).infer_objects(copy=False).astype(bool)
    else:
        df["has_github"] = False

    # Ensure text column exists
    if "text" not in df.columns:
        df["text"] = ""

    # Drop rows with missing bot names — prevents groupby("name") issues
    df = df.dropna(subset=["name"])

    return df


def _safe_first(series):
    """Return first non-null value in a series, or None."""
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else None


def _build_bot_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-bot stats using explicit per-column operations only.
    Never calls .mean() or .agg() on the full dataframe — avoids pandas
    mixed-type aggregation crashes entirely.
    """
    rows = []
    str_cols = ["permissions_decoded", "short_description", "tags",
                "invite_url", "github_url"]
    num_cols = ["server_count", "num_permissions"]

    for name, grp in df.groupby("name"):
        row = {"name": name}
        row["total_reviews"] = len(grp)
        row["privacy_reviews"] = int(grp["is_privacy"].sum())

        if "stars" in grp.columns:
            stars_num = pd.to_numeric(grp["stars"], errors="coerce")
            row["avg_rating"] = float(stars_num.mean()) if stars_num.notna().any() else float("nan")
        else:
            row["avg_rating"] = float("nan")

        if "has_github" in grp.columns:
            row["has_github"] = bool(grp["has_github"].iloc[0])
        else:
            row["has_github"] = False

        for col in str_cols:
            if col in grp.columns:
                non_null = grp[col].dropna()
                row[col] = str(non_null.iloc[0]) if not non_null.empty else ""
            else:
                row[col] = ""

        for col in num_cols:
            if col in grp.columns:
                val = pd.to_numeric(grp[col].iloc[0], errors="coerce")
                row[col] = float(val) if pd.notna(val) else float("nan")
            else:
                row[col] = float("nan")

        rows.append(row)

    stats = pd.DataFrame(rows)
    stats["sprr"] = (stats["privacy_reviews"] / stats["total_reviews"] * 100).round(2)
    return stats


def _rounded_percentages(counts: dict[str, int]) -> dict[str, float]:
    """Return one-decimal percentages that add to exactly 100.0."""
    total = sum(counts.values())
    if total == 0:
        return {group: 0.0 for group in SENTIMENT_GROUPS}

    exact_tenths = {group: counts[group] * 1000 / total for group in SENTIMENT_GROUPS}
    floor_tenths = {group: int(exact_tenths[group]) for group in SENTIMENT_GROUPS}
    remainder = 1000 - sum(floor_tenths.values())
    order = sorted(
        SENTIMENT_GROUPS,
        key=lambda group: (exact_tenths[group] - floor_tenths[group], counts[group]),
        reverse=True,
    )
    for group in order[:remainder]:
        floor_tenths[group] += 1
    return {group: floor_tenths[group] / 10 for group in SENTIMENT_GROUPS}


def _load_sp_reviews() -> pd.DataFrame | None:
    """Read the S&P review table (themes and sentiment), or None if absent."""
    if not SENTIMENT_PATH.exists():
        return None
    return pd.read_csv(SENTIMENT_PATH, low_memory=False, encoding="utf-8")


def _build_theme_counts(df: pd.DataFrame | None) -> dict[str, dict]:
    """TaxonomyFlags(b, T, D_b) for every bot, precomputed.

    For each bot: the number of its S&P reviews assigned to each of the 31
    themes, sorted by frequency (descending), plus the bot's S&P review total.
    Each review counts under its primary theme only (paper Sec. III-D2: all
    quantitative analyses use exactly one primary theme per review); reviews
    labelled 'unassigned' count toward the total but toward no theme.
    """
    if df is None or "primary_theme" not in df.columns or "name" not in df.columns:
        return {}
    if "is_privacy" in df.columns:
        df = df[pd.to_numeric(df["is_privacy"], errors="coerce").fillna(0).astype(int) == 1]
    df = df.dropna(subset=["name"])

    counts: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for key, theme in zip(df["name"].astype(str), df["primary_theme"]):
        totals[key] += 1
        if theme in _THEME_SET:
            counts[key][theme] += 1

    return {
        key: {"total": totals[key],
              "themes": sorted(counts[key].items(), key=lambda x: (-x[1], x[0]))}
        for key in totals
    }


def _load_metadata() -> dict[str, dict]:
    """Bot metadata keyed by lowercased name, for the PermissionAudit fallback.

    Top.gg names are not unique; where several bots share a name, the one on
    the most servers is kept.
    """
    if not METADATA_PATH.exists():
        return {}
    cols = ["name", "permissions_decoded", "short_description", "server_count",
            "average_rating", "github_url"]
    meta = pd.read_csv(METADATA_PATH, low_memory=False, encoding="utf-8",
                       usecols=lambda c: c in cols)
    meta = meta.dropna(subset=["name"]).copy()
    for col in ("server_count", "average_rating"):
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta["_key"] = meta["name"].astype(str).str.lower()
    meta = meta.sort_values("server_count", ascending=False, na_position="last", kind="stable")
    meta = meta.drop_duplicates(subset="_key", keep="first")
    return {row["_key"]: row for row in meta.to_dict("records")}


def _parse_permissions(raw) -> list[str]:
    """Decode a permissions_decoded cell such as "['ADMINISTRATOR', 'KICK_MEMBERS']"."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if text in ("", "nan", "None", "[]"):
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(p).strip() for p in parsed if str(p).strip()]
    except (ValueError, SyntaxError):
        pass
    return [p.strip().strip("'\"") for p in text.strip("[]").split(",") if p.strip()]


def _permission_tier(perm: str) -> str:
    explanation = PERMISSION_EXPLANATIONS.get(perm.upper(), "")
    if "⛔" in explanation:
        return "critical"
    if "⚠️" in explanation:
        return "elevated"
    return "other"


def _permission_audit(bot_name: str) -> dict | None:
    """PermissionAudit(b) in Algorithm 1: the metadata-only fallback for a bot
    that has no reviews. Returns None if the bot is not in the metadata either."""
    meta = _METADATA.get(bot_name.lower().strip())
    if meta is None:
        return None
    perms = _parse_permissions(meta.get("permissions_decoded"))
    return {
        "name": meta["name"],
        "permissions": perms,
        "critical": [p for p in perms if _permission_tier(p) == "critical"],
        "elevated": [p for p in perms if _permission_tier(p) == "elevated"],
        "server_count": meta.get("server_count"),
        "short_description": meta.get("short_description"),
    }


def _audit_summary(audit: dict) -> str:
    if audit["critical"]:
        return f"⛔ Critical: {', '.join(audit['critical'])}"
    if audit["elevated"]:
        return f"⚠️ Elevated: {', '.join(audit['elevated'])}"
    if audit["permissions"]:
        return "No critical or elevated permissions requested."
    return "No permissions listed in metadata."


def _load_sentiment_stats(df: pd.DataFrame | None) -> tuple[dict[str, dict], str]:
    """Aggregate existing sentiment labels for S&P-relevant reviews only."""
    if df is None:
        return {}, f"Sentiment dataset not found at {SENTIMENT_PATH}."

    if "privacy_sentiment" not in df.columns:
        return {}, "Missing required sentiment field: privacy_sentiment."
    if "name" not in df.columns:
        return {}, "Missing required bot identifier field: name."

    if "is_privacy" in df.columns:
        relevant = pd.to_numeric(df["is_privacy"], errors="coerce").fillna(0).astype(int) == 1
        df = df[relevant].copy()

    df["_sentiment"] = df["privacy_sentiment"].astype("string").str.strip().str.casefold()
    df = df[df["_sentiment"].isin(SENTIMENT_GROUPS)].dropna(subset=["name"])

    stats: dict[str, dict] = {}
    for name, group in df.groupby(df["name"].astype(str)):
        value_counts = group["_sentiment"].value_counts()
        counts = {sentiment: int(value_counts.get(sentiment, 0)) for sentiment in SENTIMENT_GROUPS}
        percentages = _rounded_percentages(counts)
        stats[name] = {
            "counts": counts,
            "percentages": percentages,
            "total": sum(counts.values()),
        }
    return stats, ""


def _sentiment_for_bot(bot_name: str) -> dict:
    """Return sentiment composition for a bot's S&P-relevant reviews.

    bot_name is the exact (canonical) name returned by _find_bot."""
    result = _SENTIMENT_STATS.get(bot_name)
    if result is None or result["total"] == 0:
        return {
            "counts": {group: 0 for group in SENTIMENT_GROUPS},
            "percentages": {group: 0.0 for group in SENTIMENT_GROUPS},
            "total": 0,
            "message": SENTIMENT_EMPTY_MESSAGE,
        }
    return {**result, "message": ""}


def _sentiment_lines(sentiment: dict) -> str:
    if sentiment["total"] == 0:
        return sentiment["message"]
    counts = sentiment["counts"]
    percentages = sentiment["percentages"]
    return "\n".join(
        f"  {group.title()}: {percentages[group]:.1f}% (n={counts[group]})"
        for group in SENTIMENT_GROUPS
    )


def _compact_sentiment(sentiment: dict, include_counts: bool = True) -> str:
    if sentiment["total"] == 0:
        return SENTIMENT_EMPTY_MESSAGE
    counts = sentiment["counts"]
    percentages = sentiment["percentages"]
    pieces = []
    for group, short in (("positive", "Pos"), ("negative", "Neg"), ("neutral", "Neu")):
        value = f"{short} {percentages[group]:.1f}%"
        if include_counts:
            value += f" (n={counts[group]})"
        pieces.append(value)
    return " | ".join(pieces)


def _build_sp_review_index(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each bot's S&P reviews with text, keyed by exact name, in file order."""
    sp = df[df["is_privacy"] == 1].dropna(subset=["text"])
    return {key: grp for key, grp in sp.groupby("name", sort=False)}


# Load once at module level (restart Claude Desktop after any server.py change)
try:
    _DF = _load_data()
    _BOT_STATS = _build_bot_stats(_DF)
    # Case-insensitive name -> row position; the first match wins, as before.
    _BOT_INDEX: dict[str, int] = {}
    for _pos, _name in enumerate(_BOT_STATS["name"].str.lower()):
        _BOT_INDEX.setdefault(_name, _pos)
    _SP_REVIEWS = _build_sp_review_index(_DF)
    _SP_DF = _load_sp_reviews()
    _SENTIMENT_STATS, _SENTIMENT_ERROR = _load_sentiment_stats(_SP_DF)
    _THEME_COUNTS = _build_theme_counts(_SP_DF)
    del _SP_DF
    _METADATA = _load_metadata()
    _DATA_LOADED = True
    _LOAD_ERROR = ""
except Exception as e:
    _DATA_LOADED = False
    _LOAD_ERROR = str(e)
    import traceback
    _LOAD_ERROR = traceback.format_exc()
    _DF = pd.DataFrame()
    _BOT_STATS = pd.DataFrame(columns=["name", "sprr", "total_reviews", "privacy_reviews"])
    _BOT_INDEX = {}
    _SP_REVIEWS = {}
    _SENTIMENT_STATS = {}
    _SENTIMENT_ERROR = ""
    _THEME_COUNTS = {}
    _METADATA = {}

_NO_SP_REVIEWS = pd.DataFrame(columns=_DF.columns)


def _find_bot(name: str) -> pd.DataFrame:
    """Case-insensitive bot name lookup (one row, or empty).

    Top.gg names are not unique and some differ only in case; the row found
    here fixes the bot's exact name, and every other per-bot signal (S&P
    reviews, themes, sentiment) is looked up by that exact name so a report
    never mixes two bots."""
    pos = _BOT_INDEX.get(name.lower().strip())
    return _BOT_STATS.iloc[[pos]] if pos is not None else _BOT_STATS.iloc[0:0]


def _sp_reviews_for(canonical_name: str) -> pd.DataFrame:
    return _SP_REVIEWS.get(canonical_name, _NO_SP_REVIEWS)


def _top_themes_text(canonical_name: str, k: int = 3) -> str:
    entry = _THEME_COUNTS.get(canonical_name)
    if not entry or not entry["themes"]:
        return "—"
    return ", ".join(theme for theme, _ in entry["themes"][:k])


# ===========================================================================
# PUBLIC MCP TOOLS — These six are the tools described in the paper.
# ===========================================================================

# ---------------------------------------------------------------------------
# Tool 1 — get_bot_signals
# ---------------------------------------------------------------------------

@mcp.tool()
def get_bot_signals(bot_name: str) -> str:
    """
    Return review-derived S&P signals for a Discord bot.
    Returns the existing salience label, SPRR score, review count, permissions
    summary, top S&P themes, and three-group sentiment composition.

    Args:
        bot_name: The Top.gg slug name of the bot (e.g. 'wick', 'carl-bot')
    """
    _t0 = time.perf_counter()

    if not _DATA_LOADED:
        _record_time("get_bot_signals", time.perf_counter() - _t0, bot_name)
        return f"❌ Dataset not loaded: {_LOAD_ERROR}"

    row = _find_bot(bot_name)
    if row.empty:
        audit = _permission_audit(bot_name)
        _record_time("get_bot_signals", time.perf_counter() - _t0, bot_name)
        if audit is None:
            return (
                f"❌ Bot '{bot_name}' not found in the BotLens dataset.\n"
                "Try using the exact Top.gg slug (lowercase, hyphens)."
            )
        return _format_permission_audit(audit)

    r = row.iloc[0]
    sprr = r["sprr"]
    salience = _salience_label(sprr)
    total = int(r["total_reviews"])
    privacy = int(r["privacy_reviews"])
    rating = f"{r['avg_rating']:.1f}/5" if pd.notna(r["avg_rating"]) else "N/A"
    servers = f"{int(r['server_count']):,}" if pd.notna(r["server_count"]) else "Unknown"
    desc = r["short_description"] if pd.notna(r["short_description"]) else "No description available."
    perms = r["permissions_decoded"] if pd.notna(r["permissions_decoded"]) else "None listed"
    gh = "Yes" if r["has_github"] else "No"
    sentiment = _sentiment_for_bot(r["name"])
    sentiment_text = _sentiment_lines(sentiment)
    top_themes = _THEME_COUNTS.get(r["name"], {}).get("themes", [])[:3]
    if top_themes:
        themes_text = ", ".join(f"{theme} (n={count})" for theme, count in top_themes)
    else:
        themes_text = "No S&P themes available."

    if sprr >= 10.0:
        recommendation = (
            "🚨 Recommendation: Review this bot carefully before adding it to your server. "
            "A significant portion of users have raised security and privacy concerns. "
            "Consider whether all requested permissions are necessary."
        )
    elif sprr >= 5.0:
        recommendation = (
            "⚠️ Recommendation: This bot has a moderate number of security and privacy complaints. "
            "Check which permissions it requests and limit its channel access where possible."
        )
    elif sprr >= 2.0:
        recommendation = (
            "ℹ️ Recommendation: Low security and privacy concern. Standard precautions apply — "
            "review permissions and only grant access to channels it needs."
        )
    else:
        recommendation = (
            "ℹ️ Recommendation: Few user reviews raise security or privacy concerns. "
            "A low SPRR does not show that a bot is safe; still check which permissions "
            "it requests."
        )

    result = f"""
🤖 BotLens S&P Signal Report: {r['name']}
{'='*50}
{salience}
Security & Privacy Review Rate (SPRR): {sprr}% ({privacy} of {total} reviews raise security or privacy concerns)
{SPRR_SALIENCE_NOTE}
Average Rating: {rating}
Installed on ~{servers} servers
Open Source: {gh}

📝 Description:
{desc}

🔐 Permissions Requested:
{perms}

Top Reported S&P Themes:
{themes_text}

S&P Review Sentiment
{sentiment_text}
Sentiment describes the composition of S&P-relevant review text; it is not a bot safety or risk assessment.

{recommendation}

💡 For more details use:
  - get_taxonomy_breakdown('{bot_name}') — theme-level breakdown of its S&P reviews
  - get_review_samples('{bot_name}') — read actual user security and privacy complaints
"""
    _record_time("get_bot_signals", time.perf_counter() - _t0, bot_name)
    return result


# ---------------------------------------------------------------------------
# Tool 2 — get_taxonomy_breakdown
# ---------------------------------------------------------------------------

@mcp.tool()
def get_taxonomy_breakdown(bot_name: str) -> str:
    """
    Frequency distribution of the 31 taxonomy themes in a bot's security
    and privacy reviews: for each theme, how many of the bot's S&P reviews
    are assigned to it (TaxonomyFlags, Algorithm 1). Each review counts
    under its primary theme.

    Args:
        bot_name: The Top.gg slug name of the bot
    """
    _t0 = time.perf_counter()

    if not _DATA_LOADED:
        _record_time("get_taxonomy_breakdown", time.perf_counter() - _t0, bot_name)
        return f"❌ Dataset not loaded: {_LOAD_ERROR}"

    row = _find_bot(bot_name)
    if row.empty:
        _record_time("get_taxonomy_breakdown", time.perf_counter() - _t0, bot_name)
        return f"❌ Bot '{bot_name}' not found in dataset."

    canonical_name = row.iloc[0]["name"]
    entry = _THEME_COUNTS.get(canonical_name)

    if not entry:
        _record_time("get_taxonomy_breakdown", time.perf_counter() - _t0, bot_name)
        return f"No security and privacy reviews found for '{bot_name}'."

    total_sp = entry["total"]
    themes = entry["themes"]

    if not themes:
        _record_time("get_taxonomy_breakdown", time.perf_counter() - _t0, bot_name)
        return (
            f"No taxonomy themes assigned to the {total_sp} S&P review(s) of "
            f"'{canonical_name}' (all are 'unassigned')."
        )

    lines = []
    for theme, count in themes:
        pct = count / total_sp * 100
        bar = "█" * int(pct / 2)
        lines.append(
            f"  • {theme:<42} {THEME_CATEGORY[theme]:<24} {count:>4}  ({pct:>5.1f}%)  {bar}"
        )
    unassigned = total_sp - sum(count for _, count in themes)
    if unassigned:
        lines.append(f"  • {'(unassigned)':<42} {'':<24} {unassigned:>4}  "
                     f"({unassigned / total_sp * 100:>5.1f}%)")

    result = f"""
📊 Taxonomy breakdown for '{canonical_name}'
{'='*60}
Total S&P reviews analyzed: {total_sp}
Distinct themes reported: {len(themes)} of {len(ALL_THEMES)}

Theme frequency (each review counted under its primary theme):
{chr(10).join(lines)}

💡 Use get_review_samples('{canonical_name}') to see the actual review text.
"""
    _record_time("get_taxonomy_breakdown", time.perf_counter() - _t0, bot_name)
    return result


# ---------------------------------------------------------------------------
# Tool 3 — get_review_samples
# ---------------------------------------------------------------------------

@mcp.tool()
def get_review_samples(bot_name: str, n: int = 5) -> str:
    """
    Return a sample of real user security and privacy complaints for a bot.

    Args:
        bot_name: The Top.gg slug name of the bot
        n: Number of samples to return (default 5, max 20)
    """
    _t0 = time.perf_counter()

    if not _DATA_LOADED:
        _record_time("get_review_samples", time.perf_counter() - _t0, bot_name)
        return f"❌ Dataset not loaded: {_LOAD_ERROR}"

    n = min(max(1, n), 20)
    row = _find_bot(bot_name)
    if row.empty:
        _record_time("get_review_samples", time.perf_counter() - _t0, bot_name)
        return f"❌ Bot '{bot_name}' not found."

    privacy_reviews = _sp_reviews_for(row.iloc[0]["name"])

    if privacy_reviews.empty:
        _record_time("get_review_samples", time.perf_counter() - _t0, bot_name)
        return f"ℹ️ No security/privacy-flagged reviews found for '{bot_name}'."

    sample = privacy_reviews.sample(min(n, len(privacy_reviews)), random_state=42)

    lines = []
    for i, (_, rev) in enumerate(sample.iterrows(), 1):
        stars = "⭐" * int(rev["stars"]) if pd.notna(rev["stars"]) else "N/A"
        date = rev["date"] if pd.notna(rev.get("date")) else "Unknown date"
        text = str(rev["text"])[:300] + ("..." if len(str(rev["text"])) > 300 else "")
        lines.append(f"{i}. {stars} ({date})\n   \"{text}\"")

    total = len(privacy_reviews)
    result = f"""
💬 Security & Privacy Review Samples: {bot_name}
{'='*50}
Showing {len(sample)} of {total} security/privacy-flagged reviews:

{chr(10).join(lines)}

These are real user reports from Top.gg. BotLens identified them as
security/privacy-relevant using a DeBERTa-v3-large classifier (F1=0.94).
"""
    _record_time("get_review_samples", time.perf_counter() - _t0, bot_name)
    return result


# ---------------------------------------------------------------------------
# Tool 4 — compare_bots
# ---------------------------------------------------------------------------

@mcp.tool()
def compare_bots(bot_names: str) -> str:
    """
    Side-by-side security and privacy comparison of multiple bots.
    Useful for admins choosing between similar bots.

    Args:
        bot_names: Comma-separated list of bot names (e.g. 'wick, carl-bot, dyno')
    """
    _t0 = time.perf_counter()

    if not _DATA_LOADED:
        _record_time("compare_bots", time.perf_counter() - _t0, bot_names[:40])
        return f"❌ Dataset not loaded: {_LOAD_ERROR}"

    names = [n.strip() for n in bot_names.split(",") if n.strip()]
    if len(names) < 2:
        _record_time("compare_bots", time.perf_counter() - _t0, bot_names[:40])
        return "❌ Please provide at least 2 bot names separated by commas."

    lines = []
    metadata_only = []
    not_found = []
    for name in names:
        row = _find_bot(name)
        if row.empty:
            audit = _permission_audit(name)
            if audit is None:
                not_found.append(name)
            else:
                metadata_only.append(audit)
            continue
        r = row.iloc[0]
        salience = _salience_label(r["sprr"])
        servers = f"{int(r['server_count']):,}" if pd.notna(r["server_count"]) else "?"
        rating = f"{r['avg_rating']:.1f}" if pd.notna(r["avg_rating"]) else "N/A"
        perms = int(r["num_permissions"]) if pd.notna(r["num_permissions"]) else "?"
        gh = "✓" if r["has_github"] else "✗"
        lines.append(
            f"  {r['name']:<28} {salience:<20} SPRR={r['sprr']:>5.1f}%  "
            f"Rating={rating}  Permissions={perms}  Open Source={gh}  Servers={servers}"
        )

    output = f"""
⚖️ Bot Comparison
{'='*80}
{'Bot':<28} {'Salience':<20} {'SPRR':>7}  {'Rating':>7}  {'Perms':>6}  {'OSS':>4}  Servers
{'-'*80}
{chr(10).join(lines)}
"""
    if metadata_only:
        output += "\nNo reviews (SPRR unavailable) — permission audit from metadata:\n"
        output += "\n".join(
            f"  {a['name']:<28} Permissions={len(a['permissions'])}  {_audit_summary(a)}"
            for a in metadata_only
        ) + "\n"

    if not_found:
        output += f"\n⚠️ Not found in dataset: {', '.join(not_found)}"

    found_names = [n for n in names if not _find_bot(n).empty]
    if found_names:
        lowest = min(found_names, key=lambda n: _find_bot(n).iloc[0]["sprr"])
        output += (
            f"\nℹ️ Lowest S&P discussion salience (lowest SPRR): {lowest}. "
            f"{SPRR_SALIENCE_NOTE}"
        )

    _record_time("compare_bots", time.perf_counter() - _t0, bot_names[:40])
    return output


# ---------------------------------------------------------------------------
# Tool 5 — list_high_salience_bots
# ---------------------------------------------------------------------------

@mcp.tool()
def list_high_salience_bots(top_n: int = 20, min_reviews: int = 50) -> str:
    """
    List the bots with the highest security and privacy review rate (SPRR), indicating where S&P discussion is most prevalent.
    Filters to bots with a minimum number of reviews for statistical reliability.

    Args:
        top_n: Number of bots to return (default 20)
        min_reviews: Minimum total reviews required (default 50)
    """
    _t0 = time.perf_counter()

    if not _DATA_LOADED:
        _record_time("list_high_salience_bots", time.perf_counter() - _t0)
        return f"❌ Dataset not loaded: {_LOAD_ERROR}"

    filtered = _BOT_STATS[_BOT_STATS["total_reviews"] >= min_reviews].copy()
    top = filtered.nlargest(top_n, "sprr").reset_index(drop=True)

    if top.empty:
        _record_time("list_high_salience_bots", time.perf_counter() - _t0)
        return f"No bots found with at least {min_reviews} reviews."

    lines = []
    for i, r in top.iterrows():
        salience = _salience_label(r["sprr"])
        servers = f"{int(r['server_count']):,}" if pd.notna(r["server_count"]) else "?"
        lines.append(
            f"{i+1:>2}. {r['name']:<30} {salience:<20} SPRR={r['sprr']:>5.1f}%  "
            f"Reviews={int(r['total_reviews']):>5}  Servers={servers}"
        )

    result = f"""
🏆 Top {top_n} Bots by S&P Discussion Salience (min. {min_reviews} reviews)
{'='*70}
{'#':<4} {'Bot Name':<30} {'Salience':<20} {'SPRR':>7}  {'Reviews':>8}  Servers
{'-'*70}
{chr(10).join(lines)}

SPRR = Security and Privacy Review Rate (% of reviews raising security or privacy concerns)
Dataset: 183,142 reviews, 23,000+ bots, 8,832 security and privacy reviews (4.82% overall SPRR)

💡 Use get_bot_signals('<bot_name>') for a full report on any of these bots.
"""
    _record_time("list_high_salience_bots", time.perf_counter() - _t0)
    return result


# ---------------------------------------------------------------------------
# Tool 6 — scan_server_bots
# ---------------------------------------------------------------------------

@mcp.tool()
def scan_server_bots(filepath: str) -> str:
    """
    Scan all bots in a Discord server by reading a file containing bot names.
    The file should contain one bot name (Top.gg slug) per line.
    Returns bots ranked by S&P discussion salience (SPRR), with their top
    reported themes, permissions and sentiment. Bots that have no reviews
    get a permission audit from their Top.gg metadata instead (Algorithm 1).

    Args:
        filepath: Absolute path to a .txt or .csv file with one bot name per line
    """
    _t0 = time.perf_counter()

    if not _DATA_LOADED:
        _record_time("scan_server_bots", time.perf_counter() - _t0, filepath)
        return f"❌ Dataset not loaded: {_LOAD_ERROR}"

    path = Path(filepath)
    if not path.exists():
        _record_time("scan_server_bots", time.perf_counter() - _t0, filepath)
        return (
            f"❌ File not found: {filepath}\n"
            "Make sure to provide an absolute path (e.g. C:\\Users\\...\\bots.txt)"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_names = [line.strip() for line in f if line.strip()]
    except Exception as e:
        _record_time("scan_server_bots", time.perf_counter() - _t0, filepath)
        return f"❌ Could not read file: {e}"

    if not raw_names:
        _record_time("scan_server_bots", time.perf_counter() - _t0, filepath)
        return "❌ The file is empty or contains no bot names."

    found = []
    metadata_only = []
    not_found = []

    for name in raw_names:
        row = _find_bot(name)
        if row.empty:
            audit = _permission_audit(name)
            if audit is None:
                not_found.append(name)
            else:
                metadata_only.append(audit)
        else:
            r = row.iloc[0]
            found.append({
                "name": r["name"],
                "sprr": r["sprr"],
                "salience": _salience_label(r["sprr"]),
                "total_reviews": int(r["total_reviews"]),
                "privacy_reviews": int(r["privacy_reviews"]),
                "num_permissions": r["num_permissions"],
                "sentiment": _sentiment_for_bot(r["name"]),
            })

    found_sorted = sorted(found, key=lambda x: x["sprr"], reverse=True)

    lines = []
    for i, b in enumerate(found_sorted, 1):
        sentiment_text = _compact_sentiment(b["sentiment"], include_counts=True)
        lines.append(
            f"{i:>2}. {b['name']:<30} {b['salience']:<20} SPRR={b['sprr']:>5.1f}%  "
            f"({b['privacy_reviews']}/{b['total_reviews']} reviews)  "
            f"Permissions={int(b['num_permissions']) if pd.notna(b['num_permissions']) else '?'}  "
            f"Sentiment: {sentiment_text}  "
            f"Themes: {_top_themes_text(b['name'])}"
        )

    high_count = sum(1 for b in found if b["sprr"] >= 10)
    output = f"""
🔍 Server Bot Scan Results
{'='*70}
Scanned: {len(raw_names)} bots  |  With reviews: {len(found)}  |  Metadata only: {len(metadata_only)}  |  Not found: {len(not_found)}

{'#':<4} {'Bot Name':<30} {'Salience Label':<20} {'SPRR':>7}  Reviews / Permissions / Sentiment / Top themes
{'-'*70}
{chr(10).join(lines)}

{SPRR_SALIENCE_NOTE}
Sentiment is calculated only among each bot's S&P-relevant reviews and is not a bot safety or risk assessment.
"""
    if high_count:
        output += f"\n📋 {high_count} bot(s) show high S&P discussion salience — inspect sentiment composition and permissions before granting server access."

    if metadata_only:
        output += "\n\n🔐 No reviews in the dataset (SPRR unavailable) — permission audit from Top.gg metadata:"
        for a in metadata_only:
            output += f"\n  • {a['name']:<30} Permissions={len(a['permissions']):<3} {_audit_summary(a)}"

    if not_found:
        output += f"\n\n⚠️ Not found in BotLens dataset: {', '.join(not_found)}"
        output += "\n   (These bots may be too new or too niche to be listed on Top.gg.)"

    output += "\n\n💡 Use get_bot_signals('<name>') for a full report on any bot above."

    _record_time("scan_server_bots", time.perf_counter() - _t0, filepath)
    return output


# ===========================================================================
# INTERNAL HELPERS — Functions below are kept for development and benchmarking
# but are NOT registered as MCP tools (no @mcp.tool() decorator). They can
# still be imported and called from Python directly.
# ===========================================================================


def _format_permission_audit(audit: dict) -> str:
    """Render PermissionAudit(b): the metadata-only report for a bot with no reviews."""
    perms = audit["permissions"]
    servers = (f"{int(audit['server_count']):,}" if pd.notna(audit.get("server_count"))
               else "Unknown")
    if perms:
        lines = "\n".join(
            f"  • {p}: {PERMISSION_EXPLANATIONS.get(p.upper(), f'ℹ️ {p}')}" for p in perms
        )
    else:
        lines = "  (no permissions listed in Top.gg metadata)"
    return f"""
🔐 BotLens Permission Audit: {audit['name']}
{'='*50}
This bot has no reviews in the BotLens dataset, so review-derived signals
(SPRR, themes, sentiment) are unavailable. Its requested permissions are
reported from Top.gg metadata instead (PermissionAudit, Algorithm 1).
Installed on ~{servers} servers

🔐 Permissions Requested ({len(perms)}):
{lines}

SUMMARY:
{_audit_summary(audit)}
"""


def get_permissions_audit(bot_name: str) -> str:
    """
    Plain-English explanation of every permission a bot requests, from Top.gg
    metadata. This is the PermissionAudit fallback of Algorithm 1; the public
    tools call it for bots that have no reviews.
    Internal helper — not exposed via MCP.
    """
    _t0 = time.perf_counter()

    if not _DATA_LOADED:
        _record_time("get_permissions_audit", time.perf_counter() - _t0, bot_name)
        return f"❌ Dataset not loaded: {_LOAD_ERROR}"

    audit = _permission_audit(bot_name)
    _record_time("get_permissions_audit", time.perf_counter() - _t0, bot_name)
    if audit is None:
        return f"❌ Bot '{bot_name}' not found."
    return _format_permission_audit(audit)


def search_bots(query: str, top_n: int = 10) -> str:
    """
    Search for bots in the dataset by name, tag, or description.
    Internal helper — not exposed via MCP.
    """
    _t0 = time.perf_counter()

    if not _DATA_LOADED:
        _record_time("search_bots", time.perf_counter() - _t0, query)
        return f"❌ Dataset not loaded: {_LOAD_ERROR}"

    q = query.lower().strip()
    mask = (
        _BOT_STATS["name"].str.lower().str.contains(q, na=False) |
        _BOT_STATS["short_description"].str.lower().str.contains(q, na=False) |
        _BOT_STATS["tags"].str.lower().str.contains(q, na=False)
    )
    results = _BOT_STATS[mask].nlargest(top_n, "total_reviews")

    if results.empty:
        _record_time("search_bots", time.perf_counter() - _t0, query)
        return f"No bots found matching '{query}'."

    lines = []
    for _, r in results.iterrows():
        salience = _salience_label(r["sprr"])
        desc = str(r["short_description"])[:60] + "..." if pd.notna(r["short_description"]) else ""
        lines.append(f"  • {r['name']:<30} {salience:<18} SPRR={r['sprr']:>5.1f}%  — {desc}")

    result = f"""
🔍 Search Results for '{query}':
{'='*60}
{chr(10).join(lines)}
"""
    _record_time("search_bots", time.perf_counter() - _t0, query)
    return result


def filter_bots_by_theme(theme: str, top_n: int = 15, min_reviews: int = 10) -> str:
    """
    Find bots whose S&P reviews are tagged with a specific taxonomy theme.
    Internal helper — not exposed via MCP.
    """
    _t0 = time.perf_counter()

    if not _DATA_LOADED:
        _record_time("filter_bots_by_theme", time.perf_counter() - _t0, theme)
        return f"❌ Dataset not loaded: {_LOAD_ERROR}"

    match = next((t for t in ALL_THEMES if t.lower() == theme.lower().strip()), None)
    if match is None:
        close = [t for t in ALL_THEMES if theme.lower().strip() in t.lower()]
        suggestion = f"\nDid you mean: {', '.join(close)}?" if close else ""
        _record_time("filter_bots_by_theme", time.perf_counter() - _t0, theme)
        return (
            f"❌ '{theme}' is not one of the {len(ALL_THEMES)} taxonomy themes.{suggestion}\n\n"
            f"Available themes:\n{', '.join(ALL_THEMES)}"
        )

    rows = []
    for _, r in _BOT_STATS[_BOT_STATS["total_reviews"] >= min_reviews].iterrows():
        entry = _THEME_COUNTS.get(r["name"])
        n = dict(entry["themes"]).get(match, 0) if entry else 0
        if n:
            rows.append((r, n))

    if not rows:
        _record_time("filter_bots_by_theme", time.perf_counter() - _t0, theme)
        return f"No bots with at least {min_reviews} reviews have S&P reviews tagged '{match}'."

    rows.sort(key=lambda x: -x[1])
    lines = []
    for r, n in rows[:top_n]:
        servers = f"{int(r['server_count']):,}" if pd.notna(r["server_count"]) else "?"
        lines.append(
            f"  • {r['name']:<30} {n:>4} reviews  SPRR={r['sprr']:>5.1f}%  Servers={servers}"
        )

    result = f"""
🔎 Bots with S&P reviews tagged '{match}' (top {top_n}):
{'='*60}
{chr(10).join(lines)}

Total bots with '{match}' reviews: {len(rows)}
Total reviews tagged with this theme: {sum(n for _, n in rows)}
"""
    _record_time("filter_bots_by_theme", time.perf_counter() - _t0, theme)
    return result


def get_dataset_stats() -> str:
    """
    High-level statistics about the BotLens dataset.
    Internal helper — not exposed via MCP.
    """
    _t0 = time.perf_counter()

    if not _DATA_LOADED:
        _record_time("get_dataset_stats", time.perf_counter() - _t0)
        return f"❌ Dataset not loaded: {_LOAD_ERROR}"

    total_reviews = len(_DF)
    total_bots = _BOT_STATS["name"].nunique()
    privacy_reviews = int(_DF["is_privacy"].sum())
    overall_sprr = privacy_reviews / total_reviews * 100

    high_risk = (_BOT_STATS["sprr"] >= 10).sum()
    medium_risk = ((_BOT_STATS["sprr"] >= 5) & (_BOT_STATS["sprr"] < 10)).sum()
    low_risk = (_BOT_STATS["sprr"] < 5).sum()

    open_source = int(_BOT_STATS["has_github"].sum())
    avg_perms = pd.to_numeric(_BOT_STATS["num_permissions"], errors="coerce").mean()

    result = f"""
📊 BotLens Dataset Statistics
{'='*50}
Total reviews analyzed:       {total_reviews:>10,}
Total unique bots:            {total_bots:>10,}
Security/privacy-flagged reviews:      {privacy_reviews:>10,}
Overall SPRR:                  {overall_sprr:>9.2f}%

Salience Distribution:
  🔴 High Salience  (SPRR ≥ 10%): {int(high_risk):>6,} bots
  🟡 Medium Salience (SPRR 5-10%): {int(medium_risk):>6,} bots
  🟢 Low/Very Low (SPRR < 5%): {int(low_risk):>6,} bots

Bot Metadata:
  Open source bots:           {open_source:>10,}
  Avg permissions per bot:    {avg_perms:>9.1f}

Security & Privacy Taxonomy: {len(ALL_THEMES)} themes in {len(CATEGORIES)} groups
  ({', '.join(f'{cat} {len(ts)}' for cat, ts in CATEGORIES.items())})
  20 inherited from Akgul et al. (USENIX Security 2024) + 11 new
"""
    _record_time("get_dataset_stats", time.perf_counter() - _t0)
    return result


def diagnose() -> str:
    """
    Diagnose the BotLens server status.
    Internal helper — not exposed via MCP.
    """
    _t0 = time.perf_counter()

    lines = ["🔧 BotLens Diagnostic Report", "=" * 50]

    lines.append(f"\n1. Data Load Status: {'✅ OK' if _DATA_LOADED else '❌ FAILED'}")
    if not _DATA_LOADED:
        lines.append(f"   Error:\n{_LOAD_ERROR}")

    lines.append(f"\n2. Expected data path:\n   {DATA_PATH}")
    lines.append(f"   File exists: {'✅ Yes' if DATA_PATH.exists() else '❌ No'}")

    env_val = os.environ.get("BOTLENS_DATA_DIR", "(not set)")
    lines.append(f"\n3. BOTLENS_DATA_DIR env var: {env_val}")

    if _DATA_LOADED and not _DF.empty:
        lines.append(f"\n4. Dataset shape: {_DF.shape[0]:,} rows × {_DF.shape[1]} columns")
        lines.append(f"   Columns: {', '.join(_DF.columns.tolist())}")
        lines.append(f"   Bot stats rows: {len(_BOT_STATS):,}")
        lines.append(f"   Privacy reviews: {int(_DF['is_privacy'].sum()):,}")
        lines.append(f"   Overall SPRR: {_DF['is_privacy'].mean()*100:.2f}%")
    elif _DATA_LOADED:
        lines.append("\n4. Dataset loaded but empty — check CSV file content.")

    lines.append("\n" + "=" * 50)

    result = "\n".join(lines)
    _record_time("diagnose", time.perf_counter() - _t0)
    return result


def get_timing_stats() -> str:
    """
    Aggregated response time statistics for all tool calls in this session.
    Internal helper — not exposed via MCP. Used for the paper's performance
    evaluation section.
    """
    _t0 = time.perf_counter()

    if not _TIMING_LOG:
        _record_time("get_timing_stats", time.perf_counter() - _t0)
        return "⏱️ No timing data recorded yet. Call some tools first."

    lines = ["⏱️ BotLens Tool Response Time Statistics", "=" * 60]
    lines.append(f"{'Tool':<30} {'Calls':>5}  {'Mean(ms)':>8}  {'Min(ms)':>8}  {'Max(ms)':>8}")
    lines.append("-" * 60)

    total_calls = 0
    all_times = []

    for tool_name in sorted(_TIMING_LOG.keys()):
        times = _TIMING_LOG[tool_name]
        n = len(times)
        mean_t = sum(times) / n
        min_t = min(times)
        max_t = max(times)
        total_calls += n
        all_times.extend(times)
        lines.append(
            f"  {tool_name:<28} {n:>5}  {mean_t * 1000:>8.2f}  {min_t * 1000:>8.2f}  {max_t * 1000:>8.2f}"
        )

    overall_mean = sum(all_times) / len(all_times)
    lines.append("-" * 60)
    lines.append(f"  {'OVERALL':<28} {total_calls:>5}  {overall_mean * 1000:>8.2f}")

    result = "\n".join(lines)
    _record_time("get_timing_stats", time.perf_counter() - _t0)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()