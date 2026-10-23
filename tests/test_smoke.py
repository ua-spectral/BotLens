"""
Smoke tests for the BotLens artifact.

These run in seconds, need no data, no GPU and no network, and are the fastest way
for an artifact reviewer to confirm the package is internally consistent and wired
up correctly.

    python -m pytest tests/test_smoke.py -v
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.taxonomy import PRIVACY_TAXONOMY, CATEGORIES, ALL_THEMES  # noqa: E402
from config import settings  # noqa: E402


# --------------------------------------------------------------------------
# Taxonomy — paper Table V
# --------------------------------------------------------------------------

# Theme group -> number of themes, exactly as printed in Table V.
PAPER_GROUP_SIZES = {
    "Data Practices": 7,
    "User Rights & Controls": 7,
    "Security & Safety": 8,      # 6 inherited + Phishing and Scams + Data Retention
    "Discord-Specific Risks": 9,
}


def test_taxonomy_matches_paper():
    """31 themes in four groups of 7/7/8/9 (Table V)."""
    assert len(PRIVACY_TAXONOMY) == 31
    assert set(CATEGORIES) == set(PAPER_GROUP_SIZES)
    for group, expected in PAPER_GROUP_SIZES.items():
        assert len(CATEGORIES[group]) == expected, (
            f"{group}: expected {expected} themes, got {len(CATEGORIES[group])}"
        )
    assert sum(len(v) for v in CATEGORIES.values()) == 31


def test_category_assignments_are_consistent():
    """CATEGORIES and each theme's own 'category' field must agree."""
    for group, themes in CATEGORIES.items():
        for theme in themes:
            assert theme in PRIVACY_TAXONOMY, f"{theme!r} in CATEGORIES but not in taxonomy"
            assert PRIVACY_TAXONOMY[theme]["category"] == group, (
                f"{theme!r}: CATEGORIES says {group!r}, "
                f"entry says {PRIVACY_TAXONOMY[theme]['category']!r}"
            )
    listed = {t for ts in CATEGORIES.values() for t in ts}
    assert listed == set(PRIVACY_TAXONOMY), "theme missing from CATEGORIES"


def test_two_new_security_themes_are_marked_new():
    """Phishing and Scams / Data Retention are NEW in Table V, not inherited."""
    for theme in ("Phishing and Scams", "Data Retention"):
        assert PRIVACY_TAXONOMY[theme]["source"] == "New"
        assert PRIVACY_TAXONOMY[theme]["category"] == "Security & Safety"


def test_source_split_is_20_inherited_11_new():
    """Paper Sec. III-B: 20 themes from Akgul et al. + 11 developed here."""
    sources = [t["source"] for t in PRIVACY_TAXONOMY.values()]
    assert sources.count("Akgul") == 20
    assert sources.count("New") == 11


def test_all_themes_export_is_complete():
    assert len(ALL_THEMES) == 31
    assert set(ALL_THEMES) == set(PRIVACY_TAXONOMY)


# --------------------------------------------------------------------------
# Config — paper Sec. III-C4
# --------------------------------------------------------------------------

def test_training_hyperparameters_match_paper():
    assert settings.MAX_SEQ_LENGTH == 512
    assert settings.NUM_EPOCHS == 10
    assert settings.EARLY_STOPPING_PATIENCE == 3
    assert settings.LEARNING_RATE == 2e-5
    assert settings.WARMUP_RATIO == 0.1
    assert settings.WEIGHT_DECAY == 0.01
    assert settings.TRAIN_BATCH_SIZE == 8
    assert settings.EVAL_BATCH_SIZE == 16
    assert settings.PREDICTION_THRESHOLD == 0.8
    assert settings.LABELING_CONFIDENCE_THRESHOLD == 0.7
    assert (settings.TRAIN_SPLIT, settings.VAL_SPLIT, settings.TEST_SPLIT) == (0.8, 0.1, 0.1)


def test_theme_count_setting_matches_taxonomy():
    assert settings.TOP_N_THEMES == len(PRIVACY_TAXONOMY) == 31


# --------------------------------------------------------------------------
# Figure code must not re-derive the taxonomy by hand
# --------------------------------------------------------------------------

def test_stage8_derives_mapping_from_taxonomy():
    """stage8 must derive its theme->category mapping from config/taxonomy.py.

    Restating the mapping in the figure script allows theme names to diverge from
    the taxonomy, which changes the group totals reported in Figure 4.
    """
    src = (REPO_ROOT / "scripts" / "stage8_visualizations.py").read_text(encoding="utf-8")
    assert "from config.taxonomy import CATEGORIES" in src
    assert '"Remove PII":' not in src, "mapping must come from config/taxonomy.py"

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    mapping = {t: c for c, ts in CATEGORIES.items() for t in ts}
    assert "Remove Personally Identifiable Information" in mapping
    assert mapping["Remove Personally Identifiable Information"] == "Data Practices"


# --------------------------------------------------------------------------
# MCP server — the six tools named in Sec. V-B2
# --------------------------------------------------------------------------

PAPER_TOOLS = [
    "get_bot_signals",
    "get_taxonomy_breakdown",
    "get_review_samples",
    "compare_bots",
    "list_high_salience_bots",
    "scan_server_bots",
]


def test_mcp_server_defines_the_six_paper_tools():
    """Tool names must match Sec. V-B2 exactly, using salience rather than risk."""
    src = (REPO_ROOT / "mcp_server" / "server.py").read_text(encoding="utf-8")
    for tool in PAPER_TOOLS:
        assert f"def {tool}(" in src, f"MCP tool {tool!r} from the paper is missing"
    for superseded in ("def check_bot_safety(", "def list_high_risk_bots("):
        assert superseded not in src, f"tool name does not match the paper: {superseded}"


def test_mcp_server_uses_the_31_theme_taxonomy():
    """get_taxonomy_breakdown counts the 31 Table V themes (TaxonomyFlags in
    Algorithm 1), not keyword matches against a superseded concept list."""
    src = (REPO_ROOT / "mcp_server" / "server.py").read_text(encoding="utf-8")
    assert "from config.taxonomy import ALL_THEMES, CATEGORIES" in src
    assert "TAXONOMY_CONCEPTS" not in src, "legacy 33-concept list is back"
    assert "str.contains(keyword" not in src, "taxonomy counts must not be keyword matches"


def test_mcp_server_has_permission_audit_fallback():
    """Algorithm 1, line 23: bots with no reviews get a metadata permission audit."""
    src = (REPO_ROOT / "mcp_server" / "server.py").read_text(encoding="utf-8")
    assert "def _permission_audit(" in src
    assert "bot_metadata_cleaned.csv" in src
    for tool in ("get_bot_signals", "compare_bots", "scan_server_bots"):
        body = src.split(f"def {tool}(", 1)[1].split("\n@mcp.tool()", 1)[0]
        assert "_permission_audit(" in body, f"{tool} has no PermissionAudit fallback"


def test_mcp_server_never_calls_a_bot_safe():
    """SPRR measures discussion salience, not safety (paper Sec. VI)."""
    src = (REPO_ROOT / "mcp_server" / "server.py").read_text(encoding="utf-8").lower()
    for phrase in ("safest", "appears safe", "safety report"):
        assert phrase not in src, f"tool output must not say {phrase!r}"


def test_mcp_lookups_do_not_scan_the_review_table():
    """Sec. V-D latency: tool calls are precomputed lookups, not full-table scans."""
    src = (REPO_ROOT / "mcp_server" / "server.py").read_text(encoding="utf-8")
    assert '_DF[_DF["name"].str.lower()' not in src


def test_readme_documents_the_shipped_tool_names():
    """The README must document the tool names the server actually defines."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for tool in PAPER_TOOLS:
        assert tool in readme, f"README does not document {tool!r}"
    assert "check_bot_safety" not in readme
    assert "list_high_risk_bots" not in readme


# --------------------------------------------------------------------------
# Scripts and outputs are named after the paper's RQs and figure numbers
# --------------------------------------------------------------------------

def test_rq3_is_one_script_named_for_rq3():
    """The permissions analysis is RQ3 (Sec. IV-D) and lives in one script."""
    scripts = REPO_ROOT / "scripts"
    assert (scripts / "rq3_permissions_metadata.py").is_file()
    assert not list(scripts.glob("rq4_*.py")), "legacy rq4_* scripts are back"


def test_figure_outputs_use_paper_figure_numbers():
    """Every generated figure file starts with its paper figure number."""
    expected = {
        "stage8_visualizations.py": ["fig02_", "fig03_", "fig04_", "fig05_",
                                     "fig06_", "fig07_", "fig08_", "fig09_"],
        "rq3_permissions_metadata.py": ["fig10_", "fig11_", "fig12_", "fig13_",
                                        "fig14_", "fig18_"],
        "language_affective_analysis.py": ["fig17_"],
    }
    for script, prefixes in expected.items():
        src = (REPO_ROOT / "scripts" / script).read_text(encoding="utf-8")
        for prefix in prefixes:
            assert f'"{prefix}' in src, f"{script} does not write a {prefix}* figure"
        assert '"rq4_' not in src


def test_stage7_has_no_unreported_analyses():
    """Stage 7 computes only what Sec. IV-A reports; the paper reports no KPSS test."""
    src = (REPO_ROOT / "scripts" / "stage7_analysis.py").read_text(encoding="utf-8")
    assert "kpss" not in src.lower()
    assert "statsmodels" not in src
