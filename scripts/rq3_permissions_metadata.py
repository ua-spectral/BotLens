"""
RQ3 — Permissions, Metadata and S&P Concerns (paper Sec. IV-D)

Computes the per-bot permission and metadata statistics behind Sec. IV-D,
renders Figs. 10-14 and 18, and writes the Spearman correlations behind
Fig. 14 to data/results/rq3_feature_correlations.csv.

All figures use the Wong (2011) colorblind-safe palette; Fig. 18 keeps the
matplotlib default blue it was published with.

Usage:
    python scripts/rq3_permissions_metadata.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import PROCESSED_DIR, RESULTS_DIR

# ── IEEE style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
})

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

# Semantic aliases for chart roles
C_PRIMARY    = WONG["blue"]       # Main data, "with" condition
C_SECONDARY  = WONG["sky_blue"]   # Comparison, "without" condition
C_ALERT      = WONG["vermilion"]  # High-risk, privacy, warning
C_ALERT_LT   = "#F4A582"         # Lighter vermilion-tinted (for paired bars)
C_SAFE       = WONG["green"]      # Positive, open-source
C_NEUTRAL    = "#BBBBBB"          # Gray for neutral/baseline
C_ACCENT     = WONG["orange"]     # Moderate risk
C_ACCENT_LT  = "#FDD49E"         # Lighter orange-tinted
C_DARK       = WONG["black"]      # Text, outlines
C_PURPLE     = WONG["purple"]     # Additional series

# Gradient for binned bars (low risk → high risk)
BIN_GRADIENT = [
    WONG["sky_blue"],   # lowest
    WONG["blue"],
    WONG["orange"],
    WONG["vermilion"],  # highest
]


# Discord permission flags
PERMISSION_FLAGS = {
    "CREATE_INSTANT_INVITE": 0x1, "KICK_MEMBERS": 0x2, "BAN_MEMBERS": 0x4,
    "ADMINISTRATOR": 0x8, "MANAGE_CHANNELS": 0x10, "MANAGE_GUILD": 0x20,
    "ADD_REACTIONS": 0x40, "VIEW_AUDIT_LOG": 0x80, "PRIORITY_SPEAKER": 0x100,
    "STREAM": 0x200, "VIEW_CHANNEL": 0x400, "SEND_MESSAGES": 0x800,
    "SEND_TTS_MESSAGES": 0x1000, "MANAGE_MESSAGES": 0x2000,
    "EMBED_LINKS": 0x4000, "ATTACH_FILES": 0x8000,
    "READ_MESSAGE_HISTORY": 0x10000, "MENTION_EVERYONE": 0x20000,
    "USE_EXTERNAL_EMOJIS": 0x40000, "VIEW_GUILD_INSIGHTS": 0x80000,
    "CONNECT": 0x100000, "SPEAK": 0x200000, "MUTE_MEMBERS": 0x400000,
    "DEAFEN_MEMBERS": 0x800000, "MOVE_MEMBERS": 0x1000000,
    "USE_VAD": 0x2000000, "CHANGE_NICKNAME": 0x4000000,
    "MANAGE_NICKNAMES": 0x8000000, "MANAGE_ROLES": 0x10000000,
    "MANAGE_WEBHOOKS": 0x20000000, "MANAGE_EMOJIS_AND_STICKERS": 0x40000000,
    "USE_APPLICATION_COMMANDS": 0x80000000, "MANAGE_EVENTS": 0x100000000,
    "MANAGE_THREADS": 0x200000000, "USE_PUBLIC_THREADS": 0x400000000,
    "USE_PRIVATE_THREADS": 0x800000000, "USE_EXTERNAL_STICKERS": 0x1000000000,
    "SEND_MESSAGES_IN_THREADS": 0x2000000000,
    "USE_EMBEDDED_ACTIVITIES": 0x4000000000, "MODERATE_MEMBERS": 0x10000000000,
}

HIGH_RISK_PERMS = [
    "ADMINISTRATOR", "BAN_MEMBERS", "KICK_MEMBERS", "MANAGE_GUILD",
    "MANAGE_CHANNELS", "MANAGE_ROLES", "MANAGE_WEBHOOKS",
    "MANAGE_MESSAGES", "VIEW_AUDIT_LOG", "MODERATE_MEMBERS",
    "READ_MESSAGE_HISTORY", "MENTION_EVERYONE",
]


def decode_permissions(perm_int):
    try:
        perm_int = int(perm_int)
    except (ValueError, TypeError):
        return []
    return [name for name, flag in PERMISSION_FLAGS.items() if perm_int & flag]


def count_from_decoded(decoded_str):
    if pd.isna(decoded_str) or not decoded_str:
        return 0
    return len([p for p in str(decoded_str).split(",") if p.strip()])


def perms_from_decoded(decoded_str):
    if pd.isna(decoded_str) or not decoded_str:
        return []
    return [p.strip() for p in str(decoded_str).split(",") if p.strip()]


def build_bot_df():
    """Load and prepare per-bot dataframe with SPRR and metadata."""
    all_df = pd.read_csv(PROCESSED_DIR / "all_reviews_classified.csv")
    privacy_df = pd.read_csv(PROCESSED_DIR / "privacy_reviews_full.csv")

    bot_total = all_df.groupby("bot_id").size().reset_index(name="total_reviews")
    bot_priv = privacy_df.groupby("bot_id").size().reset_index(name="privacy_reviews")
    bot_sprr = bot_total.merge(bot_priv, on="bot_id", how="inner")
    bot_sprr["sprr"] = bot_sprr["privacy_reviews"] / bot_sprr["total_reviews"] * 100

    meta_cols = [c for c in ["bot_id", "permissions", "permissions_decoded",
                              "server_count", "tags", "average_rating",
                              "has_github", "name", "primary_theme"] if c in privacy_df.columns]
    bot_meta = privacy_df[meta_cols].drop_duplicates(subset="bot_id", keep="first")
    bot_df = bot_sprr.merge(bot_meta, on="bot_id", how="left")

    if "permissions" in bot_df.columns:
        bot_df["perm_list"] = bot_df["permissions"].apply(decode_permissions)
        bot_df["num_perms"] = bot_df["perm_list"].apply(len)
        bot_df["num_high_risk"] = bot_df["perm_list"].apply(
            lambda ps: sum(1 for p in ps if p in HIGH_RISK_PERMS))
        bot_df["has_admin"] = bot_df["perm_list"].apply(lambda ps: "ADMINISTRATOR" in ps)
    elif "permissions_decoded" in bot_df.columns:
        bot_df["num_perms"] = bot_df["permissions_decoded"].apply(count_from_decoded)
        bot_df["perm_list"] = bot_df["permissions_decoded"].apply(perms_from_decoded)
        bot_df["num_high_risk"] = bot_df["perm_list"].apply(
            lambda ps: sum(1 for p in ps if p.upper().replace(" ", "_") in HIGH_RISK_PERMS))
        bot_df["has_admin"] = bot_df["perm_list"].apply(
            lambda ps: any("admin" in p.lower() for p in ps))

    return bot_df, privacy_df


def save_fig(fig, path):
    """Save figure with padding."""
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {path}")


# =====================================================================
# Fig. 10: Permission Count vs SPRR (Scatter + Binned Bar)
# =====================================================================
def fig_perm_count_vs_sprr(bot_df):
    df = bot_df[(bot_df["total_reviews"] >= 5) & (bot_df["num_perms"] > 0)].copy()
    if len(df) < 20:
        print("  [SKIP] Not enough data for perm count vs SPRR")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2),
                                    gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.35)

    # Left: Scatter
    ax1.scatter(df["num_perms"], df["sprr"], s=12, alpha=0.35,
                c=C_PRIMARY, edgecolors="none", zorder=2)
    slope, intercept, r, p, se = stats.linregress(df["num_perms"], df["sprr"])
    print(f"  Permission count vs SPRR: r = {r:.3f}, p = {p:.3f} (n = {len(df):,} bots)")
    x_line = np.linspace(df["num_perms"].min(), df["num_perms"].max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, color=C_ALERT, linewidth=1.5,
             linestyle="--", label=f"r = {r:.3f}, p = {p:.3f}", zorder=3)
    ax1.set_xlabel("Number of Requested Permissions")
    ax1.set_ylabel("SPRR (%)")
    ax1.set_title("(a) Permission Count vs Security and Privacy Review Rate")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_ylim(bottom=-0.5)

    # Right: Binned bar
    bins = [0, 5, 10, 15, 20, 25, float("inf")]
    labels = ["1–5", "6–10", "11–15", "16–20", "21–25", "26+"]
    df["perm_bin"] = pd.cut(df["num_perms"], bins=bins, labels=labels)

    bin_stats = df.groupby("perm_bin", observed=True).agg(
        total=("total_reviews", "sum"),
        privacy=("privacy_reviews", "sum"),
        n_bots=("bot_id", "count"),
    )
    bin_stats["sprr"] = bin_stats["privacy"] / bin_stats["total"] * 100
    print("  SPRR by permission-count bin: " + " / ".join(
        f"{label} {row['sprr']:.1f}%" for label, row in bin_stats.iterrows()))

    # Gradient: sky_blue → blue → orange → vermilion (low→high risk)
    n = len(bin_stats)
    if n <= 4:
        colors = BIN_GRADIENT[:n]
    else:
        # Interpolate for 5-6 bins
        colors = [WONG["sky_blue"], WONG["sky_blue"], WONG["blue"],
                  WONG["blue"], WONG["orange"], WONG["vermilion"]][:n]

    bars = ax2.bar(range(n), bin_stats["sprr"], color=colors,
                   edgecolor="white", linewidth=0.5, zorder=2)

    # Labels INSIDE bars
    for i, (idx, row) in enumerate(bin_stats.iterrows()):
        ax2.text(i, row["sprr"] * 0.5, f"{row['sprr']:.1f}%",
                 ha="center", va="center", fontsize=7.5, fontweight="bold",
                 color="white" if row["sprr"] > 1.5 else C_DARK)

    ax2.set_xticks(range(n))
    ax2.set_xticklabels(bin_stats.index, rotation=0)
    ax2.set_xlabel("Permission Count Range")
    ax2.set_ylabel("SPRR (%)")
    ax2.set_title("(b) SPRR by Permission Count Bin")

    save_fig(fig, FIGURES_DIR / "fig10_perm_count_vs_sprr.png")


# =====================================================================
# Fig. 11: High-Risk Permission SPRR Comparison
# =====================================================================
def fig_high_risk_perms(bot_df):
    df = bot_df[bot_df["total_reviews"] >= 5].copy()

    results = []
    for perm in HIGH_RISK_PERMS:
        has = df[df["perm_list"].apply(
            lambda ps: perm in ps or any(perm.lower().replace("_", " ") in p.lower().replace("_", " ") for p in ps)
        )]
        no = df[~df.index.isin(has.index)]

        if len(has) >= 10 and has["total_reviews"].sum() > 0:
            sprr_with = has["privacy_reviews"].sum() / has["total_reviews"].sum() * 100
            sprr_without = no["privacy_reviews"].sum() / no["total_reviews"].sum() * 100 if no["total_reviews"].sum() > 0 else 0
            results.append({
                "permission": perm.replace("_", " ").title(),
                "sprr_with": sprr_with,
                "sprr_without": sprr_without,
                "n_bots": len(has),
                "diff": sprr_with - sprr_without,
            })

    if not results:
        print("  [SKIP] Not enough data for high-risk perm chart")
        return

    res_df = pd.DataFrame(results).sort_values("diff", ascending=True)
    for _, row in res_df[::-1].iterrows():
        print(f"  {row['permission']:<22} with {row['sprr_with']:5.2f}% (n={row['n_bots']:,})"
              f"  without {row['sprr_without']:5.2f}%")
    print(f"  {int((res_df['diff'] > 0).sum())} of {len(res_df)} permissions: "
          f"SPRR higher with the permission than without")

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    fig.subplots_adjust(left=0.30, right=0.95, top=0.92, bottom=0.08)
    y_pos = range(len(res_df))

    # Vermilion for "with permission", sky blue for "without"
    bars_with = ax.barh(y_pos, res_df["sprr_with"], height=0.38,
                        color=C_ALERT, edgecolor="white", linewidth=0.5,
                        label="With Permission", zorder=2)
    bars_without = ax.barh([y + 0.4 for y in y_pos], res_df["sprr_without"], height=0.38,
                           color=C_SECONDARY, edgecolor="white", linewidth=0.5,
                           label="Without Permission", zorder=2)

    ax.set_yticks([y + 0.2 for y in y_pos])
    ax.set_yticklabels(res_df["permission"], fontsize=7.5)
    ax.set_xlabel("SPRR (%)")
    ax.set_title("Security and Privacy Review Rate by High-Risk Permission")
    ax.legend(loc="lower right", framealpha=0.9)

    # Labels INSIDE the bars
    for i, (_, row) in enumerate(res_df.iterrows()):
        if row["sprr_with"] > 0.8:
            ax.text(row["sprr_with"] * 0.5, i,
                    f"{row['sprr_with']:.2f}% (n={row['n_bots']:,})",
                    ha="center", va="center", fontsize=5.5,
                    color="white", fontweight="bold")
        if row["sprr_without"] > 0.8:
            ax.text(row["sprr_without"] * 0.5, i + 0.4,
                    f"{row['sprr_without']:.2f}%",
                    ha="center", va="center", fontsize=5.5,
                    color=C_DARK, fontweight="bold")

    save_fig(fig, FIGURES_DIR / "fig11_high_risk_perms.png")


# =====================================================================
# Fig. 13: Server Count vs SPRR
# =====================================================================
@plt.style.context("default")   # drawn as Fig. 13 appears in the paper
def fig_server_count_vs_sprr(bot_df):
    df = bot_df[(bot_df["total_reviews"] >= 5) &
                (bot_df["server_count"].notna()) &
                (bot_df["server_count"] > 0)].copy()

    if len(df) < 20:
        print("  [SKIP] Not enough server count data")
        return

    bins = [0, 100, 1000, 10000, 100000, float("inf")]
    labels = ["1-100", "100-1K", "1K-10K", "10K-100K", "100K+"]
    df["sc_bin"] = pd.cut(df["server_count"], bins=bins, labels=labels)

    sc_stats = df.groupby("sc_bin", observed=True).agg(
        total=("total_reviews", "sum"),
        privacy=("privacy_reviews", "sum"),
        n_bots=("bot_id", "count"),
    )
    sc_stats["sprr"] = sc_stats["privacy"] / sc_stats["total"] * 100
    print(f"  Bots disclosing a server count: {len(df):,}")
    print("  SPRR by server count: " + " / ".join(
        f"{label} {row['sprr']:.2f}% (n={row['n_bots']:,.0f})" for label, row in sc_stats.iterrows()))

    # Horizontal bars, smallest bin at the bottom.
    fig, ax = plt.subplots(figsize=(8, 3.5))
    n = len(sc_stats)
    ax.barh(range(n), sc_stats["sprr"], color="#1F77B4")
    for i, (_, row) in enumerate(sc_stats.iterrows()):
        ax.text(row["sprr"] + 0.2, i, f"{row['sprr']:.2f}% (n={row['n_bots']:,.0f})",
                va="center", fontsize=10)

    ax.set_yticks(range(n))
    ax.set_yticklabels(sc_stats.index, fontsize=12)
    ax.set_xlim(0, 15)
    ax.set_xlabel("SPRR (%)", fontsize=12)
    ax.set_ylabel("Server Count", fontsize=12)
    ax.set_title("Security and Privacy Review Rate by Bot Server Count", fontsize=13)
    fig.tight_layout()
    save_fig(fig, FIGURES_DIR / "fig13_server_count_vs_sprr.png")


# =====================================================================
# Fig. 12: Admin vs Non-Admin SPRR (with theme breakdown)
# =====================================================================
@plt.style.context("default")   # drawn as Fig. 12 appears in the paper
def fig_admin_comparison(bot_df, privacy_df):
    df = bot_df[bot_df["total_reviews"] >= 5].copy()

    if "has_admin" not in df.columns:
        print("  [SKIP] No admin data")
        return

    admin = df[df["has_admin"] == True]
    non_admin = df[df["has_admin"] == False]

    if len(admin) < 5 or len(non_admin) < 5:
        print("  [SKIP] Not enough admin/non-admin bots")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.3, 3.3),
                                    gridspec_kw={"width_ratios": [0.75, 1.25]})

    # Left: SPRR comparison bar
    admin_sprr = admin["privacy_reviews"].sum() / admin["total_reviews"].sum() * 100
    non_admin_sprr = non_admin["privacy_reviews"].sum() / non_admin["total_reviews"].sum() * 100
    print(f"  Admin {admin_sprr:.2f}% (n={len(admin):,}) vs "
          f"non-admin {non_admin_sprr:.2f}% (n={len(non_admin):,})")

    bars = ax1.bar([0, 1], [admin_sprr, non_admin_sprr],
                   color=[C_ALERT, "#1F77B4"], width=0.55, zorder=2)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["ADMIN\npermission", "Non-ADMIN"], fontsize=9)
    ax1.set_ylim(0, 6)
    ax1.set_ylabel("SPRR (%)")
    ax1.set_title("(a) SPRR: Admin vs Non-Admin Bots", fontsize=10)

    # Labels INSIDE bars
    for i, (val, n) in enumerate(zip(
        [admin_sprr, non_admin_sprr],
        [len(admin), len(non_admin)]
    )):
        ax1.text(i, val * 0.5, f"{val:.2f}%\nn={n:,}",
                 ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    # Right: Theme distribution for admin vs non-admin bots
    if "primary_theme" in privacy_df.columns:
        admin_ids = set(admin["bot_id"].astype(str))
        non_admin_ids = set(non_admin["bot_id"].astype(str))

        priv_admin = privacy_df[privacy_df["bot_id"].astype(str).isin(admin_ids)]
        priv_non_admin = privacy_df[privacy_df["bot_id"].astype(str).isin(non_admin_ids)]

        top_themes = privacy_df["primary_theme"].value_counts().head(8).index.tolist()

        admin_counts = priv_admin[priv_admin["primary_theme"].isin(top_themes)]["primary_theme"].value_counts()
        non_admin_counts = priv_non_admin[priv_non_admin["primary_theme"].isin(top_themes)]["primary_theme"].value_counts()

        admin_pcts = (admin_counts / len(priv_admin) * 100).reindex(top_themes, fill_value=0)
        non_admin_pcts = (non_admin_counts / len(priv_non_admin) * 100).reindex(top_themes, fill_value=0)

        y_pos = np.arange(len(top_themes))
        ax2.barh(y_pos - 0.18, admin_pcts, height=0.35, color=C_ALERT,
                 label="Admin Bots", zorder=2)
        ax2.barh(y_pos + 0.18, non_admin_pcts, height=0.35, color="#1F77B4",
                 label="Non-Admin Bots", zorder=2)

        short_names = [t[:22] + "..." if len(t) > 25 else t for t in top_themes]
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(short_names, fontsize=8)
        ax2.set_xlabel("% of Security and Privacy Reviews")
        ax2.set_title("(b) Theme Distribution: Admin vs Non-Admin", fontsize=10)
        ax2.grid(axis="x", alpha=0.3)
        ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    save_fig(fig, FIGURES_DIR / "fig12_admin_comparison.png")


# =====================================================================
# Fig. 14: Feature Correlation Heatmap
# =====================================================================
def fig_feature_correlations(bot_df):
    df = bot_df[bot_df["total_reviews"] >= 5].copy()

    feature_map = {}
    if "num_perms" in df.columns:
        feature_map["Num Permissions"] = df["num_perms"].fillna(0)
    if "num_high_risk" in df.columns:
        feature_map["Num High-Risk"] = df["num_high_risk"].fillna(0)
    if "server_count" in df.columns:
        feature_map["Log10 Servers"] = np.log10(df["server_count"].clip(lower=1).fillna(1))
    if "average_rating" in df.columns:
        feature_map["Avg Rating"] = df["average_rating"].fillna(0)
    if "has_admin" in df.columns:
        feature_map["Has Admin"] = df["has_admin"].astype(int)
    if "has_github" in df.columns:
        feature_map["Has GitHub"] = df["has_github"].fillna(False).astype(int)
    feature_map["SPRR (%)"] = df["sprr"]

    if len(feature_map) < 3:
        print("  [SKIP] Not enough features for heatmap")
        return

    feat_df = pd.DataFrame(feature_map).dropna()
    corr_matrix = feat_df.corr(method="spearman")

    # Persist the SPRR column of the matrix so the correlations behind Fig. 14 are
    # available in machine-readable form and always match the rendered figure.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (corr_matrix["SPRR (%)"]
        .drop(labels=["SPRR (%)"])
        .rename("spearman_rho")
        .rename_axis("feature")
        .sort_values()
        .to_csv(RESULTS_DIR / "rq3_feature_correlations.csv"))
    print("  Spearman rho with SPRR: " + ", ".join(
        f"{feat} {rho:+.2f}" for feat, rho in
        corr_matrix["SPRR (%)"].drop(labels=["SPRR (%)"]).sort_values().items()))
    print(f"  Spearman rho, Num Permissions vs Num High-Risk: "
          f"{corr_matrix.loc['Num Permissions', 'Num High-Risk']:.2f}")

    fig, ax = plt.subplots(figsize=(5, 4.2))
    fig.subplots_adjust(left=0.22, bottom=0.2, top=0.92, right=0.95)

    # Use blue-vermilion diverging colormap (no red-green)
    from matplotlib.colors import LinearSegmentedColormap
    wong_div = LinearSegmentedColormap.from_list(
        "wong_div",
        [WONG["blue"], "white", WONG["vermilion"]],
        N=256
    )

    im = ax.imshow(corr_matrix, cmap=wong_div, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr_matrix)))
    ax.set_yticks(range(len(corr_matrix)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(corr_matrix.columns, fontsize=7.5)

    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            val = corr_matrix.iloc[i, j]
            color = "white" if abs(val) > 0.5 else C_DARK
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold" if abs(val) > 0.3 else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontsize=8)
    ax.set_title("Metadata Feature Correlation Matrix (Spearman)")

    save_fig(fig, FIGURES_DIR / "fig14_correlation_heatmap.png")


# =====================================================================
# Fig. 18: GitHub Transparency vs SPRR
# =====================================================================
@plt.style.context("default")   # drawn as Fig. 18 appears in the paper
def fig_github_transparency(bot_df):
    """Horizontal bar chart: SPRR for bots with/without a public GitHub repo.

    Laid out as Fig. 18 in the paper: closed source on top, open source below,
    percentages and counts to the right of each bar.
    """
    df = bot_df[bot_df["total_reviews"] >= 5].copy()

    if "has_github" not in df.columns:
        print("  [SKIP] No GitHub data")
        return

    gh = df[df["has_github"] == True]
    no_gh = df[(df["has_github"] == False) | (df["has_github"].isna())]

    if len(gh) < 5:
        print("  [SKIP] Not enough GitHub bots")
        return

    gh_sprr = gh["privacy_reviews"].sum() / gh["total_reviews"].sum() * 100
    no_gh_sprr = no_gh["privacy_reviews"].sum() / no_gh["total_reviews"].sum() * 100
    print(f"  GitHub linked {gh_sprr:.2f}% (n={len(gh):,}) vs "
          f"not linked {no_gh_sprr:.2f}% (n={len(no_gh):,})")

    fig, ax = plt.subplots(figsize=(7, 2.1))

    y = [0, 1]                       # 0 is the bottom row, so open source sits below
    values = [gh_sprr, no_gh_sprr]
    counts = [len(gh), len(no_gh)]

    ax.barh(y, values, color="#1F77B4", height=0.6, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(["Has GitHub\n(Open Source)", "No GitHub\n(Closed Source)"])

    for i, (val, n) in enumerate(zip(values, counts)):
        ax.text(val + 0.12, i, f"{val:.2f}% (n={n:,})",
                va="center", ha="left", fontsize=8)

    ax.set_xlabel("SPRR (%)")
    ax.set_xlim(0, 8.2)
    ax.set_title("Security and Privacy Review Rate by Code Transparency", fontsize=11)

    fig.tight_layout()
    save_fig(fig, FIGURES_DIR / "fig18_github_transparency.png")
# =====================================================================
# MAIN
# =====================================================================
def main():
    print(f"{'=' * 70}")
    print(f"  RQ3 — PERMISSIONS, METADATA AND S&P CONCERNS")
    print(f"{'=' * 70}")

    bot_df, privacy_df = build_bot_df()
    print(f"  Bots: {len(bot_df):,}")
    print(f"  Bots with >=5 reviews: {(bot_df['total_reviews'] >= 5).sum():,}")

    print(f"\n[Fig. 10] Permission Count vs SPRR")
    fig_perm_count_vs_sprr(bot_df)

    print(f"\n[Fig. 11] High-Risk Permission SPRR")
    fig_high_risk_perms(bot_df)

    print(f"\n[Fig. 13] Server Count vs SPRR")
    fig_server_count_vs_sprr(bot_df)

    print(f"\n[Fig. 12] Admin vs Non-Admin")
    fig_admin_comparison(bot_df, privacy_df)

    print(f"\n[Fig. 14] Feature Correlation Heatmap")
    fig_feature_correlations(bot_df)

    print(f"\n[Fig. 18] GitHub Transparency")
    fig_github_transparency(bot_df)

    print(f"\n{'=' * 70}")
    print(f"  RQ3 COMPLETE")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print(f"  Correlations saved to: {RESULTS_DIR / 'rq3_feature_correlations.csv'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()