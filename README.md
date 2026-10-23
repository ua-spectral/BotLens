# BotLens: A Security and Privacy Auditing Framework for Third-Party Community-Based Bots on Social Platforms

**Paper:** Kofi Ampomah and Tu Le. *Measuring Users' Security and Privacy Concerns
About Third-Party Community Bots on Discord.* Network and Distributed System
Security (NDSS) Symposium 2027. https://dx.doi.org/10.14722/ndss.2027.230535

> **Artifact reviewers:** start with [`ARTIFACT-EVALUATION.md`](ARTIFACT-EVALUATION.md).
> It maps every claim in the paper to the exact command that reproduces it, lists
> expected outputs and runtimes, and says which components are out of scope.

## Overview

This repository contains the source code for the BotLens pipeline, which analyzes
security and privacy (S&P) concerns in third-party community-based bots on Discord
through large-scale review analysis and metadata auditing. The pipeline processes
183,142 user reviews across 23,102 bots scraped from [Top.gg](https://top.gg),
identifies 8,832 reviews as S&P-relevant — an observed pipeline yield of 4.82% —
and maps them onto a 31-theme S&P taxonomy.

## Repository Structure

```
BotLens/
├── README.md                          # This file
├── ARTIFACT-EVALUATION.md             # Claim → command → expected output
├── LICENSE                            # See License below
├── CITATION.cff                       # How to cite (also used by Zenodo)
├── botlens-ndss-data.zip              # The data; unpack to data/ (see Data below)
├── requirements-lock.txt              # Exact versions used for the paper
├── requirements.txt                   # Looser version bounds
│
├── config/                            # Pipeline configuration
│   ├── __init__.py
│   ├── settings.py                    # Paths, model params, thresholds
│   └── taxonomy.py                    # 31-theme S&P taxonomy (single source of truth)
│
├── scrapers/                          # Data collection scripts
│   ├── topgg_metadata_scraper.py      # Bot metadata scraper (Top.gg pages)
│   ├── topgg_reviews_scraper.py       # Bot review scraper (Top.gg reviews)
│   └── discord_server_scanner.py      # Server bot scanner — OUT OF SCOPE, see below
│
├── scripts/                           # Pipeline stages + analysis
│   ├── stage0_preprocess.py           # Merge, clean, detect language, translate
│   ├── stage1_llm_labeling.py         # Binary S&P classification via Ollama
│   ├── stage1b_active_learning.py     # Active learning loop
│   ├── stage2_prepare_data.py         # Train/val/test splits with balancing
│   ├── stage3_finetune_deberta.py     # Fine-tune DeBERTa-v3-large classifier
│   ├── stage4_classify_all.py         # Classify all 183K reviews
│   ├── stage5_theme_assignment.py     # Assign 31-theme taxonomy themes
│   ├── stage6_emotion_classification.py # GoEmotions → Ekman-style groups
│   ├── stage7_analysis.py             # Sec. IV-A: SPRR, TPR, per-tag/per-bot tables
│   ├── stage8_visualizations.py       # Figs. 2–9
│   ├── rq3_permissions_metadata.py    # RQ3 (Sec. IV-D): statistics + Figs. 10–14, 18
│   ├── language_affective_analysis.py # Appendix B: Fig. 17
│   ├── experiment_labeling_pct.py     # Labeling-budget experiment (10–45%)
│   ├── create_irr_sheets.py           # Inter-rater reliability labeling sheets
│   ├── compute_kappa.py               # Cohen's kappa (Sec. III-C3)
│   ├── check_langs.py                 # Language distribution (Table IV; needs raw crawl)
│   └── prep_30pct_for_al.py           # Prepare labels for active learning
│
├── tests/
│   └── test_smoke.py                  # Fast integrity checks, no data required
│
└── mcp_server/                        # BotLens MCP server (Sec. V)
    ├── server.py                      # The six tools of Sec. V-B2 (+ internal helpers)
    ├── example_client.py              # MCP SDK client reproducing Use Cases 1 and 2
    └── benchmark_latency.py           # Sec. V-D latency session
```

Every figure the scripts write is named after its paper figure number
(`figures/fig02_sprr_over_time.png` is Fig. 2, and so on). The scripts produce
exactly the figures in the paper: Figs. 2–14, 17 and 18. Fig. 1 is the pipeline
schematic and Figs. 15–16 are screenshots of the BotLens report UI.

## Getting the artifact

The artifact is released at **https://github.com/ua-spectral/BotLens**. Each GitHub
release is archived permanently on Zenodo with a DOI; the DOI is given in the
paper's artifact appendix and on the repository's release page.

## Data

The data ships in the repository as `botlens-ndss-data.zip` (41 MB; one file
unpacks to 515 MB, beyond GitHub's per-file limit). Unpack it once, in the
repository root:

```bash
python -m zipfile -e botlens-ndss-data.zip .
```

This creates `data/` beside `config/`; nothing else needs downloading. The layout is:

```
BotLens/
└── data/
    ├── processed/
    │   ├── all_reviews_classified.csv    # 183,142 rows — required by the MCP server
    │   ├── privacy_reviews_full.csv      # 8,832 S&P reviews w/ themes + affect
    │   ├── bot_metadata_cleaned.csv      # 23,102 bots (Table I, MCP fallback)
    │   ├── train.csv / val.csv / test.csv
    │   └── label_map.json
    ├── results/                          # Tables behind the paper's numbers
    ├── reference_figures/                # Our output of every generated figure
    └── irr/                              # Annotator workbooks (Sec. III-C3)
```

Verify integrity with `sha256sum -c SHA256SUMS.txt` from inside `data/`.
`data/README.md` describes every file.

## Pipeline Architecture

| Stage | Script | Description | Key Output |
|-------|--------|-------------|------------|
| 0 | `stage0_preprocess.py` | Merge reviews + metadata, language detection, Argos Translate | `data/raw/reviews.csv` |
| 1 | `stage1_llm_labeling.py` | Ollama/Llama 3 8B binary S&P classification | `phase1_binary_labels.csv` |
| 1b | `stage1b_active_learning.py` | Uncertainty sampling + majority-vote labels | Improved training set |
| 2 | `stage2_prepare_data.py` | Confidence filtering, 5:1 downsampling, stratified splits | `train/val/test.csv` |
| 3 | `stage3_finetune_deberta.py` | DeBERTa-v3-large fine-tuning (F1 = 0.938 at 30% labeling) | `best_model/` |
| 4 | `stage4_classify_all.py` | Batch inference on all 183,142 reviews | `all_reviews_classified.csv` |
| 5 | `stage5_theme_assignment.py` | Ollama assigns 31-theme taxonomy themes | `privacy_reviews_themed.csv` |
| 6 | `stage6_emotion_classification.py` | GoEmotions (RoBERTa) → 8 affect groups | `privacy_reviews_full.csv` |
| 7 | `stage7_analysis.py` | SPRR, TPR, per-tag and per-bot SPRR, affect by theme | `data/results/analysis_summary.json` |
| 8 | `stage8_visualizations.py` | Figs. 2–9, including the Fig. 2 OLS trend (R² = 0.35) | `figures/fig02_*` … `fig09_*` |

## Key Metrics

- **SPRR** (Security and Privacy Review Rate): the percentage of reviews that our
  pipeline identified and retained as S&P-relevant. Overall SPRR = 4.82%. This is a
  pipeline-identified review rate, not a measure of verified vulnerabilities,
  confirmed misconduct, or bot-level risk.
- **TPR** (Theme Prevalence Rate): distribution of S&P reviews across the 31-theme
  taxonomy, excluding reviews labeled `unassigned`.

## Taxonomy

The 31-theme S&P taxonomy (paper Table V) comprises **20 themes inherited from
Akgul et al. (USENIX Security 2024)** and **11 themes developed inductively in this
study** — 2 additional Security & Safety themes and 9 Discord-specific themes.
`config/taxonomy.py` is the single source of truth; `stage8_visualizations.py`
derives its groupings from it.

**Theme Group 1 — Data Practices (7, all inherited):**
Data Collection, Data Sharing, Data Deletion, Remove Personally Identifiable
Information, Data Exposure, Data Hiding, Location and Tracking

**Theme Group 2 — User Rights & Controls (7, all inherited):**
Consent, Privacy Controls, Anonymity / Identification, Advertising, Privacy Policies
and Laws, Privacy Invasion, Positive Privacy

**Theme Group 3 — Security & Safety (6 inherited + 2 new):**
Data Security, Password Issues, Data Accuracy, Safety, Selling Data, Surveillance,
*Phishing and Scams* (new), *Data Retention* (new)

**Theme Group 4 — Discord-Specific Risks (9, all new):**
Over-Listening, Permission Scope Mismatch, Message Content Access, DM Privacy,
Gateway Intent Abuse, Webhook Misuse, Server Profiling, Voice Privacy, Bot
Moderation Transparency

## Data Collection

### Metadata Scraper (`scrapers/topgg_metadata_scraper.py`)
Collects public bot metadata from Top.gg using parallel undetected-chromedriver
instances, across 90+ category tags, with auto-restart and progress persistence.

### Reviews Scraper (`scrapers/topgg_reviews_scraper.py`)
Scrapes public user reviews from Top.gg bot pages using async Playwright with 4
concurrent browser instances. Handles pagination, deduplication, and JSON export per
bot. Reviewer display names are visible on the public page but are deliberately
**not** read or retained; only review content and public metadata are collected.

### Discord Server Scanner (`scrapers/discord_server_scanner.py`) — out of scope
This script automates a login to a **real Discord user account** to enumerate APP-badged members of a server, producing the bot manifest used to
demonstrate `scan_server_bots` (paper Use Case 2).

**Artifact reviewers should not run this script.** Automating a user account is
contrary to Discord's Terms of Service, and running it requires putting real
credentials in `.env`. It is included only for completeness of the record. To
exercise `scan_server_bots` without it, pass any text file containing one bot name
per line — see `ARTIFACT-EVALUATION.md`.

## MCP Server

`mcp_server/server.py` exposes the following tools over the Model Context Protocol,
providing on-demand S&P decision support in any MCP-compatible client.

### The six tools described in the paper

1. **`get_bot_signals`** — SPRR, star rating, requested permissions, reported S&P
   themes, and sentiment composition for a named bot.
2. **`get_taxonomy_breakdown`** — frequency distribution of the 31 taxonomy themes
   in a bot's S&P reviews, each review counted under its primary theme.
3. **`get_review_samples`** — representative S&P-relevant review text.
4. **`compare_bots`** — side-by-side comparison of two or more bots.
5. **`list_high_salience_bots`** — top-*k* bots ranked by SPRR.
6. **`scan_server_bots`** — accepts a bot manifest and returns a server-wide
   comparative report.

These tools report the *salience* of S&P discussion. They do not verify the claims
in reviews and do not determine whether a bot is safe.

The tools follow Algorithm 1 (Appendix A). A bot listed on Top.gg that has no
reviews in the dataset (10,637 of the 23,102 bots) has no SPRR, so
`get_bot_signals`, `compare_bots` and `scan_server_bots` fall back to a permission
audit from its Top.gg metadata (`PermissionAudit`) instead of reporting it as not
found.

### Auxiliary helpers (not part of the paper's contribution)

Used for development, benchmarking, and ad-hoc exploration; these are plain
functions and are **not** registered as MCP tools:
`get_permissions_audit`, `search_bots`, `filter_bots_by_theme`,
`get_dataset_stats`, `diagnose`, `get_timing_stats`.

### Performance

All classification decisions are precomputed offline, and the server builds per-bot
indexes when it starts, so tool calls reduce to dictionary lookups with no model
inference at query time. Measured latency across the 22-query session reported in
the paper: `get_bot_signals` averaged 1.7 ms (11 calls), `get_taxonomy_breakdown`
1.6 ms (1 call), `get_review_samples` 1.9 ms (4 calls), and `scan_server_bots` — the
heaviest — averaged 14.7 ms with a peak of 31.3 ms (6 calls). Session-wide mean:
5.0 ms. `python mcp_server/benchmark_latency.py` replays a session of the same
composition; every tool stays under 2 ms.

## Visualization Standard

Each figure is drawn as it appears in the paper. Most use the Wong (2011)
colorblind-safe palette (*Nature Methods* 8, 441); Figs. 12, 13 and 18 use
matplotlib's default style, as printed:

| Color | Hex | Role |
|-------|-----|------|
| Blue | `#0072B2` | Primary data |
| Orange | `#E69F00` | Moderate / accent |
| Green | `#009E73` | Positive |
| Vermilion | `#D55E00` | Alert |
| Sky Blue | `#56B4E9` | Secondary / comparison |
| Purple | `#CC79A7` | Additional series |
| Yellow | `#F0E442` | Supplementary |
| Black | `#000000` | Text / outlines |



## Requirements

### Hardware
- GPU: NVIDIA GPU with ≥16 GB VRAM recommended for Stages 1–4 (RTX 5090, 32 GB, used
  in the paper). Stages 7–8 and the MCP server are CPU-only.
- RAM: ≥32 GB recommended for the full pipeline.
- Disk: ~570 MB for the data component. Training your own classifier needs
  substantially more — the production run retained ~52 GB of epoch checkpoints.

### Software
- Python 3.11+
- CUDA 12.8 and `torch>=2.7` if using an RTX 5090 / Blackwell GPU (earlier torch
  builds do not support sm_120).
- Ollama with `llama3:8b` pulled (Stages 1, 1b, 5 only).
- See `requirements.txt`.

## Setup

```bash
pip install -r requirements-lock.txt
ollama pull llama3:8b        # only needed for Stages 1, 1b, 5
```

Unpack the data first (see **Data** above). Run every script from the
repository root; `config/settings.py` resolves paths relative to it, so no path
editing is needed.

## Running the Pipeline

Stages 0–4 regenerate the dataset from scratch and take several days on a single
GPU. To verify the paper's reported results, start from the distributed data and run
Stages 7–8 only — see `ARTIFACT-EVALUATION.md`.

```bash
# Stage 0: Preprocess and merge
python scripts/stage0_preprocess.py \
    --reviews data/raw/bot_reviews_cleaned.csv \
    --metadata data/raw/bot_metadata_cleaned.csv

# Stage 1: LLM labeling (requires Ollama running)
python scripts/stage1_llm_labeling.py --input data/raw/reviews.csv

# Stage 1b: Active learning (optional, improves classifier)
python scripts/stage1b_active_learning.py --rounds 5 --samples-per-round 500

# Stage 2: Prepare training data
python scripts/stage2_prepare_data.py

# Stage 3: Fine-tune DeBERTa
python scripts/stage3_finetune_deberta.py

# Stage 4: Classify all reviews
python scripts/stage4_classify_all.py --input data/raw/reviews.csv

# Stage 5: Theme assignment (requires Ollama running)
python scripts/stage5_theme_assignment.py

# Stage 6: Affective-expression classification
python scripts/stage6_emotion_classification.py

# Stage 7: Analysis
python scripts/stage7_analysis.py

# Stage 8: Visualizations
python scripts/stage8_visualizations.py
```

## Labeling-Budget Experiment

```bash
python scripts/experiment_labeling_pct.py
# Evaluates 10%, 20%, 30%, 40%, 45%; highest observed F1 at 30% (F1 = 0.938)
```

Each budget uses its own stratified split, so this is an exploratory comparison
rather than a controlled one on a shared test set (paper Sec. III-C1a).

## Inter-Rater Reliability

The completed annotation sheets ship in `data/irr/`. To recompute Cohen's kappa from
them:

```bash
python scripts/compute_kappa.py     # reads data/irr/, writes data/results/IRR_Results.xlsx
```

`create_irr_sheets.py` is the script that generated those sheets. It reads
`data/labeled/experiment_labels_accumulated.csv`, which is not redistributed, so it
cannot be re-run from the distributed data; it is included as a record of how the
sample was drawn.

## Ethical Considerations

- All scraped data is publicly available on Top.gg. Our institution's IRB determined
  that review was not required for this study.
- Reviewer display names are not collected or retained; findings are reported in
  aggregate and no PII from reviews is published.
- `discord_server_scanner.py` requires authenticating as a Discord user and is
  **out of scope for artifact evaluation** (see above). No credentials are stored by
  the code; it reads them from `.env` at runtime.
- SPRR reports the salience of S&P discussion, not verified misconduct. A high SPRR
  does not mean a bot is unsafe, and a low SPRR does not mean it is safe. The
  framework is designed to support user and administrator decisions, not to issue
  verdicts.

## License

Copyright © 2026 Kofi Ampomah and Tu Le, The University of Alabama.

- **Code** (everything outside `data/`): [MIT License](LICENSE).
- **Data** (everything in `data/`): CC BY-NC-SA 4.0. The CC BY-NC-SA 4.0 license
  applies strictly to the annotations, labels, and dataset compilation created by
  the authors. The underlying data remains subject to the original creators'
  terms. See `data/LICENSE`.

If you use this artifact, please cite the paper above.
