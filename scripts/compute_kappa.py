"""
Compute Inter-Rater Reliability (Cohen's Kappa)

Reads the completed IRR spreadsheets from both annotators, computes
pairwise kappa scores (human-human and human-LLM), and writes a
formatted results workbook with confusion matrices and per-class stats.

Run: python scripts/compute_kappa.py   (from the repository root)
Requires: data/irr/IRR_Annotator1.xlsx and data/irr/IRR_Annotator2.xlsx (fully labeled)
Output:   data/results/IRR_Results.xlsx
          (data/irr/IRR_Results.xlsx is the copy produced for the paper)
"""

# Windows consoles default to cp1252; bot names and review text are not.
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import sys
import os

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, RESULTS_DIR  # noqa: E402

ANN1_FILE  = DATA_DIR / "irr" / "IRR_Annotator1.xlsx"
ANN2_FILE  = DATA_DIR / "irr" / "IRR_Annotator2.xlsx"
OUTPUT     = RESULTS_DIR / "IRR_Results.xlsx"
ANN1_COL  = "Annotator 1 Label"
ANN2_COL   = "Annotator 2 Label"
LLM_COL    = "LLM Label"        # column D (hidden)

# ── Load sheets ──────────────────────────────────────────────────────────────

def load_labels(filepath, annotator_col):
    df = pd.read_excel(filepath, sheet_name="Labeling")
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    required = [annotator_col, LLM_COL]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: Column '{col}' not found in {filepath}")
            print(f"  Available columns: {list(df.columns)}")
            sys.exit(1)
    df = df[[annotator_col, LLM_COL]].dropna(subset=[annotator_col])
    df[annotator_col] = df[annotator_col].str.strip().str.lower()
    df[LLM_COL]       = df[LLM_COL].str.strip().str.lower()
    return df

print("Loading Annotator 1's labels...")
df_ann1 = load_labels(ANN1_FILE, ANN1_COL)

print("Loading Annotator 2's labels...")
df_ann2 = load_labels(ANN2_FILE, ANN2_COL)

# Align on same rows
n_ann1 = len(df_ann1)
n_ann2 = len(df_ann2)

if n_ann1 != n_ann2:
    print(f"WARNING: Row count mismatch — Annotator 1={n_ann1}, Annotator 2={n_ann2}")
    print("         Using intersection of completed rows.")
    min_n = min(n_ann1, n_ann2)
    df_ann1 = df_ann1.iloc[:min_n]
    df_ann2 = df_ann2.iloc[:min_n]

n = len(df_ann1)
print(f"\nRows with labels from both annotators: {n:,}")

ann1_labels = df_ann1["Annotator 1 Label"].values
ann2_labels = df_ann2["Annotator 2 Label"].values
llm_labels  = df_ann1[LLM_COL].values   # LLM labels same in both files

# ── Compute kappa ─────────────────────────────────────────────────────────────

def kappa_stats(y1, y2, name1, name2):
    valid = [(a, b) for a, b in zip(y1, y2) if a in ("privacy", "not_privacy") and b in ("privacy", "not_privacy")]
    if not valid:
        print(f"ERROR: No valid label pairs for {name1} vs {name2}")
        sys.exit(1)
    a, b = zip(*valid)
    kappa   = cohen_kappa_score(a, b)
    agree   = sum(x == y for x, y in zip(a, b)) / len(a)
    cm      = confusion_matrix(a, b, labels=["privacy", "not_privacy"])
    report  = classification_report(a, b, labels=["privacy", "not_privacy"], output_dict=True)
    return {
        "name1": name1, "name2": name2,
        "n": len(a),
        "kappa": round(kappa, 4),
        "agreement": round(agree * 100, 2),
        "confusion": cm,
        "report": report,
        "labels_a": a, "labels_b": b,
    }

pairs = [
    kappa_stats(ann1_labels, ann2_labels, "Annotator 1",  "Annotator 2"),
    kappa_stats(ann1_labels, llm_labels,  "Annotator 1", "LLM"),
    kappa_stats(ann2_labels, llm_labels,  "Annotator 2", "LLM"),
]

# ── Print to console ──────────────────────────────────────────────────────────

def kappa_interpretation(k):
    if k < 0:      return "Poor (< 0)"
    elif k < 0.20: return "Slight (0–0.20)"
    elif k < 0.40: return "Fair (0.21–0.40)"
    elif k < 0.60: return "Moderate (0.41–0.60)"
    elif k < 0.80: return "Substantial (0.61–0.80)"
    else:          return "Almost Perfect (0.81–1.0)"

print("\n" + "="*60)
print("INTER-RATER RELIABILITY RESULTS")
print("="*60)
for p in pairs:
    print(f"\n{p['name1']} vs {p['name2']}")
    print(f"  N reviewed:   {p['n']:,}")
    print(f"  Agreement:    {p['agreement']}%")
    print(f"  Cohen's κ:    {p['kappa']}  →  {kappa_interpretation(p['kappa'])}")
print("="*60)

# ── Write IRR_Results.xlsx ────────────────────────────────────────────────────

header_fill  = PatternFill("solid", fgColor="1F4E79")
header_font  = Font(bold=True, color="FFFFFF", name="Arial", size=11)
good_fill    = PatternFill("solid", fgColor="E2EFDA")   # green
warn_fill    = PatternFill("solid", fgColor="FFF2CC")   # yellow
bad_fill     = PatternFill("solid", fgColor="FCE4D6")   # red
border = Border(
    left=Side(style="thin", color="CCCCCC"),
    right=Side(style="thin", color="CCCCCC"),
    top=Side(style="thin", color="CCCCCC"),
    bottom=Side(style="thin", color="CCCCCC"),
)

def fill_for_kappa(k):
    if k >= 0.61:  return good_fill
    elif k >= 0.41: return warn_fill
    else:           return bad_fill

wb = Workbook()

# ── Sheet 1: Summary ──────────────────────────────────────────────────────────
ws = wb.active
ws.title = "Summary"
ws.column_dimensions["A"].width = 30
ws.column_dimensions["B"].width = 18
ws.column_dimensions["C"].width = 18
ws.column_dimensions["D"].width = 18
ws.column_dimensions["E"].width = 30

title = ws.cell(row=1, column=1, value="BotLens — IRR Results Summary")
title.font = Font(bold=True, size=14, name="Arial")
ws.merge_cells("A1:E1")

ws.cell(row=2, column=1, value=f"Reviews per annotator: {n:,}").font = Font(size=11, name="Arial")
ws.merge_cells("A2:E2")

headers = ["Comparison", "N", "Agreement (%)", "Cohen's κ", "Interpretation"]
for col, h in enumerate(headers, 1):
    c = ws.cell(row=4, column=col, value=h)
    c.fill = header_fill
    c.font = header_font
    c.alignment = Alignment(horizontal="center")
    c.border = border

for i, p in enumerate(pairs, 5):
    ws.cell(row=i, column=1, value=f"{p['name1']} vs {p['name2']}").border = border
    ws.cell(row=i, column=2, value=p["n"]).border = border
    ws.cell(row=i, column=3, value=p["agreement"]).border = border

    kc = ws.cell(row=i, column=4, value=p["kappa"])
    kc.border = border
    kc.fill = fill_for_kappa(p["kappa"])
    kc.font = Font(bold=True, name="Arial", size=11)
    kc.alignment = Alignment(horizontal="center")

    ic = ws.cell(row=i, column=5, value=kappa_interpretation(p["kappa"]))
    ic.border = border
    ic.fill = fill_for_kappa(p["kappa"])

    for col in [1, 2, 3]:
        ws.cell(row=i, column=col).font = Font(name="Arial", size=11)
        ws.cell(row=i, column=col).alignment = Alignment(horizontal="center")

# Kappa scale legend
ws.cell(row=9, column=1, value="Kappa Scale:").font = Font(bold=True, name="Arial", size=10)
scale = [
    ("< 0.20", "Slight", bad_fill),
    ("0.21–0.40", "Fair", bad_fill),
    ("0.41–0.60", "Moderate", warn_fill),
    ("0.61–0.80", "Substantial", good_fill),
    ("0.81–1.0", "Almost Perfect", good_fill),
]
for j, (rng, label, fill) in enumerate(scale, 10):
    c1 = ws.cell(row=j, column=1, value=rng)
    c2 = ws.cell(row=j, column=2, value=label)
    for c in [c1, c2]:
        c.fill = fill
        c.font = Font(name="Arial", size=10)
        c.border = border

# ── Sheet 2: Confusion Matrices ───────────────────────────────────────────────
ws2 = wb.create_sheet("Confusion Matrices")
ws2.column_dimensions["A"].width = 22
for col in ["B", "C", "D"]: ws2.column_dimensions[col].width = 18

row_offset = 1
for p in pairs:
    title = ws2.cell(row=row_offset, column=1, value=f"{p['name1']} vs {p['name2']}")
    title.font = Font(bold=True, size=12, name="Arial")

    ws2.cell(row=row_offset+1, column=1, value="").font = Font(bold=True, name="Arial")
    ws2.cell(row=row_offset+1, column=2, value=f"Pred: privacy").font = Font(bold=True, name="Arial", color="1F4E79")
    ws2.cell(row=row_offset+1, column=3, value=f"Pred: not_privacy").font = Font(bold=True, name="Arial", color="1F4E79")
    ws2.cell(row=row_offset+1, column=4, value="Total").font = Font(bold=True, name="Arial")

    cm = p["confusion"]
    labels_r = ["True: privacy", "True: not_privacy"]
    for ri, label_r in enumerate(labels_r):
        ws2.cell(row=row_offset+2+ri, column=1, value=label_r).font = Font(bold=True, name="Arial", color="1F4E79")
        for ci in range(2):
            val = int(cm[ri][ci])
            c = ws2.cell(row=row_offset+2+ri, column=2+ci, value=val)
            c.alignment = Alignment(horizontal="center")
            c.font = Font(name="Arial", size=11)
            c.border = border
            # Highlight diagonal (agreements) in green
            if ri == ci:
                c.fill = good_fill
        ws2.cell(row=row_offset+2+ri, column=4, value=int(cm[ri].sum())).font = Font(name="Arial", size=11)

    # Per-class stats
    rep = p["report"]
    ws2.cell(row=row_offset+5, column=1, value="Class").font = Font(bold=True, name="Arial")
    ws2.cell(row=row_offset+5, column=2, value="Precision").font = Font(bold=True, name="Arial")
    ws2.cell(row=row_offset+5, column=3, value="Recall").font = Font(bold=True, name="Arial")
    ws2.cell(row=row_offset+5, column=4, value="F1").font = Font(bold=True, name="Arial")
    for ri, cls in enumerate(["privacy", "not_privacy"]):
        if cls in rep:
            ws2.cell(row=row_offset+6+ri, column=1, value=cls).font = Font(name="Arial", size=10)
            ws2.cell(row=row_offset+6+ri, column=2, value=round(rep[cls]["precision"], 3)).font = Font(name="Arial", size=10)
            ws2.cell(row=row_offset+6+ri, column=3, value=round(rep[cls]["recall"], 3)).font = Font(name="Arial", size=10)
            ws2.cell(row=row_offset+6+ri, column=4, value=round(rep[cls]["f1-score"], 3)).font = Font(name="Arial", size=10)

    row_offset += 11

# ── Sheet 3: Raw Labels ───────────────────────────────────────────────────────
ws3 = wb.create_sheet("Raw Labels")
ws3.column_dimensions["A"].width = 8
ws3.column_dimensions["B"].width = 18
ws3.column_dimensions["C"].width = 18
ws3.column_dimensions["D"].width = 18
ws3.column_dimensions["E"].width = 18
ws3.column_dimensions["F"].width = 18

raw_headers = ["#", "Annotator 1", "Annotator 2", "LLM", "Ann1=Ann2?", "Ann1=LLM?"]
for col, h in enumerate(raw_headers, 1):
    c = ws3.cell(row=1, column=col, value=h)
    c.fill = header_fill
    c.font = header_font
    c.alignment = Alignment(horizontal="center")
    c.border = border

for i in range(n):
    r = i + 2
    k  = ann1_labels[i]
    d  = ann2_labels[i]
    ll = llm_labels[i]

    ws3.cell(row=r, column=1, value=i+1).border = border
    ws3.cell(row=r, column=2, value=k).border = border
    ws3.cell(row=r, column=3, value=d).border = border
    ws3.cell(row=r, column=4, value=ll).border = border

    agree_kd = ws3.cell(row=r, column=5, value="✓" if k == d else "✗")
    agree_kd.border = border
    agree_kd.alignment = Alignment(horizontal="center")
    agree_kd.fill = good_fill if k == d else bad_fill

    agree_kl = ws3.cell(row=r, column=6, value="✓" if k == ll else "✗")
    agree_kl.border = border
    agree_kl.alignment = Alignment(horizontal="center")
    agree_kl.fill = good_fill if k == ll else bad_fill

    for col in range(1, 7):
        ws3.cell(row=r, column=col).font = Font(name="Arial", size=10)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
wb.save(OUTPUT)
print(f"\nResults saved to: {OUTPUT}")