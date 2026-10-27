#!/usr/bin/env python3
"""Regenerate every manuscript figure (PDF + 300 dpi PNG) and CAPTIONS.md from committed result files.

Usage (from the project root):  .venv/bin/python manuscript/figures/make_figures.py [--worktree]

Every plotted number is read from a committed results file through the selector language of
``scripts/check_report_numbers.py`` (json / md selectors; the same ones REPORT.md's source tags use) or
from a committed CSV. By default files are read from git ``HEAD``; ``--worktree`` reads the working tree.
Each value is logged with its selector in ``figure_data.csv`` next to the figures, so any point can be
traced back to its source. No number is typed by hand except method parameters in the Fig 1 schematic
(taken from docs/METHOD.md, which has no machine-readable form).

Sources:
  results/test/v{1,2}/summary.json        pre-registered families, secondary metrics, strata, test ablations
  results/test/v2/posthoc.json            post-hoc ligand-class / LIGYSIS_NI / O6-interface fold strata
  results/test/v2/DESCRIPTIVE.md          fast preset, AF naive / pLDDT filter, isolated-chain dimer rows
  results/validation/ligysis_reproduction/reproduction.csv   published predictions rescored by our harness
  results/baselines/HEADLINES.md          P2Rank 2.5.1 / fpocket 4.2.3 re-runs (cited by REPORT.md)
  docs/PERF.md                            final like-for-like timing and Track T1 throughput
  results/ablations/ablations_long.csv    dev ablations (ABLATIONS.md is generated from it)
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_report_numbers as crn  # noqa: E402

# ----------------------------------------------------------------------------------------------
# Style: one palette for every figure (dataviz reference palette, validated with
# validate_palette.js: blue/orange/aqua pass all-pairs; yellow/violet pass; ordinal light->dark blue pass).
# Colour follows the entity: BURROW = blue, fpocket(+PRANK) = orange, P2Rank = aqua, other published = gray.
# ----------------------------------------------------------------------------------------------
BLUE = "#2a78d6"        # BURROW v2 (and its presets)
BLUE_LIGHT = "#86b6ef"  # BURROW v1 (ordinal step of the same hue: earlier version)
ORANGE = "#eb6834"      # fpocket / fpocket+PRANK
AQUA = "#1baf7a"        # P2Rank
GRAY = "#7a7974"        # other published methods / neutral series
YELLOW = "#eda100"      # Fig 5C: 24 CPU workers
VIOLET = "#6250d6"      # Fig 5C: 3 GPUs
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LEARN_FILL = "#e6f0fb"  # Fig 1: learned step (tint of BLUE)

MAX_W, MAX_H = 7.5, 8.75  # PLOS limits, inches
FS = 8                    # minimum font size used anywhere


def setup_fonts() -> str:
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Helvetica"):
        if name in avail:
            return name
    fdir = HERE / "fonts"
    for ttf in sorted(fdir.glob("LiberationSans-*.ttf")):
        font_manager.fontManager.addfont(str(ttf))
    avail = {f.name for f in font_manager.fontManager.ttflist}
    return "Liberation Sans" if "Liberation Sans" in avail else "DejaVu Sans"


FONT = setup_fonts()
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS,
    "axes.labelsize": FS,
    "axes.titlesize": FS + 0.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 4,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.0,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.2,
    "legend.columnspacing": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.dpi": 100,
    "lines.solid_capstyle": "round",
})

# ----------------------------------------------------------------------------------------------
# Sourcing
# ----------------------------------------------------------------------------------------------
SRC: crn.Source = crn.Source("HEAD")
LOG: list[tuple[str, str, float, str]] = []
CUR_FIG = ""


def S(sel: str, item: str = "") -> float:
    """Resolve a check_report_numbers selector and log it for the current figure."""
    v = crn.resolve(sel, SRC)
    LOG.append((CUR_FIG, item, v, sel))
    return v


def S_opt(sel: str, item: str = "") -> float | None:
    try:
        return S(sel, item)
    except crn.SelectorError:
        return None


def flt(**kw) -> str:
    return "{" + ";".join(f"{k}={v}" for k, v in kw.items()) + "}"


def csv_rows(path: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(SRC.text(path))))


def C(path: str, key_cols: dict, col: str, item: str = "", scale: float = 1.0) -> float:
    rows = [r for r in csv_rows(path) if all(r[k] == v for k, v in key_cols.items())]
    if len(rows) != 1:
        raise crn.SelectorError(f"{path}: {key_cols} matched {len(rows)} rows")
    v = float(rows[0][col]) * scale
    keys = ";".join(f"{k}={v_}" for k, v_ in key_cols.items())
    LOG.append((CUR_FIG, item, v, f"csv:{path}#{{{keys}}}/{col}" + (f"*{scale:g}" if scale != 1 else "")))
    return v


def J(alias: str, listname: str, filt: dict, field: str):
    """Raw (non-numeric) field of a summary.json list element, e.g. a pass flag; logged like S()."""
    hits = [e for e in SRC.json(crn.ALIASES[alias])[listname]
            if all(str(e.get(k)) == str(v) for k, v in filt.items())]
    if len(hits) != 1:
        raise crn.SelectorError(f"{alias}:{listname}{filt} matched {len(hits)}")
    v = hits[0][field]
    LOG.append((CUR_FIG, f"{alias} {listname} {filt} {field}={v}", float(v) if isinstance(v, (bool, int, float)) else float("nan"),
                f"{alias}:{listname}{flt(**filt)}/{field}"))
    return v


def sec(alias: str, bench: str, method: str, metric: str, field: str = "value", pct: bool = True) -> float | None:
    """descriptive/secondary entry of a summary.json (fractions -> %)."""
    sel = f"{alias}:descriptive/secondary{flt(benchmark=bench, method=method, metric=metric)}/{field}"
    if pct:
        sel += "*100"
    return S_opt(sel, f"{alias} {bench} {method} {metric} {field}")


def sec3(alias, bench, method, metric, pct=True):
    return (sec(alias, bench, method, metric, "value", pct), sec(alias, bench, method, metric, "ci_low", pct),
            sec(alias, bench, method, metric, "ci_high", pct))


# DESCRIPTIVE.md / HEADLINES.md / PERF.md row helpers -------------------------------------------
DESC = "results/test/v2/DESCRIPTIVE.md"
HEAD = "results/baselines/HEADLINES.md"
PERF = "docs/PERF.md"
REPRO = "results/validation/ligysis_reproduction/reproduction.csv"
ABL = "results/ablations/ablations_long.csv"


def esc(s: str) -> str:
    out = ""
    for ch in s:
        out += "\\" + ch if ch in ".+*?()[]{}|^$\\" else ch
    return out


def desc_cell(section: str, cells: list[str], col: int, k: int = 0, item: str = "") -> float:
    row = r"^\| " + r" \| ".join(esc(c) for c in cells) + r" \|"
    return S(f"md:{DESC}::{section}::{row}::c{col}.{k}", item or " | ".join(cells))


def headline(section: str, variant: str, col: int) -> tuple[float, float, float]:
    row = r"^\| " + esc(variant) + r" \|"
    return tuple(S(f"md:{HEAD}::^## {section}$::{row}::c{col}.{k}", f"HEADLINES {section} {variant} c{col}.{k}")
                 for k in range(3))


# ----------------------------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------------------------
def head(ax, letter, title="", x=None, dy=0.012, tfs=None):
    """Panel letter (10 pt bold) and title at the top-left of an axes, in figure coordinates."""
    fig = ax.figure
    bb = ax.get_position()
    x = bb.x0 if x is None else x
    fig.text(x, bb.y1 + dy, letter, fontsize=10, fontweight="bold", ha="left", va="bottom", color=INK)
    if title:
        fig.text(x + 0.03, bb.y1 + dy, title, fontsize=tfs or FS + 0.5, fontweight="bold", ha="left",
                 va="bottom", color=INK)


def panel_label(ax, label, x=-0.02, y=1.02, fig_coords=None):
    if fig_coords is not None:
        ax.figure.text(*fig_coords, label, fontsize=10, fontweight="bold", va="top", ha="left", color=INK)
    else:
        ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right",
                color=INK)


def xgrid(ax):
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def dot(ax, x, y, lo=None, hi=None, color=BLUE, marker="o", ms=5.0, filled=True, z=3, lw=1.1):
    if lo is not None and hi is not None:
        ax.plot([lo, hi], [y, y], color=color, lw=lw, zorder=z, solid_capstyle="butt")
    ax.plot([x], [y], marker=marker, ms=ms, mfc=color if filled else "white", mec=color if not filled else "white",
            mew=1.0 if not filled else 0.7, ls="none", zorder=z + 1)


def vline(ax, x, ls="-", color=INK2, lw=0.6, z=1, **kw):
    ax.axvline(x, color=color, lw=lw, ls=ls, zorder=z, **kw)


def save(fig, name):
    w, h = fig.get_size_inches()
    assert w <= MAX_W + 1e-6 and h <= MAX_H + 1e-6, (name, w, h)
    fig.savefig(HERE / f"{name}.pdf")
    fig.savefig(HERE / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {name}.pdf/.png ({w:.2f} x {h:.2f} in)")


def f1(v):
    return "–" if v is None else f"{v:.1f}"


def signed(v):
    return f"{v:+.1f}".replace("-", "−")


def leg_handle(color, marker="o", filled=True, ms=5, line=False):
    return Line2D([], [], color=color, lw=1.1 if line else 0, marker=marker, ms=ms,
                  mfc=color if filled else "white", mec=color if not filled else "white", mew=1.0)


CAPTIONS: dict[str, str] = {}


# ==============================================================================================
# Fig 1 — pipeline schematic
# ==============================================================================================
def fig1():
    global CUR_FIG
    CUR_FIG = "fig1_pipeline"
    fig = plt.figure(figsize=(7.5, 3.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 3.3)
    ax.axis("off")

    # (title, body, kind) — kind: 'io' input/output, 'geo' analytical geometry, 'learn' learned
    row1 = [
        ("Input structure", "PDB / mmCIF chain,\nbiological assembly\n(any no. of chains)\nor AlphaFold model", "io"),
        ("Polymer-only atoms", "HETATM, waters, ions\nremoved; canonical\nPCA frame\nAF mode: prune\npLDDT < 50 segments;\nsoft occlusion weights", "geo"),
        ("Candidate points", "1 Å solvent-shell grid\nLIGSITE PSP cascade\n(PSP ≥ 2 + coarse\nsub-lattice)\n+ metal-coordination\ncandidates", "geo"),
        ("Per-point features", "multi-scale ray\nburiedness: full-atom\n& backbone-only\nfirst-hit chemistry\natom / residue\ndensities; WCN;\nprotrusion", "geo"),
        ("Point scorer", "LightGBM classifier\n→ ligand-contact\nprobability\nper point", "learn"),
    ]
    row2 = [
        ("Pocket formation", "multi-scale single\nlinkage at several\nburiedness & score\nthresholds;\nchaining control", "geo"),
        ("Pocket ranker", "cross-fitted LightGBM\non pocket descriptors\n→ calibrated score,\ncomparable across\nproteins", "learn"),
        ("Suppression", "greedy NMS:\ncentres < 10 Å apart,\nresidue / point\noverlap", "geo"),
        ("Output floor (v2)", "drop pockets with\nscore < 0.03 once\n3 are kept\n(--min-score none\n= v1 list)", "geo"),
        ("Ranked pockets", "rank, score, centre,\nresidues, points\n(CSV)", "io"),
    ]
    n = 5
    left, right = 0.12, 7.38
    gap = 0.26
    bw = (right - left - gap * (n - 1)) / n
    bh = 1.2
    y1, y2 = 2.02, 0.42  # box bottoms

    def box(x, y, title, body, kind):
        if kind == "learn":
            fc, ec, lw = LEARN_FILL, BLUE, 1.0
        elif kind == "io":
            fc, ec, lw = "#f3f2ee", INK2, 0.8
        else:
            fc, ec, lw = "white", MUTED, 0.8
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + 0.08, y + bh - 0.09, title, ha="left", va="top", fontsize=FS + 0.5, fontweight="bold",
                color=INK, zorder=3)
        ax.text(x + 0.08, y + bh - 0.32, body, ha="left", va="top", fontsize=FS, color=INK2, zorder=3,
                linespacing=1.18)

    def arrow(x0, y0, x1, y1_):
        ax.annotate("", xy=(x1, y1_), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=INK2, lw=0.9,
                                    shrinkA=0, shrinkB=0), zorder=1)

    xs = [left + i * (bw + gap) for i in range(n)]
    for i, (t, b, k) in enumerate(row1):
        box(xs[i], y1, t, b, k)
        if i:
            arrow(xs[i - 1] + bw + 0.02, y1 + bh / 2, xs[i] - 0.02, y1 + bh / 2)
    for i, (t, b, k) in enumerate(row2):
        box(xs[i], y2, t, b, k)
        if i:
            arrow(xs[i - 1] + bw + 0.02, y2 + bh / 2, xs[i] - 0.02, y2 + bh / 2)
    # wrap connector: bottom of box 5 -> top of box 6
    ymid = (y1 + y2 + bh) / 2
    xa, xb = xs[-1] + bw / 2, xs[0] + bw / 2
    ax.plot([xa, xa, xb], [y1, ymid, ymid], color=INK2, lw=0.9, zorder=1, solid_capstyle="butt")
    arrow(xb, ymid, xb, y2 + bh + 0.01)
    ax.text((xa + xb) / 2, ymid + 0.05, "point scores", ha="center", va="bottom", fontsize=FS, color=INK2,
            style="italic")

    # legend
    ly = 0.13
    items = [("white", MUTED, "analytical geometry / chemistry"), (LEARN_FILL, BLUE, "learned (LightGBM)"),
             ("#f3f2ee", INK2, "input / output")]
    x = left
    for fc, ec, lab in items:
        ax.add_patch(FancyBboxPatch((x, ly - 0.06), 0.22, 0.13, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(x + 0.3, ly, lab, va="center", ha="left", fontsize=FS, color=INK2)
        x += 2.2
    save(fig, "fig1_pipeline")
    CAPTIONS["Fig1"] = (
        "**Fig 1. BURROW pipeline.** Top row: analytical geometry. Protein atoms only (ligands, waters and ions "
        "removed); in `--af-mode`, low-confidence segments (pLDDT < 50) are pruned and atoms get soft occlusion "
        "weights, w = clip((pLDDT − 40)/40, 0, 1). Candidate volume elements come from a 1 Å solvent-shell grid "
        "filtered by a LIGSITE protein–solvent–protein (PSP) cascade, plus metal-coordination candidates. Each "
        "point is described by ray-cast buriedness at several ray lengths (through a full-atom grid and a "
        "backbone-only grid), first-hit chemistry, atom- and residue-class densities, weighted contact number "
        "(WCN) and protrusion. A LightGBM classifier scores the points. Bottom row: candidate pockets are formed "
        "by single linkage at several buriedness and point-score thresholds, with chaining control. They are "
        "ranked by a LightGBM ranker trained on cross-fitted point scores, which gives scores that are "
        "comparable across proteins. Pockets whose centres lie within 10 Å of a better-ranked pocket, or that "
        "overlap it, are suppressed. The v2 output floor then drops pockets scoring below 0.03 once three are "
        "kept. Blue boxes are learned steps. Parameters as in docs/METHOD.md (FREEZE v2, default preset)."
    )


# ==============================================================================================
# Fig 2 — LIGYSIS comparison
# ==============================================================================================
PUBLISHED = ["P2Rank+Cons", "DeepPocket-Rescored", "P2Rank", "GrASP", "DeepPocket-Segmented-NR", "Ligsite-AA",
             "PocketFinder-AA", "Surfnet-AA", "fpocket", "VN-EGNN-NR", "PUResNet+PRANK", "IF-SitePred-rescored-NR"]


def fam_color(name: str) -> str:
    n = name.lower()
    if "burrow" in n:
        return BLUE
    if "p2rank" in n and "fpocket" not in n:
        return AQUA
    if "fpocket" in n:
        return ORANGE
    return GRAY


def ligysis_rows():
    """Rows for Fig 2: (group, label, {metric: (v, lo, hi)}, pockets/chain)."""
    rows = []
    B = "BURROW[burrow_all_default_s0]"
    BA = "BURROW[burrow_all_accurate_s0]"

    def burrow(alias, method, label):
        m = {k: sec3(alias, "ligysis", method, met) for k, met in
             (("topn2", "recall_topn2"), ("dca4", "recall_ldca_topn2"), ("p1k", "P1K"))}
        ppc = sec(alias, "ligysis", method, "pockets_per_structure", pct=False)
        return ("This work", label, m, ppc)

    rows.append(burrow("v2", B, "BURROW v2 (default)"))
    rows.append(burrow("v2", BA, "BURROW v2 (accurate)"))
    fsec = "^## `--preset fast`"
    fast = {
        "topn2": (desc_cell(fsec, ["ligysis", "desc.fast_all_s0", "v2:burrow_all_default_s0", "recall_topn2"], 4),
                  None, None),
        "dca4": (desc_cell(fsec, ["ligysis", "desc.fast_all_s0", "v2:burrow_all_default_s0", "recall_ldca_topn2"], 4),
                 None, None),
        "p1k": (desc_cell(fsec, ["ligysis", "desc.fast_all_s0", "v2:burrow_all_default_s0", "P1K"], 4), None, None),
    }
    rows.append(("This work", "BURROW v2 (fast preset)", fast, None))
    rows.append(burrow("v1", B, "BURROW v1 (default)"))

    # re-runs on identical inputs
    fr = "fpocket_4.2.3+PRANK_2.4.2"
    m = {k: sec3("v2", "ligysis", fr, met) for k, met in
         (("topn2", "recall_topn2"), ("dca4", "recall_ldca_topn2"), ("p1k", "P1K"))}
    rows.append(("Re-run here", "fpocket 4.2.3 + PRANK 2.4.2", m,
                 sec("v2", "ligysis", fr, "pockets_per_structure", pct=False)))
    p2 = {"topn2": headline("ligysis", "p2rank_2.5.1", 2), "dca4": headline("ligysis", "p2rank_2.5.1", 4),
          "p1k": headline("ligysis", "p2rank_2.5.1", 5)}
    rows.append(("Re-run here", "P2Rank 2.5.1", p2, None))

    # published predictions, rescored by our harness
    pub = "published:fpocket+PRANK"
    m = {k: sec3("v2", "ligysis", pub, met) for k, met in
         (("topn2", "recall_topn2"), ("dca4", "recall_ldca_topn2"), ("p1k", "P1K"))}
    prow = [("Published predictions", "fpocket+PRANK", m, sec("v2", "ligysis", pub, "pockets_per_structure", pct=False))]
    for name in PUBLISHED:
        mm = {"topn2": (C(REPRO, {"method": name}, "topN2", f"{name} topN+2", 100), None, None),
              "dca4": (None, None, None),
              "p1k": (C(REPRO, {"method": name}, "P1K", f"{name} P1K", 100), None, None)}
        prow.append(("Published predictions", name, mm, None))
    prow.sort(key=lambda r: -r[2]["topn2"][0])
    rows.extend(prow)
    return rows


def fig2():
    global CUR_FIG
    CUR_FIG = "fig2_ligysis"
    rows = ligysis_rows()
    groups = []
    for r in rows:
        if not groups or groups[-1][0] != r[0]:
            groups.append((r[0], []))
        groups[-1][1].append(r)
    # y positions with a gap + header per group
    ys, labels, y = [], [], 0.0
    headers = []
    for g, rr in groups:
        headers.append((g, y))
        y += 1.0
        for r in rr:
            ys.append(y)
            labels.append(r)
            y += 1.0
        y += 0.35
    ymax = y
    fig = plt.figure(figsize=(7.5, 4.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.0, 1.0, 0.28], left=0.255, right=0.99, top=0.885,
                          bottom=0.085, wspace=0.16)
    metrics = [("topn2", "Top-N+2, DCC ≤ 12 Å (C1)", "A"), ("dca4", "Top-N+2, DCA ≤ 4 Å", "B"),
               ("p1k", "P1K (C2)", "C")]
    axes = []
    for j, (key, xlabel, pl) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, j])
        axes.append(ax)
        xgrid(ax)
        for yy, r in zip(ys, labels):
            v, lo, hi = r[2][key]
            if v is None:
                continue
            col = fam_color(r[1])
            big = r[1] == "BURROW v2 (default)"
            dot(ax, v, yy, lo, hi, color=col, ms=6.0 if big else 4.8)
        ax.set_ylim(ymax - 0.2, -0.8)
        ax.set_yticks([])
        ax.set_xlabel("%")
        ax.tick_params(axis="x", length=2.5)
        ax.text(0.0, 1.0, pl, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")
        ax.text(0.075 if j == 0 else 0.095, 1.0, xlabel, transform=ax.transAxes, fontsize=FS, ha="left",
                va="bottom", color=INK)
    # DCA4 not available for published predictions (except fpocket+PRANK): one shaded note
    ax = axes[1]
    pub_y = [yy for yy, r in zip(ys, labels) if r[0] == "Published predictions" and r[2]["dca4"][0] is None]
    if pub_y:
        ax.text(0.5, (min(pub_y) + max(pub_y)) / 2, "not computed for\npublished predictions",
                transform=blended_transform_factory(ax.transAxes, ax.transData), ha="center", va="center",
                color=MUTED, fontsize=FS, linespacing=1.2)
    # x ranges
    axes[0].set_xlim(35, 80)
    axes[1].set_xlim(15, 62)
    axes[2].set_xlim(40, 100)
    # row labels (left) and group headers
    tr = blended_transform_factory(fig.transFigure, axes[0].transData)
    for yy, r in zip(ys, labels):
        bold = r[1] == "BURROW v2 (default)"
        axes[0].text(0.247, yy, r[1], transform=tr, ha="right", va="center", fontsize=FS, color=INK,
                     fontweight="bold" if bold else "normal")
    for g, yy in headers:
        axes[0].text(0.01, yy, g, transform=tr, ha="left", va="center", fontsize=FS, color=INK2,
                     fontweight="bold")
    # pockets per chain column
    axp = fig.add_subplot(gs[0, 3])
    axp.set_ylim(ymax - 0.2, -0.8)
    axp.axis("off")
    axp.text(0.5, -0.9, "pockets\n/ chain", ha="center", va="bottom", fontsize=FS, color=INK2,
             transform=blended_transform_factory(axp.transAxes, axp.transData))
    for yy, r in zip(ys, labels):
        axp.text(0.5, yy, f1(r[3]) if r[3] is not None else "–", ha="center", va="center", fontsize=FS,
                 color=INK if r[3] is not None else MUTED,
                 transform=blended_transform_factory(axp.transAxes, axp.transData))
    handles = [leg_handle(BLUE), leg_handle(ORANGE), leg_handle(AQUA), leg_handle(GRAY)]
    fig.legend(handles, ["BURROW", "fpocket(+PRANK)", "P2Rank", "other published methods"], loc="upper left",
               bbox_to_anchor=(0.255, 0.995), ncol=4)
    save(fig, "fig2_ligysis")

    rep_rows = csv_rows(REPRO)
    n_rep = len(rep_rows)
    gate_max = max(abs(float(r[c])) for r in rep_rows for c in ("topN_diff_pts", "topN2_diff_pts", "max_recall_diff_pts")
                   if r[c] not in ("", "nan"))
    LOG.append((CUR_FIG, "reproduction: n methods", n_rep, f"csv:{REPRO}#count"))
    LOG.append((CUR_FIG, "reproduction: max |diff| pts", gate_max, f"csv:{REPRO}#max(|*_diff_pts|)"))
    n_sites = S("v2:family1{endpoint=C1;seed_index=0;comparator=published:fpocket+PRANK}/n_rows", "C1 n sites")
    n_units = S("v2:family1{endpoint=C1;seed_index=0;comparator=published:fpocket+PRANK}/n_units", "C1 n units")
    n_dca = S("v2:family1{endpoint=C1-strict;seed_index=0;comparator=published:fpocket+PRANK}/n_rows", "DCA4 n")
    n_ch = S("v2:denominators{benchmark=ligysis;method=BURROW[burrow_all_default_s0]}/n_structures_or_units",
             "LIGYSIS chains")
    CAPTIONS["Fig2"] = (
        f"**Fig 2. Pocket prediction on the LIGYSIS test set ({n_ch:,.0f} chains, {n_sites:,.0f} binding sites).** "
        "(A) Top-N+2 success: a site counts as found when the pocket closest to its centre is within 12 Å (DCC) "
        "and ranks within N+2, where N is the number of sites on the chain (pre-registered endpoint C1). "
        f"(B) The same with a stricter hit, distance to any ligand atom (DCA) ≤ 4 Å ({n_dca:,.0f} sites with "
        "ligand atoms; C1-strict). (C) Pooled precision P1K (endpoint C2): "
        "the share of true positives among the 1,000 highest-scoring pockets pooled over all chains. "
        "Right column: mean pockets output per chain. The closest-pocket rule penalises long output lists, so "
        "top-N+2 must be read together with list length (see Fig 4). Points: seed 0 (BURROW). Bars: 95 % "
        f"paired cluster-bootstrap CIs (30 % sequence clusters, {n_units:,.0f} units, 1,000 resamples), shown "
        "where a CI was computed. 'Re-run here': P2Rank 2.5.1 and fpocket 4.2.3 + PRANK 2.4.2 run by us on "
        "identical inputs. 'Published predictions': the LIGYSIS authors' per-chain predictions, rescored by "
        f"our harness, which reproduces their published numbers within {gate_max:.2f} points (gate ±0.5) for all "
        f"{n_rep} method variants "
        "(results/validation/ligysis_reproduction.md); for these no CIs or pockets per chain were computed, "
        "and chains without predictions count as misses. The fast preset is not pre-registered (descriptive). "
        "Sources: results/test/v{1,2}/summary.json, results/test/v2/DESCRIPTIVE.md, "
        "results/baselines/HEADLINES.md, results/validation/ligysis_reproduction/reproduction.csv."
    )


# ==============================================================================================
# Fig 3 — pre-registered comparisons
# ==============================================================================================
F2_NAMES = {"O2a": "O2a COACH420", "O2b": "O2b HOLO4K", "O3": "O3 PocketWatch", "O4": "O4 CryptoBench (apo)",
            "O5": "O5 AF set", "O6": "O6 dimer interface"}
METRIC_NAMES = {"success_DCA4_topn": "DCA4 top-n", "recall_topn2": "top-N+2", "success_DCC12_topn2": "DCC12 top-N+2",
                "recall_ldca_topn2": "DCA4 top-N+2"}
COMP_NAMES = {"published:fpocket+PRANK": "published fpocket+PRANK", "fpocket_4.2.3+PRANK_2.4.2": "re-run fpocket+PRANK",
              "p2rank_2.5.1": "P2Rank 2.5.1", "p2rank_2.5.1_af": "P2Rank 2.5.1 -c alphafold",
              "fpocket_4.2.3+PRANK_2.5.1": "fpocket+PRANK"}


def f2_decision(alias, ep, seed):
    base = f"{alias}:family2{flt(endpoint=ep, seed_index=seed)}"
    sup = SRC.json(crn.ALIASES[alias])
    e = [x for x in sup["family2"] if x["endpoint"] == ep and x["seed_index"] == seed][0]
    LOG.append((CUR_FIG, f"{alias} {ep} s{seed} decision={e['decision']} superior={e['superior']}", float(e["superior"]),
                base + "/superior"))
    if e["decision"] == "pass":
        return "superior" if e["superior"] else "non-inferior"
    return e["decision"]


def fig3():
    global CUR_FIG
    CUR_FIG = "fig3_prereg"
    fig = plt.figure(figsize=(7.5, 6.0))
    L, R = 0.355, 0.785
    gs = fig.add_gridspec(2, 1, height_ratios=[6, 6], left=L, right=R, top=0.9, bottom=0.075, hspace=0.36)
    off = {"v1": -0.17, "v2": 0.17}
    col = {"v2": BLUE, "v1": BLUE_LIGHT}
    mk = {"v2": "o", "v1": "s"}

    def seedtick(ax, x, y):
        ax.plot([x], [y], marker="|", ms=5, mew=0.9, color=INK2, zorder=5)

    def threshold(ax, x, y0, y1, label, ytxt):
        ax.plot([x, x], [y0, y1], color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(x, ytxt, label, fontsize=FS, color=INK2, ha="center", va="bottom")

    # ---- family 1
    ax = fig.add_subplot(gs[0])
    xgrid(ax)
    rows = [(ep, comp) for ep in ("C1", "C1-strict", "C2")
            for comp in ("published:fpocket+PRANK", "fpocket_4.2.3+PRANK_2.4.2")]
    tr_lab = blended_transform_factory(fig.transFigure, ax.transData)
    for i, (ep, comp) in enumerate(rows):
        b = S(f"v2:family1{flt(endpoint=ep, seed_index=0, comparator=comp)}/b", f"{ep} {comp} comparator")
        ax.text(L - 0.012, i, f"{ep} vs {COMP_NAMES[comp]} ({b:.1f})", transform=tr_lab, ha="right",
                va="center", fontsize=FS, color=INK)
        passes = []
        for ver in ("v2", "v1"):
            q = f"{ver}:family1{flt(endpoint=ep, seed_index=0, comparator=comp)}"
            d, lo, hi = (S(f"{q}/{f}", f"{ver} {ep} {comp} s0 {f}") for f in ("diff", "ci_low", "ci_high"))
            dot(ax, d, i + off[ver], lo, hi, color=col[ver], marker=mk[ver], ms=4.8)
            for sd in (1, 2):
                seedtick(ax, S(f"{ver}:family1{flt(endpoint=ep, seed_index=sd, comparator=comp)}/diff",
                               f"{ver} {ep} {comp} s{sd}"), i + off[ver])
            allpass = all(bool(J(ver, "family1", dict(endpoint=ep, seed_index=sd, comparator=comp), "pass"))
                          for sd in (0, 1, 2))
            passes.append("pass" if allpass else "fail")
        ax.text(R + 0.012, i, f"{passes[1]} / {passes[0]}", transform=tr_lab, ha="left", va="center",
                fontsize=FS, color=INK)
    ax.set_ylim(len(rows) - 0.5, -1.0)
    ax.set_yticks([])
    vline(ax, 0, color=INK2, lw=0.7)
    # pre-registered criterion per row group (src/pocketdetect/prereg.py FAMILY1): C1 Δ ≥ +2 and CI > 0;
    # C1-strict CI > 0 only (the solid 0 line); C2 non-inferiority, lower CI bound > −3
    threshold(ax, 2, -0.5, 1.5, "+2", -0.92)
    threshold(ax, -3, 3.5, 5.5, "−3", 3.38)
    for yc, txt in ((0.5, "C1: Δ ≥ +2\nand CI > 0"), (2.5, "C1-strict:\nCI > 0"), (4.5, "C2: lower CI\nbound > −3")):
        ax.text(-14.6, yc, txt, ha="left", va="center", fontsize=FS, color=INK2, style="italic", linespacing=1.1)
    for yb in (1.5, 3.5):
        ax.axhline(yb, color=GRID, lw=0.6, zorder=0)
    ax.set_xlim(-15, 26)
    ax.set_xlabel("Δ BURROW − comparator on LIGYSIS (percentage points)")
    ax.text(R + 0.012, -0.62, "pass v1 / v2\n(3 seeds)", transform=tr_lab, ha="left", va="bottom", fontsize=FS,
            color=INK2)
    bb = ax.get_position()
    fig.text(0.012, bb.y1 + 0.012, "A", fontsize=10, fontweight="bold", ha="left", va="bottom")
    fig.text(0.045, bb.y1 + 0.012, "Family 1: LIGYSIS gatekeeping sequence C1 → C1-strict → C2",
             fontsize=FS + 0.5, fontweight="bold", ha="left", va="bottom")
    fig.legend([leg_handle(BLUE_LIGHT, "s", line=True), leg_handle(BLUE, "o", line=True),
                Line2D([], [], marker="|", ms=5, mew=0.9, color=INK2, ls="none")],
               ["v1, seed 0 (95 % CI)", "v2, seed 0 (95 % CI)", "seeds 1 and 2"], loc="upper left",
               bbox_to_anchor=(0.012, 0.995), ncol=3)

    # ---- family 2
    ax = fig.add_subplot(gs[1])
    xgrid(ax)
    eps = ["O2a", "O2b", "O3", "O4", "O5", "O6"]
    tr_lab = blended_transform_factory(fig.transFigure, ax.transData)
    for i, ep in enumerate(eps):
        q = f"family2{flt(endpoint=ep, seed_index=0)}"
        b = S(f"v2:{q}/b", f"{ep} comparator value")
        e = [x for x in SRC.json(crn.ALIASES["v2"])["family2"] if x["endpoint"] == ep and x["seed_index"] == 0][0]
        n = S(f"v2:{q}/n_rows", f"{ep} n")
        comp = COMP_NAMES[e["best_comparator"]].replace(" 2.5.1", "")
        lab = f"{F2_NAMES[ep]}, {METRIC_NAMES[e['metric']]} (n = {n:,.0f})\nvs {comp} ({b:.1f})"
        ax.text(L - 0.012, i, lab, transform=tr_lab, ha="right", va="center", fontsize=FS, color=INK,
                linespacing=1.15)
        dec = []
        for ver in ("v2", "v1"):
            d, lo, hi = (S(f"{ver}:{q}/{f}", f"{ver} {ep} {f}") for f in ("diff", "ci_low", "ci_high"))
            dot(ax, d, i + off[ver], lo, hi, color=col[ver], marker=mk[ver], ms=4.8)
            for sd in (1, 2):
                seedtick(ax, S(f"{ver}:family2{flt(endpoint=ep, seed_index=sd)}/diff", f"{ver} {ep} s{sd}"),
                         i + off[ver])
            dec.append(f2_decision(ver, ep, 0))
        ax.text(R + 0.012, i, f"v1: {dec[1]}\nv2: {dec[0]}", transform=tr_lab, ha="left", va="center",
                fontsize=FS, color=INK, linespacing=1.15)
    ax.set_ylim(len(eps) - 0.5, -1.0)
    ax.set_yticks([])
    vline(ax, 0, color=INK2, lw=0.7)
    threshold(ax, -2, -0.5, len(eps) - 0.5, "−2 margin", -0.92)
    ax.set_xlim(-13, 17)
    ax.set_xlabel("Δ BURROW − best baseline (percentage points)")
    ax.text(R + 0.012, -0.62, "Holm decision\n(seed 0)", transform=tr_lab, ha="left", va="bottom", fontsize=FS,
            color=INK2)
    bb = ax.get_position()
    fig.text(0.012, bb.y1 + 0.012, "B", fontsize=10, fontweight="bold", ha="left", va="bottom")
    fig.text(0.045, bb.y1 + 0.012, "Family 2: secondary benchmarks (Holm-adjusted, non-inferiority margin −2)",
             fontsize=FS + 0.5, fontweight="bold", ha="left", va="bottom")
    save(fig, "fig3_prereg")
    CAPTIONS["Fig3"] = (
        "**Fig 3. Pre-registered test comparisons (docs/PREREG.md).** Paired differences BURROW − comparator "
        "for the two frozen versions, v1 (squares) and v2 (circles). Seed 0 is shown with 95 % paired "
        "cluster-bootstrap CIs (30 % sequence clusters, 1,000 resamples); ticks mark the point estimates of "
        "seeds 1 and 2. (A) Family 1, a fixed testing sequence on LIGYSIS: C1 = top-N+2 at DCC ≤ 12 Å; "
        "C1-strict = top-N+2 at DCA ≤ 4 Å; C2 = pooled P1K. Comparators are the published fpocket+PRANK "
        "predictions and our re-run of fpocket 4.2.3 + PRANK 2.4.2 (comparator value in brackets). C1 "
        "required Δ ≥ +2 points with a CI above 0, and C1-strict required a CI above 0, for every seed and "
        "both comparators. C2 was a non-inferiority test at −3 (lower CI bound > −3). Each row group's "
        "criterion is printed at its left; dashed lines mark the +2 bar (C1 rows only) and the −3 margin "
        "(C2 rows only). (B) Family 2: Holm-adjusted non-inferiority (margin −2) against the "
        "better of P2Rank 2.5.1 and fpocket 4.2.3 + PRANK on identical inputs; 'superior' means the CI "
        "excludes 0. n = scored sites (ligands for COACH420/HOLO4K). Right: pre-registered decisions. "
        "Source: results/test/v1/summary.json and results/test/v2/summary.json (family1, family2)."
    )


# ==============================================================================================
# Fig 4 — output length trade-off and ligand class
# ==============================================================================================
def fig4():
    global CUR_FIG
    CUR_FIG = "fig4_composition_tradeoff"
    B = "BURROW[burrow_all_default_s0]"
    pub = "published:fpocket+PRANK"
    metrics = [("recall_topn2", "topN2", "Closest-pocket top-N+2"),
               ("recall_anyp_topn2", "topN2_anyp", "Any-pocket top-N+2"),
               ("recall_1to1_topn2", "topN2_1to1", "One-to-one top-N+2"),
               ("recall_max", "max_recall", "Max recall (all pockets)")]
    series = [("BURROW v1", BLUE_LIGHT, "s", ("v1", B)), ("BURROW v2", BLUE, "o", ("v2", B)),
              ("fpocket+PRANK (published)", ORANGE, "o", ("v2", pub)), ("P2Rank (published)", AQUA, "o", None)]
    fig = plt.figure(figsize=(7.5, 5.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.05], left=0.20, right=0.985,
                          top=0.89, bottom=0.085, wspace=0.42, hspace=0.62)
    ax = fig.add_subplot(gs[0, 0])
    xgrid(ax)
    offs = [-0.27, -0.09, 0.09, 0.27]
    for i, (met, rcol, lab) in enumerate(metrics):
        for (name, c, m, src), o in zip(series, offs):
            if src is None:
                v = C(REPRO, {"method": "P2Rank"}, rcol, f"P2Rank published {rcol}", 100)
                lo = hi = None
            else:
                v, lo, hi = sec3(src[0], "ligysis", src[1], met)
            dot(ax, v, i + o, lo, hi, color=c, marker=m, ms=4.6)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m[2] for m in metrics])
    ax.set_ylim(len(metrics) - 0.5, -0.5)
    ax.set_xlim(40, 101)
    ax.set_xlabel("LIGYSIS sites found, %")
    head(ax, "A", "Recall under four matching rules (LIGYSIS, seed 0)", x=0.012)
    hs = [leg_handle(c, m) for _, c, m, _ in series] + [leg_handle(INK2, "o", filled=False)]
    fig.legend(hs, [s[0] for s in series] + ["† post-hoc (open marker)"], loc="upper left",
               bbox_to_anchor=(0.006, 0.998), ncol=5, columnspacing=0.9)

    # B: pockets per chain
    axb = fig.add_subplot(gs[0, 1])
    vals = []
    for name, c, m, src in series:
        v = None if src is None else sec(src[0], "ligysis", src[1], "pockets_per_structure", pct=False)
        vals.append((name, c, v))
    for i, (name, c, v) in enumerate(vals):
        if v is None:
            axb.text(0.6, i, "n/a", va="center", ha="left", fontsize=FS, color=MUTED)
            continue
        axb.barh(i, v, height=0.5, color=c, edgecolor="white", lw=0, zorder=2)
        axb.text(v + 0.6, i, f"{v:.1f}", va="center", ha="left", fontsize=FS, color=INK)
    axb.set_yticks(range(len(vals)))
    axb.set_yticklabels([v[0].replace(" (published)", "\n(published)") for v in vals])
    axb.set_ylim(len(vals) - 0.5, -0.5)
    axb.set_xlim(0, 40)
    axb.spines["left"].set_visible(True)
    axb.spines["left"].set_color(AXIS)
    axb.set_xlabel("Pockets output per chain")
    head(axb, "B", "List length", x=0.60)

    # C: ligand class Δ
    gsc = gs[1, :].subgridspec(1, 2, wspace=0.75)
    # LIGYSIS
    axl = fig.add_subplot(gsc[0])
    xgrid(axl)
    classes = ["ion", "non-ion", "nucleotide-cofactor", "other"]
    ylab = []
    for i, cl in enumerate(classes):
        for ver, c, m, o in (("v1", BLUE_LIGHT, "s", -0.15), ("v2", BLUE, "o", 0.15)):
            if cl == "non-ion":
                q = f"json:results/test/v2/posthoc.json#ligysis_ni{flt(method=f'{ver}:burrow_small_default_s0', comparator='fpocket_4.2.3+PRANK_2.4.2', metric='recall_topn2')}"
                d, lo, hi = (S(f"{q}/{k}", f"LIGYSIS_NI {ver} {k}") for k in ("diff", "ci_low", "ci_high"))
                n = S(f"{q}/n_rows", "LIGYSIS_NI n")
                filled = False
            else:
                q = f"{ver}:descriptive/strata{flt(endpoint='C1', kind='ligand_class', stratum=cl)}"
                d, lo, hi = (S(f"{q}/{k}", f"C1 {cl} {ver} {k}") for k in ("diff", "ci_low", "ci_high"))
                n = S(f"{q}/n_rows", f"C1 {cl} n")
                filled = True
            dot(axl, d, i + o, lo, hi, color=c, marker=m, ms=4.6, filled=filled)
        ylab.append(f"{cl}{' †' if cl == 'non-ion' else ''}\n(n = {n:,.0f})")
    axl.set_yticks(range(len(classes)))
    axl.set_yticklabels(ylab, linespacing=1.1)
    axl.set_ylim(len(classes) - 0.5, -0.5)
    vline(axl, 0, color=INK2, lw=0.7)
    axl.set_xlim(-10, 35)
    axl.set_xlabel("Δ top-N+2 vs fpocket 4.2.3 + PRANK 2.4.2 (pp)")
    head(axl, "C", "LIGYSIS by ligand class", x=0.012)
    # AF set (post-hoc, v2 only)
    axa = fig.add_subplot(gsc[1])
    xgrid(axa)
    ylab = []
    for i, cl in enumerate(classes):
        split = "2-class" if cl in ("ion", "non-ion") else "3-class"
        q = f"json:results/test/v2/posthoc.json#o5_ligand_class{flt(split=split, **{'class': cl}, metric='recall_topn2')}"
        d, lo, hi = (S(f"{q}/{k}", f"O5 {cl} {k}") for k in ("diff", "ci_low", "ci_high"))
        n = S(f"{q}/n_rows", f"O5 {cl} n")
        dot(axa, d, i, lo, hi, color=BLUE, marker="o", ms=4.6, filled=False)
        ylab.append(f"{cl} †\n(n = {n:,.0f})")
    axa.set_yticks(range(len(classes)))
    axa.set_yticklabels(ylab, linespacing=1.1)
    axa.set_ylim(len(classes) - 0.5, -0.5)
    vline(axa, 0, color=INK2, lw=0.7)
    axa.set_xlim(-10, 40)
    axa.set_xlabel("Δ top-N+2 vs P2Rank 2.5.1 -c alphafold (pp)")
    head(axa, "D", "AF set (O5) by ligand class, v2", x=0.515)
    save(fig, "fig4_composition_tradeoff")
    CAPTIONS["Fig4"] = (
        "**Fig 4. What drives the LIGYSIS margin: output length and ligand class.** (A) LIGYSIS recall under four "
        "matching rules (seed 0). Closest-pocket top-N+2 is the LIGYSIS protocol (endpoint C1): a site is "
        "credited only to the pocket nearest its centre. Any-pocket: any of the top N+2 pockets within 12 Å. "
        "One-to-one: Hungarian matching of pockets to sites. Max recall: any output pocket. Bars: 95 % "
        "cluster-bootstrap CIs (not computed for the published P2Rank predictions, which come from "
        "results/validation/ligysis_reproduction/reproduction.csv). (B) Mean pockets output per chain (not "
        "available for the published P2Rank predictions). The v2 output floor shortens the list, which raises "
        "closest-pocket top-N+2 and lowers coverage (any-pocket, one-to-one, max recall), so the C1 gain from "
        "v1 to v2 is an output-length effect rather than better ranking (REPORT.md §4.2). (C, D) Δ top-N+2 by ligand class. (C) LIGYSIS ion, "
        "nucleotide-cofactor and other: pre-registered descriptive strata of C1 (`all` ranker) vs our "
        "fpocket+PRANK re-run. LIGYSIS non-ion (†, post-hoc): the LIGYSIS_NI site set with the `small` ranker. "
        "(D) AF set (†, post-hoc, v2 only) vs P2Rank -c alphafold; non-ion = nucleotide-cofactor + other. Open "
        "markers: post-hoc analyses (not pre-registered). n = scored sites. Bars: 95 % paired cluster-bootstrap "
        "CIs. Sources: results/test/v{1,2}/summary.json (secondary, strata), results/test/v2/posthoc.json."
    )


# ==============================================================================================
# Fig 5 — speed
# ==============================================================================================
def fig5():
    global CUR_FIG
    CUR_FIG = "fig5_speed"
    sec_final = "^## Final like-for-like timing"
    sec_t1 = "^## Track T1"

    def perf(row_regex, col, k, item):
        return S(f"md:{PERF}::{sec_final}::{row_regex}::c{col}.{k}", item)

    B = "BURROW[burrow_all_default_s0]"
    BA = "BURROW[burrow_all_accurate_s0]"
    fsec = "^## `--preset fast`"
    pts = []  # (label, color, marker, ms/structure, topn2 (v,lo,hi), dca4 (v,lo,hi))
    pts.append(("BURROW fast", BLUE, "^", perf(r"^\| BURROW fast \|", 3, 1, "fast ms/structure"),
                (desc_cell(fsec, ["ligysis", "desc.fast_all_s0", "v2:burrow_all_default_s0", "recall_topn2"], 4), None, None),
                (desc_cell(fsec, ["ligysis", "desc.fast_all_s0", "v2:burrow_all_default_s0", "recall_ldca_topn2"], 4), None, None)))
    pts.append(("BURROW default", BLUE, "o", perf(r"^\| BURROW default \|", 3, 1, "default ms/structure"),
                sec3("v2", "ligysis", B, "recall_topn2"), sec3("v2", "ligysis", B, "recall_ldca_topn2")))
    pts.append(("BURROW accurate", BLUE, "s", perf(r"^\| BURROW accurate \|", 3, 1, "accurate ms/structure"),
                sec3("v2", "ligysis", BA, "recall_topn2"), sec3("v2", "ligysis", BA, "recall_ldca_topn2")))
    pts.append(("P2Rank 2.5.1", AQUA, "o", perf(r"^\| P2Rank 2\.5\.1", 3, 1, "P2Rank ms/structure"),
                headline("ligysis", "p2rank_2.5.1", 2), headline("ligysis", "p2rank_2.5.1", 4)))
    pts.append(("fpocket 4.2.3", ORANGE, "o", perf(r"^\| fpocket 4\.2\.3 \(1 process\)", 3, 1, "fpocket ms/structure"),
                headline("ligysis", "fpocket_4.2.3", 2), headline("ligysis", "fpocket_4.2.3", 4)))

    fig = plt.figure(figsize=(7.5, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], left=0.07, right=0.985, top=0.86, bottom=0.17,
                          wspace=0.42)
    nudge = {"BURROW fast": (0, 7, "center", "bottom"), "BURROW default": (0, 7, "center", "bottom"),
             "BURROW accurate": (0, -9, "center", "top"), "P2Rank 2.5.1": (-7, 0, "right", "center"),
             "fpocket 4.2.3": (0, -9, "center", "top")}
    for j, (idx, ylab, pl, ylim) in enumerate(((4, "LIGYSIS top-N+2, %", "A", (35, 80)),
                                                (5, "LIGYSIS DCA4 top-N+2, %", "B", (10, 62)))):
        ax = fig.add_subplot(gs[0, j])
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        for p in pts:
            v, lo, hi = p[idx]
            if lo is not None:
                ax.plot([p[3], p[3]], [lo, hi], color=p[1], lw=1.1, zorder=3, solid_capstyle="butt")
            ax.plot([p[3]], [v], marker=p[2], ms=6 if p[2] != "^" else 6.5, mfc=p[1], mec="white", mew=0.7,
                    ls="none", zorder=4)
            dx, dy, ha, va = nudge[p[0]]
            ax.annotate(p[0].replace("BURROW ", ""), (p[3], v), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, fontsize=FS, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(25, 1200)
        ax.set_xticks([30, 100, 300, 1000])
        ax.set_xticklabels(["30", "100", "300", "1,000"])
        ax.minorticks_off()
        ax.set_ylim(*ylim)
        ax.tick_params(axis="y", labelcolor=INK2)
        ax.set_xlabel("ms per structure (1 process)")
        ax.set_ylabel(ylab)
        panel_label(ax, pl, x=-0.2, y=1.03)
    fig.legend([leg_handle(BLUE), leg_handle(AQUA), leg_handle(ORANGE)], ["BURROW presets", "P2Rank", "fpocket"],
               loc="upper left", bbox_to_anchor=(0.07, 1.0), ncol=3)

    # C: throughput
    ax = fig.add_subplot(gs[0, 2])
    xgrid(ax)
    presets = ["fast", "default", "accurate"]
    hw = [("24 CPU workers", YELLOW, "24 CPU workers, 10,000 structures"),
          ("3 GPUs (8 workers each)", VIOLET, "3 GPUs \\(8 workers each\\), 10,000 structures")]
    h = 0.34
    for i, p in enumerate(presets):
        for k, (lab, c, rx) in enumerate(hw):
            v = S(f"md:{PERF}::{sec_t1}::^\\| {p} \\| CLI, {rx}::c3.0", f"{p} {lab} structures/s")
            y = i + (k - 0.5) * (h + 0.04)
            ax.barh(y, v, height=h, color=c, lw=0, zorder=2)
            ax.text(v + 12, y, f"{v:.0f}", va="center", ha="left", fontsize=FS, color=INK)
    ax.set_yticks(range(3))
    ax.set_yticklabels(presets)
    ax.set_ylim(2.6, -0.6)
    ax.set_xlim(0, 760)
    ax.spines["left"].set_visible(True)
    ax.set_xlabel("Structures per second (10,000-structure run)")
    panel_label(ax, "C", x=-0.2, y=1.03)
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=c, lw=0) for _, c, _ in hw], [h_[0] for h_ in hw],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=1)
    save(fig, "fig5_speed")
    CAPTIONS["Fig5"] = (
        "**Fig 5. Accuracy and speed.** (A, B) LIGYSIS top-N+2 (A) and DCA4 top-N+2 (B) against wall time per "
        "structure, from the final like-for-like timing in docs/PERF.md. All tools ran in one session on the "
        "same 200 LIGYSIS chains as a single process on one core (BLAS/Numba and P2Rank pinned to one thread). "
        "Times are amortised and include process start-up, parsing and writing; P2Rank 2.5.1 runs in batch "
        "mode in one JVM, fpocket 4.2.3 runs one process per structure. Accuracy: BURROW v2 seed 0, `all` "
        "ranker, presets fast / default / accurate (fast is not pre-registered); P2Rank 2.5.1 and fpocket 4.2.3 "
        "re-run on identical inputs (results/baselines/HEADLINES.md). Bars: 95 % cluster-bootstrap CIs where "
        "computed (none for the fast preset). fpocket + PRANK is not shown because it was not timed in the "
        "final session. (C) Throughput of the BURROW CLI over 10,000 structures with 24 CPU workers or 3 GPUs with 8 CPU workers each, including start-up, parsing and "
        "writing (docs/PERF.md, Track T1, 'after' column). Shared node: other users' jobs ran concurrently."
    )


# ==============================================================================================
# Fig 6 — AF models and dimers
# ==============================================================================================
def fig6():
    global CUR_FIG
    CUR_FIG = "fig6_af_dimers"
    nsec = "^## Naive \\(no --af-mode\\)"
    ref = "v2:burrow_all_default_s0_af"
    rows = []
    for lab, meth in (("BURROW naive input", "desc.v2_naive_all_s0"),
                      ("BURROW naive + pLDDT < 70\npocket filter", "desc.v2_naive_plddt70_all_s0")):
        t = desc_cell(nsec, ["af_set", meth, ref, "recall_topn2"], 4, item=f"{meth} top-N+2")
        hr = desc_cell(nsec, ["af_set", meth, ref, "hallucination_rate_topn2"], 4, item=f"{meth} halluc top-N+2")
        rows.append((lab, BLUE, (t, None, None), hr))
    AF = "BURROW[burrow_all_default_s0_af]"
    rows.append(("BURROW --af-mode (v2)", BLUE, sec3("v2", "af_set", AF, "recall_topn2"),
                 sec("v2", "af_set", AF, "hallucination_rate_topn2")))
    rows.append(("P2Rank 2.5.1 -c alphafold", AQUA, sec3("v2", "af_set", "p2rank_2.5.1_af", "recall_topn2"),
                 sec("v2", "af_set", "p2rank_2.5.1_af", "hallucination_rate_topn2")))

    fig = plt.figure(figsize=(7.5, 4.6))
    gs = fig.add_gridspec(2, 2, left=0.235, right=0.985, top=0.92, bottom=0.10, wspace=0.12, hspace=0.62,
                          height_ratios=[1, 0.85])
    ax = fig.add_subplot(gs[0, 0])
    xgrid(ax)
    for i, (lab, c, (v, lo, hi), _) in enumerate(rows):
        dot(ax, v, i, lo, hi, color=c, ms=5)
        ax.text(v, i - 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=FS, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], linespacing=1.1)
    ax.set_ylim(len(rows) - 0.5, -0.75)
    ax.set_xlim(45, 70)
    ax.set_xlabel("AF set top-N+2, %")
    head(ax, "A", "AlphaFold models (AF set, v2 seed 0)", x=0.012)
    ax2 = fig.add_subplot(gs[0, 1])
    xgrid(ax2)
    for i, (lab, c, _, hr) in enumerate(rows):
        ax2.barh(i, hr, height=0.45, color=c, lw=0, zorder=2)
        ax2.text(hr + 0.05, i, f"{hr:.2f}", va="center", ha="left", fontsize=FS, color=INK)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([])
    ax2.set_ylim(len(rows) - 0.5, -0.75)
    ax2.set_xlim(0, 3.4)
    ax2.spines["left"].set_visible(True)
    ax2.set_xlabel("Hallucinated top-N+2 pockets, %")

    # B: dimers — assembly vs isolated-chain input
    csec = "^## Dimer benchmark, isolated-chain inputs"
    meths = [("BURROW v2", BLUE, "BURROW[burrow_all_default_s0]", "desc.v2chains_all_s0", None),
             ("P2Rank 2.5.1", AQUA, "p2rank_2.5.1", "desc.v2chains_all_s0", "p2rank_2.5.1 (chain mode)"),
             ("fpocket 4.2.3 + PRANK 2.5.1", ORANGE, "fpocket_4.2.3+PRANK_2.5.1", "desc.v2chains_all_s0",
              "fpocket_4.2.3+PRANK_2.5.1 (chain mode)")]
    for j, (scope, title) in enumerate((("interface", "Interface sites"), ("all", "All sites"))):
        ax = fig.add_subplot(gs[1, j])
        xgrid(ax)
        for i, (lab, c, asm, chm, chref) in enumerate(meths):
            va, lo, hi = sec3("v2", "dimer", asm, f"recall_DCC12_topn2@{scope}")
            metric = f"success_DCC12_topn2@{scope}"
            if chref is None:  # BURROW chain-mode value: method column of the vs-P2Rank row
                vc = desc_cell(csec, ["desc.v2chains_all_s0", "p2rank_2.5.1 (chain mode)", metric], 3,
                               item=f"BURROW chain {scope}")
            else:
                vc = desc_cell(csec, ["desc.v2chains_all_s0", chref, metric], 4, item=f"{lab} chain {scope}")
            dot(ax, va, i - 0.16, lo, hi, color=c, ms=5)
            dot(ax, vc, i + 0.16, color=c, ms=5, filled=False)
        ax.set_yticks(range(len(meths)))
        ax.set_yticklabels([m[0] for m in meths] if j == 0 else [])
        ax.set_ylim(len(meths) - 0.5, -0.6)
        ax.set_xlim(50, 95)
        ax.set_xlabel(f"{title}: DCC12 top-N+2, %")
        if j == 0:
            head(ax, "B", "Dimer benchmark: full-assembly vs isolated-chain input (v2 seed 0)", x=0.012)
            bb = ax.get_position()
            fig.legend([leg_handle(INK2, "o", line=True), leg_handle(INK2, "o", filled=False)],
                       ["full-assembly input (95 % CI)", "isolated-chain input"], loc="lower right",
                       bbox_to_anchor=(0.985, bb.y1 + 0.004), ncol=2)
    save(fig, "fig6_af_dimers")
    n_af = S(f"v2:family2{flt(endpoint='O5', seed_index=0)}/n_rows", "O5 n sites")
    n_af_units = S(f"v2:family2{flt(endpoint='O5', seed_index=0)}/n_units", "O5 units")
    n_af_str = S("v2:denominators{benchmark=af_set;method=BURROW[burrow_all_default_s0_af]}/n_structures_or_units",
                 "AF set structures")
    dc = SRC.json("results/test/v2/posthoc.json")["dimer_counts"]
    for k in ("assemblies_scored", "sites", "interface_sites"):
        S(f"json:results/test/v2/posthoc.json#dimer_counts/{k}", f"dimer {k}")
    srow = ["desc.v2chains_all_s0 (site-level, any chain)", "v2:burrow_all_default_s0 (assembly mode)"]
    sl = [desc_cell(csec, srow + ["success_DCC12_topn2@interface"], 5, k, item=f"BURROW site-level chain vs asm DCC12 {k}")
          for k in range(3)]
    sd = [desc_cell(csec, srow + ["success_DCA4_topn2@interface"], 5, k, item=f"BURROW site-level chain vs asm DCA4 {k}")
          for k in range(3)]
    n_ci = desc_cell(csec, ["desc.v2chains_all_s0", "p2rank_2.5.1 (chain mode)", "success_DCC12_topn2@interface"], 6,
                     item="chain-mode interface rows")
    n_ca = desc_cell(csec, ["desc.v2chains_all_s0", "p2rank_2.5.1 (chain mode)", "success_DCC12_topn2@all"], 6,
                     item="chain-mode all rows")
    CAPTIONS["Fig6"] = (
        f"**Fig 6. AlphaFold models and protein dimers.** (A) AF set ({n_af_str:,.0f} AlphaFold models, "
        f"{n_af:,.0f} sites, {n_af_units:,.0f} cluster units). Left: top-N+2 (DCC ≤ 12 Å). "
        "Right: share of top-N+2 pockets that are hallucinated, i.e. whose mean pLDDT is < 50 and that match no "
        "site within 12 Å (pooled; no CI). BURROW rows: v2 `all` default model, seed 0, with naive input "
        "(no AF handling), with naive input plus removal of pockets with mean pLDDT < 70, and in `--af-mode` "
        "(the pre-registered O5 variant). The naive and filter rows are descriptive and have CIs only for their "
        "paired differences vs AF mode (results/test/v2/DESCRIPTIVE.md). Bars on points: 95 % cluster-bootstrap "
        f"CIs. (B) Dimer benchmark ({dc['assemblies_scored']} assemblies, {dc['sites']} unique sites, of which "
        f"{dc['interface_sites']} lie at an interface). Filled: full-assembly input (site level, with 95 % CI). "
        "Open: every chain predicted alone. In chain mode the unit is a chain copy and a site is "
        f"scored on each chain unit that carries it ({n_ci:,.0f} interface and {n_ca:,.0f} total rows), so the "
        "two input modes have different denominators. Counting a site as found when any isolated chain finds it, "
        f"BURROW's interface DCC12 top-N+2 changes by {signed(sl[0])} [{signed(sl[1])}, {signed(sl[2])}] pp vs "
        f"assembly input (DCA4: {signed(sd[0])} [{signed(sd[1])}, {signed(sd[2])}] pp). Isolated-chain rows are descriptive. Sources: results/test/v2/summary.json "
        "(secondary: af_set, dimer), results/test/v2/DESCRIPTIVE.md."
    )


# ==============================================================================================
# S1 — fold-novelty strata
# ==============================================================================================
def figS1():
    global CUR_FIG
    CUR_FIG = "figS1_strata"
    panels = [("C1", "LIGYSIS (C1) vs fpocket+PRANK re-run"), ("O5", "AF set (O5) vs P2Rank -c alphafold"),
              ("O6i", "Dimer interface (O6) vs P2Rank †"), ("O2b", "HOLO4K (O2b) vs P2Rank"),
              ("O3", "PocketWatch (O3) vs P2Rank -c alphafold"), ("O4", "CryptoBench (O4) vs P2Rank")]
    strata = ["<0.5", "0.5-0.7", ">0.7"]
    slab = {"<0.5": "< 0.5", "0.5-0.7": "0.5–0.7", ">0.7": "> 0.7"}
    fig = plt.figure(figsize=(7.5, 4.6))
    gs = fig.add_gridspec(2, 3, left=0.115, right=0.985, top=0.855, bottom=0.11, wspace=0.75, hspace=0.75)
    for k, (ep, title) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        xgrid(ax)
        ylab = []
        lo_all, hi_all = [], []
        for i, st in enumerate(strata):
            if ep == "O6i":
                q = f"json:results/test/v2/posthoc.json#o6_interface_fold_novelty{flt(stratum=st)}"
                d, lo, hi = (S(f"{q}/{f}", f"O6 interface {st} {f}") for f in ("diff", "ci_low", "ci_high"))
                n = S(f"{q}/n_rows", f"O6 interface {st} n")
                dot(ax, d, i, lo, hi, color=BLUE, marker="o", ms=4.6, filled=False)
                lo_all += [lo]
                hi_all += [hi]
            else:
                for ver, c, m, o in (("v1", BLUE_LIGHT, "s", -0.15), ("v2", BLUE, "o", 0.15)):
                    q = f"{ver}:descriptive/strata{flt(endpoint=ep, kind='fold_novelty_maxTM', stratum=st)}"
                    d, lo, hi = (S(f"{q}/{f}", f"{ep} {st} {ver} {f}") for f in ("diff", "ci_low", "ci_high"))
                    n = S(f"{q}/n_rows", f"{ep} {st} n")
                    dot(ax, d, i + o, lo, hi, color=c, marker=m, ms=4.6)
                    lo_all += [lo]
                    hi_all += [hi]
            ylab.append(f"{slab[st]}{' *' if n < 100 else ''}\nn = {n:,.0f}")
        ax.set_yticks(range(3))
        ax.set_yticklabels(ylab, linespacing=1.1)
        ax.set_ylim(2.5, -0.5)
        vline(ax, 0, color=INK2, lw=0.7)
        span = max(abs(min(lo_all)), abs(max(hi_all)))
        ax.set_xlim(min(-5, min(lo_all) - 1.5), max(5, max(hi_all) + 1.5))
        ax.set_xlabel("Δ vs comparator (pp)")
        head(ax, "ABCDEF"[k], title, x=ax.get_position().x0 - 0.105, tfs=FS)
        del span
    fig.legend([leg_handle(BLUE_LIGHT, "s", line=True), leg_handle(BLUE, "o", line=True),
                leg_handle(BLUE, "o", filled=False, line=True)],
               ["v1", "v2", "v2, post-hoc (interface sites only)"], loc="upper left", bbox_to_anchor=(0.005, 1.0),
               ncol=3)
    fig.text(0.012, 0.955, "Rows: maximum Foldseek TM-score of the test structure to any training structure",
             fontsize=FS, color=INK2, ha="left", va="top")
    save(fig, "figS1_strata")
    n_if = f'{S("json:results/test/v2/posthoc.json#dimer_counts/interface_sites", "dimer interface sites"):,.0f}'
    CAPTIONS["S1_Fig"] = (
        "**S1 Fig. Differences by fold novelty.** Δ (BURROW − comparator, percentage points; seed 0, default "
        "preset) on each benchmark's endpoint metric, stratified by the maximum Foldseek TM-score between the "
        "test structure and any training structure (< 0.5 = novel fold). Comparators as in the pre-registered "
        "endpoints: (A) LIGYSIS top-N+2 vs fpocket 4.2.3 + PRANK 2.4.2; (B) AF set top-N+2 vs P2Rank 2.5.1 "
        "-c alphafold; (C) dimer interface DCC12 top-N+2 vs P2Rank 2.5.1, recomputed post-hoc (†, open "
        f"markers) on the {n_if} interface sites only, because the O6 fold strata in SUMMARY.md were computed over "
        "all dimer sites (stratum-filter bug, descriptive only); v2 only; (D) HOLO4K DCA4 top-n; (E) PocketWatch "
        "top-N+2; (F) CryptoBench (apo) top-N+2. n = sites (ligands for HOLO4K) per stratum; * = fewer than "
        "100. Bars: 95 % paired cluster-bootstrap CIs. Strata are descriptive. Sources: "
        "results/test/v{1,2}/summary.json (descriptive/strata), results/test/v2/posthoc.json."
    )


# ==============================================================================================
# S2 — ablations
# ==============================================================================================
def figS2():
    global CUR_FIG
    CUR_FIG = "figS2_ablations"
    dev_rows = [("fold_disjoint", "A1 fold-disjoint training"), ("chen11", "A2 CHEN11-only training"),
                ("drop:rays", "A4 − ray buriedness"), ("drop:bbbur", "A4 − backbone-only buriedness"),
                ("drop:hitchem", "A4 − first-hit chemistry"), ("drop:metal", "A4 − metal features"),
                ("drop:rescomp", "A4 − residue composition"), ("drop:atomclass", "A4 − atom-class densities"),
                ("shuffled", "A3 shuffled labels")]
    fig = plt.figure(figsize=(7.5, 8.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.1], left=0.0, right=1.0, top=0.925, bottom=0.05,
                             hspace=0.2)
    top = outer[0].subgridspec(1, 2, wspace=0.06)

    def offscale(ax, xl, y, v, lo, hi, color, prefix=""):
        """Value left of the axis range: arrow at the left edge plus the printed value and CI."""
        w = xl[1] - xl[0]
        ax.annotate("", xy=(xl[0] + 0.005 * w, y), xytext=(xl[0] + 0.1 * w, y),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15", color=color, lw=1.0))
        ax.text(xl[0] + 0.115 * w, y, prefix + f"{signed(v)} [{lo:.1f}, {hi:.1f}]".replace("-", "−"), va="center",
                ha="left", fontsize=FS, color=INK2, zorder=6,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))

    XD = (-17, 6)
    dev_axes = []
    for j_, (table, short) in enumerate((("BioLiP dev", "BioLiP dev"), ("apo-dev cryptic", "apo-dev, cryptic sites"))):
        sub = top[0, j_].subgridspec(1, 1)
        ax = fig.add_subplot(sub[0])
        dev_axes.append(ax)
        pos = ax.get_position()
        lab_w = 0.235 if j_ == 0 else 0.0
        ax.set_position([pos.x0 + (0.245 if j_ == 0 else 0.02), pos.y0, pos.width - (0.255 if j_ == 0 else 0.03),
                         pos.height])
        xgrid(ax)
        ypos = [i + (0.35 if abl == "shuffled" else 0.0) for i, (abl, _) in enumerate(dev_rows)]
        for i, (abl, lab) in enumerate(dev_rows):
            i = ypos[i]
            sep = 0.36 if abl == "shuffled" else 0.17  # off-scale rows carry printed labels: more room
            for var, filled, o in (("all", True, -sep), ("small", False, sep)):
                key = {"ablation": abl, "variant": var, "table": table, "metric": "recall_topn2"}
                d = C(ABL, key, "diff", f"{table} {abl} {var} diff")
                lo = C(ABL, key, "lo", f"{table} {abl} {var} lo")
                hi = C(ABL, key, "hi", f"{table} {abl} {var} hi")
                if lo < XD[0]:
                    offscale(ax, XD, i + o, d, lo, hi, INK2, prefix=f"{var}: ")
                else:
                    dot(ax, d, i + o, lo, hi, color=INK2, ms=4.6, filled=filled)
        n = C(ABL, {"ablation": "fold_disjoint", "variant": "all", "table": table, "metric": "recall_topn2"},
              "n_sites", f"{table} n_sites")
        ax.set_yticks(ypos)
        ax.set_yticklabels([r[1] for r in dev_rows] if j_ == 0 else [])
        ax.set_ylim(ypos[-1] + 0.65, -0.6)
        vline(ax, 0, color=INK2, lw=0.7)
        ax.set_xlim(*XD)
        ax.set_xlabel("Δ DCC12 top-N+2 vs v1 (pp)")
        head(ax, "AB"[j_], f"{short} ({n:,.0f} sites)", x=0.012 if j_ == 0 else ax.get_position().x0)
        del lab_w
    fig.text(0.012, 0.995, "Development sets (v1 recipe, seed 0)", fontsize=FS + 1, fontweight="bold", ha="left",
             va="top")
    fig.legend([leg_handle(INK2, "o", line=True), leg_handle(INK2, "o", filled=False, line=True)],
               ["all ranker", "small ranker"], loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=2)

    # ---- test: two side-by-side panels, each with two ablation groups and its own x range
    bench = [("ligysis", "recall_topn2", "LIGYSIS top-N+2"), ("ligysis", "recall_ldca_topn2", "LIGYSIS DCA4"),
             ("ligysis_ni", "recall_topn2", "LIGYSIS_NI top-N+2"), ("coach420_mlig", "success_DCA4_topn", "COACH420 DCA4 top-n"),
             ("holo4k_mlig", "success_DCA4_topn", "HOLO4K DCA4 top-n"), ("cryptobench", "recall_topn2", "CryptoBench top-N+2"),
             ("pocketwatch", "recall_topn2", "PocketWatch top-N+2"), ("af_set", "recall_topn2", "AF set top-N+2"),
             ("af_set", "recall_ldca_topn2", "AF set DCA4"),
             ("dimer", "success_DCC12_topn2@interface", "Dimer interface")]
    panels = [("C", "Test sets: fold-disjoint training (A1)", (-8, 6),
               [("A1 vs v1", "abl_fold_disjoint", "v1"), ("A1 + v2 output floor vs v2", "abl_fold_disjoint_floor", "v2")]),
              ("D", "Test sets: CHEN11-only (A2), shuffled labels (A3)", (-55, 8),
               [("A2 CHEN11-only vs v1", "abl_chen11", "v1"), ("A3 shuffled labels vs v1", "abl_shuffled", "v1")])]
    ep = SRC.json(crn.ALIASES["v2"])["descriptive"]["extra_paired"]
    bot = outer[1].subgridspec(1, 2, wspace=0.0)
    for k, (letter, title, xl, groups) in enumerate(panels):
        ax = fig.add_subplot(bot[0, k])
        pos = ax.get_position()
        ax.set_position([pos.x0 + 0.215, pos.y0, pos.width - 0.235, pos.height])
        xgrid(ax)
        y = 0
        yt, yl = [], []
        tr = blended_transform_factory(fig.transFigure, ax.transData)
        for gname, prefix, refv in groups:
            ax.text(pos.x0 + 0.012, y, gname, transform=tr, ha="left", va="center", fontsize=FS,
                    fontweight="bold", color=INK)
            y += 1
            for b, met, lab in bench:
                var = "small" if b in ("ligysis_ni", "coach420_mlig", "holo4k_mlig", "cryptobench", "pocketwatch") else "all"
                suf = "_af" if b in ("af_set", "pocketwatch") else ""
                meth = f"desc.{prefix}_{var}_s0{suf}"
                refm = f"{refv}:burrow_{var}_default_s0{suf}"
                if not [e for e in ep if e["benchmark"] == b and e["method"] == meth and e["reference"] == refm
                        and e["metric"] == met]:
                    continue
                q = f"v2:descriptive/extra_paired{flt(benchmark=b, method=meth, reference=refm, metric=met)}"
                d, lo, hi = (S(f"{q}/{f}", f"{meth} {b} {met} {f}") for f in ("diff", "ci_low", "ci_high"))
                if lo < xl[0]:
                    offscale(ax, xl, y, d, lo, hi, BLUE)
                else:
                    dot(ax, d, y, lo, hi, color=BLUE, ms=4.6)
                yt.append(y)
                yl.append(lab)
                y += 1
            y += 0.5
        ax.set_yticks(yt)
        ax.set_yticklabels(yl)
        ax.set_ylim(y - 0.9, -0.7)
        vline(ax, 0, color=INK2, lw=0.7)
        ax.set_xlim(*xl)
        ax.set_xlabel("Δ ablation − reference (pp)")
        head(ax, letter, title, x=pos.x0 + 0.012)
    save(fig, "figS2_ablations")
    CAPTIONS["S2_Fig"] = (
        "**S2 Fig. Ablations and controls.** Paired Δ (ablation − reference, percentage points) with 95 % paired "
        "cluster-bootstrap CIs (1,000 resamples; BioLiP dev and test: 30 % sequence clusters; apo-dev: UniProt "
        "units). In (A, B), values beyond the axis are drawn as arrows with the value and CI printed. (A, B) Development "
        "sets, models trained with the frozen v1 recipe (seed 0) and paired against the v1 seed-0 model of the "
        "same ranker (filled: `all`; open: `small`); metric DCC12 top-N+2. A1: training entries with Foldseek "
        "TM ≥ 0.5 to any test chain removed. A2: trained on the CHEN11 structures only. A3: point labels and "
        "pocket targets permuted within each protein (negative control). A4: one feature group removed from "
        "the point model and the matching pocket descriptors from the ranker. (C, D) Test sets (descriptive; not "
        "part of the pre-registered families), on each benchmark's endpoint metric (DCA4 rows: DCA ≤ 4 Å "
        "top-N+2; dimer: interface DCC12 top-N+2). (C) A1 vs v1, and A1 retrained with the v2 output floor vs "
        "v2. (D) A2 and A3 vs v1, on a wider axis. Sources: results/ablations/ablations_long.csv "
        "(ABLATIONS.md), results/test/v2/summary.json (descriptive/extra_paired; DESCRIPTIVE.md)."
    )


# ==============================================================================================
def write_outputs():
    with open(HERE / "figure_data.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["figure", "item", "value", "selector"])
        for row in LOG:
            w.writerow([row[0], row[1], f"{row[2]:.6g}", row[3]])
    order = ["Fig1", "Fig2", "Fig3", "Fig4", "Fig5", "Fig6", "S1_Fig", "S2_Fig"]
    files = {"Fig1": "fig1_pipeline", "Fig2": "fig2_ligysis", "Fig3": "fig3_prereg",
             "Fig4": "fig4_composition_tradeoff", "Fig5": "fig5_speed", "Fig6": "fig6_af_dimers",
             "S1_Fig": "figS1_strata", "S2_Fig": "figS2_ablations"}
    lines = ["# Figure captions (draft)", "",
             "Generated by `manuscript/figures/make_figures.py` (do not edit by hand; edit the script). Every "
             "number in the figures is read from a committed results file; `figure_data.csv` lists each plotted "
             "value with its source selector (the selector language of `scripts/check_report_numbers.py`). "
             "pp = percentage points. CIs are 95 % paired cluster-bootstrap intervals (30 % MMseqs2 sequence "
             "clusters, 1,000 resamples) unless stated otherwise.", ""]
    for k in order:
        if k in CAPTIONS:
            lines += [f"## {k} — `{files[k]}.pdf` / `.png`", "", CAPTIONS[k], ""]
    (HERE / "CAPTIONS.md").write_text("\n".join(lines))
    print(f"wrote CAPTIONS.md, figure_data.csv ({len(LOG)} values)")


def main(argv=None):
    global SRC
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worktree", action="store_true", help="read result files from the working tree, not git HEAD")
    ap.add_argument("--only", nargs="*", help="subset of figures, e.g. fig2 figS1")
    a = ap.parse_args(argv)
    SRC = crn.Source(None if a.worktree else "HEAD")
    print(f"font: {FONT}; sources: {'working tree' if a.worktree else 'git HEAD'}")
    funcs = [fig1, fig2, fig3, fig4, fig5, fig6, figS1, figS2]
    for f in funcs:
        if a.only and f.__name__ not in a.only:
            continue
        f()
    if a.only:
        print("--only: CAPTIONS.md / figure_data.csv left unchanged (run without --only to rewrite them)")
    else:
        write_outputs()


if __name__ == "__main__":
    main()
