"""Shared style, naming and export for every paper figure.

One place for the three things that must not drift between figures: what a
method is called, what its colour and marker mean, and how a figure is
exported. Importing this module is the only way the figure scripts get any of
them.

Sizing follows IEEE two-column practice: 88.9 mm for a single column and
181 mm across both. No venue is fixed, so those widths, the font floor and the
accepted formats are PENDING VERIFICATION against the target's live guidance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # relation_aware_pipeline/
SKILL = Path.home() / ".claude" / "skills" / "scientific-visualization" / "scripts"
sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(ROOT))
from figure_export import export_figure as _export          # noqa: E402
from style_presets import style_context                     # noqa: E402

MM = 1 / 25.4
W1, W2 = 88.9, 181.0                            # IEEE single / double column

# ---------------------------------------------------------------- naming
# The left column is every spelling that has appeared in an earlier figure or
# in the evaluation code; the right column is the only spelling a paper figure
# may use. README.md repeats this map for the reader.
NAMES = {
    "B0-adapt":                 "Flat-Adaptive",
    "B0-k":                     "Flat-Fixed-k",
    "B1-D3L":                   "D3L-Inspired",
    "B1e-D3L-official":         "D3L (authors' code)",
    "B1b-AutoDDG-k":            "AutoDDG-Fixed-k",
    "B1c-AutoDDG-adapt":        "AutoDDG-Adaptive",
    "B1d-AutoDDG-D3L":          "AutoDDG + D3L",
    "B2-family":                "Group-Expansion",
    "B2b-family+join":          "Group + Join Candidates",
    "B3-structure":             "Structure-1Seed",
    "B3-seed3":                 "Structure-3Seeds",
    "B3-seed5":                 "Structure-5Seeds",
    "B3x-crossfam-only":        "Cross-Group-Relations-Only",
    "B3-budget20":              "Structure-3Seeds (budget 20)",
    "B3-noquestion":            "Structure-3Seeds (no question gate)",
    "B4-relationset":           "SANTOS Relationship Set",
    "B5-family+incl":           "Group + Containment",
    "B6-AutoDDG+structure":     "AutoDDG + Structure",
    "B6b-AutoDDG+family":       "AutoDDG + Group",
    "B7-Aurum":                 "Aurum Neighborhood",
    "B8-TUS":                   "Table-Union Search",
    "B9-SchemaJaccard":         "Schema-Name Jaccard",
    "B10-ValueOverlap":         "Value Overlap",
    "B11-JOSIE":                "JOSIE Containment",
    "B12-PRF":                  "Pseudo-Relevance Feedback",
    "B13-random":               "Random-k",
    "B14-Starmie-approx":       "Starmie-Inspired",
    "B15-LakeClustering":       "Lake Clustering",
    "B16-Navigation":           "Lake Navigation",
    "oracle-k":                 "Oracle-k",
    "B0-sized-as-B2-family":    "Flat @ Group-Expansion size",
    "B0-sized-as-B3-structure": "Flat @ Structure-1Seed size",
    "B0-sized-as-B3-seed3":     "Flat @ Structure-3Seeds size",
}

OURS = {"B2-family", "B2b-family+join", "B3-structure", "B3-seed3",
        "B3-seed5", "B3x-crossfam-only", "B3-budget20", "B3-noquestion",
        "B5-family+incl", "B6-AutoDDG+structure", "B6b-AutoDDG+family"}
REFERENCE = {"B0-adapt", "B0-k", "B13-random", "oracle-k",
             "B0-sized-as-B2-family", "B0-sized-as-B3-structure",
             "B0-sized-as-B3-seed3"}

# ---------------------------------------------------------------- colour
INK, MUTED, HAIR = "#16191C", "#6B7075", "#C9CDD1"
OURS_C  = "#D55E00"        # this work
PRIOR_C = "#0072B2"        # prior method
REF_C   = "#6B7075"        # reference, control, ceiling — not a system
NEG_C   = "#7A3E10"        # a contrast that goes the wrong way
FILL_BG = "#FAFAFA"

# marker semantics, stated once and repeated in every legend
MARKER = {"ours": "s", "prior": "o", "ref": "^"}


def category(arm: str) -> str:
    return "ours" if arm in OURS else "ref" if arm in REFERENCE else "prior"


def colour(arm: str) -> str:
    return {"ours": OURS_C, "prior": PRIOR_C, "ref": REF_C}[category(arm)]


def label(arm: str) -> str:
    if arm not in NAMES:
        raise KeyError(f"{arm} has no paper-facing name; add it to NAMES")
    return NAMES[arm]


# ---------------------------------------------------------------- type scale
FS = {"title": 7.4, "sub": 5.8, "axis": 6.6, "tick": 6.2, "legend": 6.0,
      "annot": 5.8}


def tidy(ax, xgrid=True, ygrid=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(HAIR)
    ax.tick_params(labelsize=FS["tick"], color=HAIR)
    if xgrid:
        ax.grid(axis="x", color=HAIR, lw=.4, alpha=.6)
    if ygrid:
        ax.grid(axis="y", color=HAIR, lw=.4, alpha=.6)
    ax.set_axisbelow(True)


def panel_title(ax, letter, title, sub=""):
    ax.set_title(f"{letter}   {title}", fontsize=FS["title"], loc="left",
                 color=INK, pad=13 if sub else 6)
    if sub:
        ax.text(0.0, 1.005, sub, transform=ax.transAxes, fontsize=FS["sub"],
                color=MUTED, va="bottom", ha="left")


def legend_patches(ax, x, ys, entries):
    """entries: [(category, text)] — colour and marker together, never colour
    alone, because the palette audit puts two of these hues within 1 unit of
    each other in grayscale."""
    for y, (cat, text) in zip(ys, entries):
        c = {"ours": OURS_C, "prior": PRIOR_C, "ref": REF_C}[cat]
        ax.plot(x, y, marker=MARKER[cat], ms=4.4, color=c, markerfacecolor=c,
                clip_on=False)
        ax.text(x, y, "   " + text, fontsize=FS["legend"], color=c,
                va="center", ha="left")


def export(fig, stem: str, provenance: dict):
    prov = {"style_module": "paper_artifacts/figures/fig_style.py",
            "naming": "paper-facing names from fig_style.NAMES",
            "publisher_choices": "PENDING VERIFICATION — IEEE two-column "
                                 "widths assumed, no venue fixed",
            "colour_is_redundant": "category is carried by marker shape as "
                                   "well as hue; see README",
            **provenance}
    return _export(fig, str(HERE.parent / stem), formats=["pdf", "png"], dpi=600,
                   bbox_inches=None, facecolor="white", font_mode="truetype",
                   overwrite=True, provenance=prov, write_manifest=True)
