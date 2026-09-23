"""The retrieval summary figure, in two versions.

    fig_retrieval_main       ten arms, for the paper body
    fig_retrieval_appendix   every arm that was scored, for coverage

Both share panels B and C and the same visual grammar, so the appendix
version is the main version with panel A extended — not a different figure.

Panel A orders arms as listed, not by effect size, so the two versions put
the same method in a comparable place. The reference is Flat-Adaptive
throughout: choosing the answer size requires no structure, so measuring
against a fixed cut would credit the method with something it did not do.
Flat-Fixed-k appears as an arm instead.
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE.parent          # figures/
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import fig_style as S                                        # noqa: E402
sys.path.insert(0, str(S.ROOT))
from ramp.paths import CM                                    # noqa: E402
sys.path.insert(0, str(CM / "round2"))
from r5_eval import sign_test                                # noqa: E402
from ramp.stats import clustered_bootstrap                   # noqa: E402

REF = "B0-adapt"
LAKES = (("out_santos", "SANTOS"), ("out_astro2", "KramaBench"))

# top of the panel first
MAIN = ["B3-seed5", "B3-seed3", "B3-structure", "B2-family",
        "B6-AutoDDG+structure", "B1c-AutoDDG-adapt", "B1-D3L", "B0-k",
        "B3x-crossfam-only", "oracle-k"]

APPENDIX = ["B3-seed5", "B3-seed3", "oracle-k", "B2-family", "B5-family+incl",
            "B3-structure", "B2b-family+join", "B6b-AutoDDG+family",
            "B6-AutoDDG+structure", "B1c-AutoDDG-adapt", "B1d-AutoDDG-D3L",
            "B4-relationset", "B1b-AutoDDG-k", "B8-TUS", "B9-SchemaJaccard",
            "B1-D3L", "B12-PRF", "B0-k", "B7-Aurum", "B10-ValueOverlap",
            "B1e-D3L-official", "B11-JOSIE", "B3x-crossfam-only", "B13-random"]

PANEL_B = [
    ("B2-family",         "B0-sized-as-B2-family",    "Group-Expansion vs flat @ same size", True),
    ("B3-structure",      "B0-sized-as-B3-structure", "Structure-1Seed vs flat @ same size", True),
    ("B3-seed3",          "B0-sized-as-B3-seed3",     "Structure-3Seeds vs flat @ same size", True),
    ("B3x-crossfam-only", "B2-family",                "drop Group-Expansion", False),
    ("B2b-family+join",   "B2-family",                "add Join Candidates", False),
    ("B3-structure",      "B2b-family+join",          "add Cross-Group Hop", False),
    ("B3-seed3",          "B3-structure",             "seeds: 3 vs 1", False),
    ("B3-seed5",          "B3-seed3",                 "seeds: 5 vs 3", False),
    ("B3-noquestion",     "B3-structure",             "question gate: off vs on", False),
    ("B3-budget20",       "B3-seed3",                 "budget: 20 vs 40", False),
]


def load():
    by = defaultdict(dict)
    for d, lake in LAKES:
        for r in csv.DictReader((S.ROOT / d / "discovery_per_query.csv").open()):
            if r["retriever"] == "dense_chunked":
                r["lake"] = lake
                by[r["arm"]][f'{lake}|{r["query"]}'] = r
    return by


def contrast(by, arm, ref):
    ks = sorted(set(by[arm]) & set(by[ref]))
    a = [float(by[arm][k]["f1"]) for k in ks]
    b = [float(by[ref][k]["f1"]) for k in ks]
    lo, hi = clustered_bootstrap(
        a, b, [f'{by[arm][k]["lake"]}|{by[arm][k]["parent"]}' for k in ks])
    t = sign_test(a, b)
    return {"n": len(ks), "diff": st.mean(a) - st.mean(b), "lo": lo, "hi": hi,
            "p": t["p_two_sided"],
            "npred": st.mean(float(by[arm][k]["n_pred"]) for k in ks),
            "sig": bool(lo * hi > 0 and t["p_two_sided"] < 0.05)}


def dot(ax, i, r, c, mk, ms=4.4, lw=1.0):
    ax.plot([r["lo"], r["hi"]], [i, i], color=c, lw=lw, zorder=2,
            solid_capstyle="round")
    ax.plot(r["diff"], i, marker=mk, ms=ms, color=c, zorder=3,
            markerfacecolor=c if r["sig"] else "white", markeredgecolor=c,
            markeredgewidth=1.0)


def build(by, arms, stem, height, npred_x, legend_loc):
    keys = sorted(by[REF])
    ref_f1 = st.mean(float(by[REF][k]["f1"]) for k in keys)
    A = [(a, contrast(by, a, REF)) for a in arms][::-1]   # bottom-up
    B = [(lab, hi, contrast(by, a, c)) for a, c, lab, hi in PANEL_B][::-1]
    d = np.sort(np.array([float(by["B3-seed3"][k]["f1"])
                          - float(by[REF][k]["f1"]) for k in keys]))

    fig = plt.figure(figsize=(S.W2 * S.MM, height * S.MM), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.04, 1],
                          height_ratios=[len(A), 11], wspace=.06, hspace=.28)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    for i, (arm, r) in enumerate(A):
        dot(axA, i, r, S.colour(arm), S.MARKER[S.category(arm)])
        axA.text(npred_x, i, f'{r["npred"]:.1f}', fontsize=S.FS["annot"],
                 color=S.MUTED, va="center", ha="right")
    axA.axvline(0, color=S.INK, lw=.8, zorder=1)
    axA.set_yticks(range(len(A)))
    axA.set_yticklabels([S.label(a) for a, _ in A], fontsize=S.FS["tick"])
    for t, (a, _) in zip(axA.get_yticklabels(), A):
        t.set_color(S.colour(a))
    axA.set_ylim(-.6, len(A) + .25)
    axA.set_xlim(-0.30, npred_x + .008)
    axA.set_xlabel("difference in query-macro F1 vs Flat-Adaptive",
                   fontsize=S.FS["axis"])
    axA.text(npred_x, len(A) - .15, "|pred|", fontsize=S.FS["annot"],
             color=S.MUTED, ha="right", va="center", style="italic")
    S.panel_title(axA, "A", "Comprehensive retrieval comparison",
                  f"{len(keys)} queries, 2 benchmarks, "
                  f"reference arm F1 = {ref_f1:.3f}")
    handles = [
        Line2D([], [], marker="s", color=S.OURS_C, markerfacecolor=S.OURS_C,
               ls="none", ms=4.4, label="our methods"),
        Line2D([], [], marker="o", color=S.PRIOR_C, markerfacecolor=S.PRIOR_C,
               ls="none", ms=4.4, label="prior / baseline"),
        Line2D([], [], marker="^", color=S.REF_C, markerfacecolor=S.REF_C,
               ls="none", ms=4.4, label="reference / ceiling"),
        Line2D([], [], marker="s", color=S.INK, markerfacecolor="white",
               ls="none", ms=4.4, label="tests disagree (hollow)")]
    axA.legend(handles=handles, loc=legend_loc, frameon=False,
               fontsize=S.FS["legend"], handletextpad=.5, borderpad=.1,
               labelspacing=.35)

    for i, (lab, emph, r) in enumerate(B):
        c = S.OURS_C if r["diff"] >= 0 else S.NEG_C
        dot(axB, i, r, c, "s", ms=5.2 if emph else 4.2, lw=1.4 if emph else 1.0)
    axB.axvline(0, color=S.INK, lw=.8, zorder=1)
    axB.set_yticks(range(len(B)))
    axB.set_yticklabels([b[0] for b in B], fontsize=S.FS["tick"])
    for t, (_, emph, _) in zip(axB.get_yticklabels(), B):
        if emph:
            t.set_color(S.INK)
            t.set_fontweight("bold")
    axB.set_ylim(-.6, len(B) - .4)
    axB.set_xlabel("difference in query-macro F1 vs the stated control",
                   fontsize=S.FS["axis"])
    S.panel_title(axB, "B", "Component ablation and controls",
                  "each contrast is paired and uses its own stated control")

    axC.bar(np.arange(len(d)), d, width=1.0,
            color=np.where(d > 0, S.OURS_C, S.PRIOR_C), linewidth=0)
    axC.axhline(0, color=S.INK, lw=.8)
    med = float(np.median(d))
    share = float(d[-max(1, len(d) // 10):].sum() / d.sum())
    axC.text(6, .98, f"{int((d > 0).sum())} better\n{int((d < 0).sum())} worse\n"
             f"{int((d == 0).sum())} unchanged\nmedian {med:+.3f}",
             fontsize=S.FS["annot"], color=S.MUTED, va="top", linespacing=1.5)
    axC.set_xlim(-2, len(d) + 1)
    axC.set_ylim(-.82, 1.05)
    axC.set_xlabel("queries, sorted by their own change", fontsize=S.FS["axis"])
    axC.set_ylabel("per-query change in F1", fontsize=S.FS["axis"])
    S.panel_title(axC, "C", "Query-level effect heterogeneity",
                  "Structure-3Seeds vs Flat-Adaptive")

    S.tidy(axA); S.tidy(axB); S.tidy(axC, xgrid=False, ygrid=True)

    from figure_export import export_figure
    export_figure(fig, str(OUT / stem), formats=["pdf", "png"], dpi=600,
                  bbox_inches=None, facecolor="white", font_mode="truetype",
                  overwrite=True, write_manifest=True,
                  provenance={
        "raw_data": [f"{d_}/discovery_per_query.csv" for d_, _ in LAKES],
        "metric": "query-macro F1 over the returned dataset set",
        "reference_arm": "Flat-Adaptive", "reference_arm_f1": round(ref_f1, 4),
        "arm_order": "as listed, not by effect size, so the main and appendix "
                     "versions place the same method comparably",
        "uncertainty": "95% bootstrap CI, 1000 resamples, seed 20260921, "
                       "clusters = (benchmark, parent task)",
        "test": "paired sign test, ties discarded",
        "significance_rule": "CI excludes zero AND sign-test p < 0.05",
        "panel_C_net_gain_denominator": "sum of all 245 signed differences, "
                                        "negatives included; against a "
                                        "positive-only denominator the share "
                                        "is 0.530",
        "retriever": "dense_chunked; bm25 rows excluded",
        "publisher_choices": "PENDING VERIFICATION — IEEE double-column width "
                             "assumed, no venue fixed"})
    plt.close(fig)
    return A, B, d, share, ref_f1


def main():
    by = load()
    log = HERE / "build_retrieval.log"
    with log.open("w") as fh:
        # 24 rows leave no room at the bottom, 10 rows leave none at the top
        for arms, stem, h, nx, loc in (
                (MAIN, "fig2_retrieval_main", 112, 0.205, "lower left"),
                (APPENDIX, "figA1_retrieval_appendix", 168, 0.205, "upper left")):
            A, B, d, share, ref = build(by, arms, stem, h, nx, loc)
            fh.write(f"\n=== {stem} — {len(A)} arms, reference {ref:.4f}\n")
            for arm, r in reversed(A):
                fh.write(f"  {S.label(arm):30}{r['diff']:+.4f} "
                         f"[{r['lo']},{r['hi']}] p={r['p']:.4f} "
                         f"|pred|={r['npred']:.1f} sig={r['sig']}\n")
            print(f"{stem}: {len(A)} arms")
        fh.write(f"\npanel C: top-decile share of net gain = {share:.4f}\n")
    print("log ->", log)


if __name__ == "__main__":
    with S.style_context("default", palette_name="okabe_ito_on_white"):
        main()
