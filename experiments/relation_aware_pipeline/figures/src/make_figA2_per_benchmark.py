"""Figure 4 — the same methods on each benchmark and pooled.

Absolute F1, not a difference, because the question here is whether the two
benchmarks point the same way, and a difference plot hides that their
reference levels are 0.716 and 0.180. Dots on a common scale per panel; the
panels do not share an x range and say so.
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent          # figures/
sys.path.insert(0, str(HERE))
import fig_style as S                                        # noqa: E402

LAKES = (("out_santos", "SANTOS"), ("out_astro2", "KramaBench"))
ARMS = ["B0-adapt", "B1-D3L", "B1c-AutoDDG-adapt", "B2-family",
        "B3-structure", "B3-seed3", "B6-AutoDDG+structure"]


def main():
    by = defaultdict(dict)
    for d, lake in LAKES:
        for r in csv.DictReader((S.ROOT / d / "discovery_per_query.csv").open()):
            if r["retriever"] == "dense_chunked":
                r["lake"] = lake
                by[r["arm"]][f'{lake}|{r["query"]}'] = r
    keys = sorted(by["B0-adapt"])
    cols = [("SANTOS", [k for k in keys if k.startswith("SANTOS")]),
            ("KramaBench", [k for k in keys if k.startswith("KramaBench")]),
            ("Pooled", keys)]

    fig, axes = plt.subplots(1, 3, figsize=(S.W2 * S.MM, 52 * S.MM),
                             layout="constrained")
    for ax, (name, ks) in zip(axes, cols):
        vals = [(a, st.mean(float(by[a][k]["f1"]) for k in ks if k in by[a]))
                for a in ARMS]
        base = dict(vals)["B0-adapt"]
        for i, (a, v) in enumerate(vals):
            c = S.colour(a)
            ax.plot([base, v], [i, i], color=c, lw=.9, alpha=.45, zorder=1)
            ax.plot(v, i, marker=S.MARKER[S.category(a)], ms=5.0, color=c,
                    markerfacecolor=c, markeredgecolor=c, zorder=3)
        ax.axvline(base, color=S.INK, lw=.8, ls=(0, (3, 2)), zorder=2)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([S.label(a) for a, _ in vals]
                           if name == "SANTOS" else [],
                           fontsize=S.FS["tick"])
        if name == "SANTOS":
            for t, (a, _) in zip(ax.get_yticklabels(), vals):
                t.set_color(S.colour(a))
        ax.set_ylim(-.6, len(vals) - .4)
        lo = min(v for _, v in vals)
        hi = max(v for _, v in vals)
        pad = (hi - lo) * .22 + .01
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel("query-macro F1", fontsize=S.FS["axis"])
        S.panel_title(ax, "ABC"[cols.index((name, ks))], name,
                      f"n = {len(ks)}  ·  Flat-Adaptive = {base:.3f}")
        S.tidy(ax)

    S.export(fig, "figA2_per_benchmark_summary", {
        "research_question": "RQ2 — do the two benchmarks point the same way",
        "raw_data": [f"{d}/discovery_per_query.csv" for d, _ in LAKES],
        "metric": "query-macro F1, absolute (not a difference)",
        "axis_note": "the three panels do NOT share an x range; the dashed "
                     "line in each is that panel's own Flat-Adaptive level",
        "no_intervals_drawn": "this panel shows levels only; the paired "
                              "intervals are in Figure 2A",
    })
    with (HERE / "build_figA2.log").open("w") as fh:
        for name, ks in cols:
            fh.write(f"{name} n={len(ks)}\n")
            for a in ARMS:
                v = st.mean(float(by[a][k]["f1"]) for k in ks if k in by[a])
                fh.write(f"   {S.label(a):28}{v:.4f}\n")
    print("fig4 ok")


if __name__ == "__main__":
    with S.style_context("default", palette_name="okabe_ito_on_white"):
        main()
