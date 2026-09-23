"""Figure 5 — secondary: what the structure does to the generated text.

Optional. It reports one contrast in which every other input is held fixed —
the same generator, the same judge, the same sampled tables, the same prompt
blocks — and only the structure block is added or withheld. The result is
a null on the overall score, so the figure exists to show that, not to
support a claim.

One benchmark, 34 tables. That is stated on the figure, not only in the text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent          # figures/
sys.path.insert(0, str(HERE))
import fig_style as S                                        # noqa: E402

SRC = S.ROOT / "out_astro2" / "structure_arm.json"
DIMS = [("completeness", "Completeness"), ("conciseness", "Conciseness"),
        ("readability", "Readability"), ("mean", "Overall (mean)")]


def main():
    d = json.loads(SRC.read_text())
    pt = d["pointwise_paired"]

    fig, ax = plt.subplots(figsize=(S.W1 * S.MM, 46 * S.MM),
                           layout="constrained")
    for i, (k, lab) in enumerate(DIMS):
        r = pt[k]
        lo, hi = r["ci95_clustered_by_family"]
        c = S.OURS_C if r["diff"] >= 0 else S.NEG_C
        ax.plot([lo, hi], [i, i], color=c, lw=1.1, zorder=2,
                solid_capstyle="round")
        ax.plot(r["diff"], i, marker="s", ms=4.6, color=c, zorder=3,
                markerfacecolor=c if r["significant"] else "white",
                markeredgecolor=c, markeredgewidth=1.0)
    ax.axvline(0, color=S.INK, lw=.8, zorder=1)
    ax.set_yticks(range(len(DIMS)))
    ax.set_yticklabels([lab for _, lab in DIMS], fontsize=S.FS["tick"])
    ax.set_ylim(-.6, len(DIMS) - .4)
    ax.set_xlabel("difference in judge score (1–10)", fontsize=S.FS["axis"])
    S.panel_title(ax, "", "AutoDDG + Structure vs AutoDDG",
                  f"one benchmark, {d['n_tables']} tables; only the structure "
                  "block differs")
    ax.text(0.0, -1.35, "no marker is filled: no dimension clears both the "
            "interval and the sign test", fontsize=S.FS["annot"],
            color=S.MUTED, ha="center")
    S.tidy(ax)

    S.export(fig, "figA3_text_generation_secondary", {
        "research_question": "secondary — does the structure change the "
                             "generated description",
        "raw_data": "out_astro2/structure_arm.json",
        "protocol": "AutoDDG's own pointwise prompts, judge gpt-4o, "
                    "temperature 0.3",
        "held_fixed": "generator, judge, sampled tables, every prompt block "
                      "except the structure block",
        "uncertainty": "95% bootstrap CI clustered by group",
        "test": "paired sign test",
        "scope": "one benchmark (KramaBench), 34 tables; NOT run on SANTOS",
        "result_is_a_null": "the overall difference is 0.000; the figure "
                            "reports that rather than supporting a claim",
    })
    (HERE / "build_figA3.log").write_text(json.dumps(pt, indent=1))
    print("fig5 ok")


if __name__ == "__main__":
    with S.style_context("default", palette_name="okabe_ito_on_white"):
        main()
