"""Figure 3 — one query, three panels: what is asked, what flat retrieval
returns, what group completion returns.

The example is not chosen to flatter the method. It is the query where the
gold set is exactly one inferred group, so the mechanism is visible without
argument; the caption states that this is the favourable case and Figure 2C
gives the distribution over all queries.

Everything on this panel is read from the artifacts: the question text, the
gold set, the group and its type, and both arms' returned sets.
"""
from __future__ import annotations

import csv
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE.parent          # figures/
sys.path.insert(0, str(HERE))
import fig_style as S                                        # noqa: E402

CASE = ("out_astro2", "environment-hard-8-7")
GROUP = "G018"


def short(t: str) -> str:
    n = Path(t).name.replace(".csv", "").replace("_datasheet", "")
    return n.replace("_", " ")


def main():
    lake, qid = CASE
    q = {x["id"]: x for x in json.loads(
        (S.ROOT / lake / "questions.json").read_text())}[qid]
    gold = list(q["gold"])
    st = json.loads((S.ROOT / lake / "structure.json").read_text())
    fam = st["nodes"]["Family"][GROUP]
    per = defaultdict(dict)
    for r in csv.DictReader((S.ROOT / lake / "discovery_per_query.csv").open()):
        if r["retriever"] == "dense_chunked":
            per[r["arm"]][r["query"]] = r
    flat, struct = per["B0-adapt"][qid], per["B3-seed3"][qid]

    fig = plt.figure(figsize=(S.W2 * S.MM, 62 * S.MM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, S.W2)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def tile(x, y, w, h, fill, edge, lw=.5, dash=None):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                               linewidth=lw, zorder=3,
                               linestyle=dash or "solid"))
        ax.add_patch(Rectangle((x, y + h - 1.0), w, 1.0, facecolor=edge,
                               edgecolor=edge, linewidth=lw, zorder=4,
                               alpha=.35))

    # ---- left: the question and the seed --------------------------------
    ax.text(4, 56.5, "A   The query", fontsize=S.FS["title"],
            fontweight="bold", color=S.INK)
    ax.add_patch(FancyBboxPatch((4, 36), 46, 16,
                                boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor="#F6F7F8", edgecolor=S.HAIR,
                                linewidth=.6, zorder=2))
    # wrap on words: slicing by character count overflowed into panel B
    wrapped = "\n".join(textwrap.wrap(q["question"], 42)[:4])
    ax.text(5.6, 49.6, "“" + wrapped + "”", fontsize=S.FS["annot"],
            color=S.INK, va="top", linespacing=1.6)
    ax.text(5.6, 39.0, f"answer: {len(gold)} datasets", fontsize=S.FS["annot"],
            color=S.MUTED)
    ax.text(4, 31, "top-ranked dataset (the seed)", fontsize=S.FS["annot"],
            color=S.MUTED)
    tile(4, 22, 20, 6.4, "#FBE7D8", S.OURS_C, lw=1.2)
    ax.text(14, 25.2, short(gold[1]), fontsize=S.FS["annot"], ha="center",
            color=S.INK)
    ax.text(4, 16.5, f"inferred group {GROUP}\n{fam['type'].replace('_',' ')}, "
            f"{fam['size']} members", fontsize=S.FS["annot"], color=S.MUTED,
            va="top", linespacing=1.5)

    # ---- middle: flat ----------------------------------------------------
    ax.text(58, 56.5, "B   Flat-Adaptive", fontsize=S.FS["title"],
            fontweight="bold", color=S.INK)
    ax.text(58, 52.6, f"returns {float(flat['n_pred']):.0f} datasets  ·  "
            f"F1 {float(flat['f1']):.3f}", fontsize=S.FS["annot"],
            color=S.MUTED)
    ax.text(58, 47.2, "one member, then it drifts into a\n"
                      "different group that shares a subject",
            fontsize=S.FS["annot"], color=S.MUTED, va="top", linespacing=1.5)
    hit = short(gold[1])
    rows = [(hit, True)] + [(f"water-body-testing-{y}", False)
                            for y in (2023, 2020, 2019, 2015, 2002)]
    for i, (nm, ok) in enumerate(rows):
        y = 36.8 - i * 5.8
        tile(58, y, 30, 4.6, "#FBE7D8" if ok else "white",
             S.OURS_C if ok else S.HAIR, lw=1.0 if ok else .5)
        ax.text(59.4, y + 1.6, nm[:26], fontsize=S.FS["annot"],
                color=S.INK if ok else S.MUTED)
        if not ok:
            ax.text(89.4, y + 1.6, "not in the answer",
                    fontsize=S.FS["annot"], color=S.PRIOR_C)
    ax.text(58, 6.2, "…and 16 more", fontsize=S.FS["annot"], color=S.MUTED)

    # ---- right: structure -------------------------------------------------
    ax.text(123, 56.5, "C   Structure-3Seeds", fontsize=S.FS["title"],
            fontweight="bold", color=S.INK)
    ax.text(123, 52.6, f"returns {float(struct['n_pred']):.0f} datasets  ·  "
            f"F1 {float(struct['f1']):.3f}", fontsize=S.FS["annot"],
            color=S.MUTED)
    ax.text(123, 48.0, "the seed's group, completed",
            fontsize=S.FS["annot"], color=S.MUTED, va="top")
    ax.add_patch(FancyBboxPatch((121.5, 5.5), 56, 38,
                                boxstyle="round,pad=0,rounding_size=1.8",
                                facecolor="#FDF3EC", edgecolor=S.OURS_C,
                                linewidth=1.0, linestyle=(0, (3, 2)),
                                zorder=1))
    for i, g in enumerate(gold):
        col, row = i % 2, i // 2
        x = 123.5 + col * 27.5
        y = 37.0 - row * 8.0
        seed = (g == gold[1])
        tile(x, y, 25, 4.6, "#FBE7D8" if seed else "white",
             S.OURS_C, lw=1.2 if seed else .7)
        ax.text(x + 1.3, y + 1.6, short(g)[:24], fontsize=S.FS["annot"],
                color=S.INK)
    ax.text(123.5, 2.4, f"{GROUP}: every member returned, nothing else",
            fontsize=S.FS["annot"], color=S.OURS_C)

    for x0, x1 in ((50.5, 57.0), (115.5, 121.0)):
        ax.add_patch(FancyArrowPatch((x0, 30), (x1, 30),
                                     arrowstyle="-|>", mutation_scale=7,
                                     linewidth=1.0, color=S.MUTED, zorder=5))

    S.export(fig, "fig3_group_completion_case_study", {
        "research_question": "RQ2 — the mechanism behind the change",
        "case": f"{lake}/{qid}",
        "raw_data": [f"{lake}/questions.json", f"{lake}/structure.json",
                     f"{lake}/discovery_per_query.csv"],
        "selection_rule": "a query whose gold set equals one inferred group "
                          "exactly; this is the favourable case and the "
                          "caption says so",
        "gold_size": len(gold),
        "flat_f1": float(flat["f1"]), "structure_f1": float(struct["f1"]),
        "not_generalisable_from": "one query; Figure 2C gives the "
                                  "distribution over all 245",
    })
    (HERE / "build_fig3.log").write_text(
        f"case {lake}/{qid}\nquestion: {q['question']}\n"
        f"gold {len(gold)}: {[Path(g).name for g in gold]}\n"
        f"group {GROUP} type={fam['type']} size={fam['size']}\n"
        f"flat  n_pred={flat['n_pred']} P={flat['precision']} "
        f"R={flat['recall']} F1={flat['f1']}\n"
        f"struct n_pred={struct['n_pred']} P={struct['precision']} "
        f"R={struct['recall']} F1={struct['f1']}\n")
    print("fig3 ok")


if __name__ == "__main__":
    with S.style_context("default", palette_name="okabe_ito_on_white"):
        main()
