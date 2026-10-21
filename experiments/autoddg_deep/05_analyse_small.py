"""Descriptive analysis for a small-subset run (E0/E1): input/output diffs and per-query deltas.

Usage:
    python -I 05_analyse_small.py --subset configs/subset30.txt --eval results/eval/e0e1_subset30.json \
        --a baseline__r1 --a2 baseline__r2 --b bugfix__r1 --setting "sfd|bm25|per_query|all"

Reports, for the subset tables:
  * status per arm (ok / error / no_table)
  * basic_profile differs between arm A and arm B (the summary_fixed coverage fix)
    and how many profile lines differ
  * UFD/SFD text identical between A and A2 (generation noise) and between A and B
and, for one eval setting, the per-query NDCG deltas B-A and A2-A side by side,
with the number of judged subset tables per query.
Writes results/eval/<eval stem>_analysis.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, DEFAULT_NTCIR_DIR, RESULTS_DIR, read_jsonl, select_benchmark  # noqa: E402


def gen_record(arm_run: str, did: str) -> dict:
    arm, run = arm_run.split("__")
    p = CACHE_DIR / "gen" / arm / run / f"{did}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--a", default="baseline__r1")
    ap.add_argument("--a2", default="baseline__r2")
    ap.add_argument("--b", default="bugfix__r1")
    ap.add_argument("--setting", default="sfd|bm25|per_query|all")
    ap.add_argument("--metrics", default="ndcg@10,ndcg@20")
    args = ap.parse_args()

    ids = [l.strip() for l in args.subset.read_text(encoding="utf-8").splitlines() if l.strip()]
    stem = args.subset.stem
    arms = {"a": args.a, "a2": args.a2, "b": args.b}
    rows = {k: {r["dataset_id"]: r for r in read_jsonl(RESULTS_DIR / "descriptions" / f"{v}__{stem}.jsonl")}
            for k, v in arms.items()}

    status = {k: {} for k in arms}
    for k in arms:
        for did in ids:
            s = rows[k].get(did, {}).get("status", "missing")
            status[k][s] = status[k].get(s, 0) + 1

    tables = []
    for did in ids:
        ga, gb = gen_record(args.a, did), gen_record(args.b, did)
        pa, pb = (ga.get("basic_profile") or "").splitlines(), (gb.get("basic_profile") or "").splitlines()
        t = {
            "dataset_id": did,
            "status": {k: rows[k].get(did, {}).get("status") for k in arms},
            "profile_lines_diff_a_b": sum(x != y for x, y in zip(pa, pb)) + abs(len(pa) - len(pb)),
        }
        for view in ("ufd", "sfd"):
            va, va2, vb = (rows[k].get(did, {}).get(view) for k in ("a", "a2", "b"))
            t[f"{view}_same_a_a2"] = va is not None and va == va2
            t[f"{view}_same_a_b"] = va is not None and va == vb
        tables.append(t)
    ok = [t for t in tables if all(s == "ok" for s in t["status"].values())]
    text_summary = {
        "n_ok_all_arms": len(ok),
        "profile_differs_a_b": sum(t["profile_lines_diff_a_b"] > 0 for t in ok),
        **{f"{v}_identical_{p}": sum(t[f"{v}_same_{p}"] for t in ok)
           for v in ("ufd", "sfd") for p in ("a_a2", "a_b")},
    }

    queries, qrels, corpus = select_benchmark(DEFAULT_NTCIR_DIR, "test", 3, 1, csv_only=True)
    subset = set(ids)
    per_query = json.loads(args.eval.read_text(encoding="utf-8"))["per_query"]
    key = lambda arm: f"{arm}__{stem}|{args.setting}"  # noqa: E731
    sa, sa2, sb = per_query[key(args.a)], per_query[key(args.a2)], per_query[key(args.b)]
    qtable = []
    for q in sorted(set(sa) & set(sb) & set(sa2)):
        row = {
            "qid": q,
            "query": queries[q],
            "n_judged": len(qrels[q]),
            "n_judged_subset": sum(d in subset for d in qrels[q]),
            "n_relevant_subset": sum(d in subset and r > 0 for d, r in qrels[q].items()),
        }
        for m in args.metrics.split(","):
            row[f"{m}_a"] = sa[q][m]
            row[f"{m}_delta_b_a"] = sb[q][m] - sa[q][m]
            row[f"{m}_delta_a2_a"] = sa2[q][m] - sa[q][m]
            row[f"{m}_delta_b_a2"] = sb[q][m] - sa2[q][m]
        qtable.append(row)

    out = {"arms": arms, "setting": args.setting, "status": status, "text_summary": text_summary,
           "tables": tables, "per_query": qtable}
    path = args.eval.with_name(args.eval.stem + "_analysis.json")
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": status, "text_summary": text_summary}, indent=1))
    m = args.metrics.split(",")[0]
    print(f"\nper-query {m} ({args.setting}), only queries with a judged subset table or a nonzero delta:")
    print("qid  judged subset rel_sub   A      B-A     A2-A    B-A2")
    for r in qtable:
        if r["n_judged_subset"] or any(abs(r[f"{m}_delta_{p}"]) > 1e-9 for p in ("b_a", "a2_a")):
            print(f"{r['qid']:5s} {r['n_judged']:4d} {r['n_judged_subset']:5d} {r['n_relevant_subset']:5d}  "
                  f"{r[f'{m}_a']:.4f} {r[f'{m}_delta_b_a']:+.4f} {r[f'{m}_delta_a2_a']:+.4f} {r[f'{m}_delta_b_a2']:+.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
