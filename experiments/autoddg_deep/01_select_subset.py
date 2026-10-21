"""Draw the fixed small-scale subset of tables used by every small experiment.

Usage:
    python -I 01_select_subset.py --n 30 --seed 9 --out configs/subset30.txt

Rule: candidates = corpus datasets (02_generate's corpus: judged pool of the 32
test queries, CSV only) whose download succeeded (manifest status ok/truncated)
and whose CSV load_table() can read. Candidates are sorted by dataset id and
pandas.Series.sample(n, random_state=seed) draws the subset. Wide tables are
not excluded at this step; they are handled at generation time (error ->
title + original description in every arm).

Writes the id list (one per line, sorted) and <out>.json with the rule,
candidate counts and the unreadable ids.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_NTCIR_DIR, MANIFEST, TABLES_DIR, read_jsonl, select_benchmark  # noqa: E402
from tables import load_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--out", type=Path, default=Path("configs/subset30.txt"))
    ap.add_argument("--ntcir-dir", type=Path, default=DEFAULT_NTCIR_DIR)
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-relevant", type=int, default=3)
    ap.add_argument("--min-total", type=int, default=1)
    args = ap.parse_args()

    _, _, corpus = select_benchmark(
        args.ntcir_dir, args.split, args.min_relevant, args.min_total, csv_only=True
    )
    manifest = {r["dataset_id"]: r for r in read_jsonl(MANIFEST)}
    downloaded = sorted(d for d in corpus if manifest.get(d, {}).get("status") in {"ok", "truncated"})
    readable, unreadable = [], {}
    for did in downloaded:
        try:
            df, _ = load_table(TABLES_DIR / f"{did}.csv")
            if df.shape[1] == 0:
                raise ValueError("no columns")
            readable.append(did)
        except Exception as e:
            unreadable[did] = f"{type(e).__name__}: {e}"[:200]

    subset = sorted(pd.Series(readable).sample(n=args.n, random_state=args.seed).tolist())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(subset) + "\n", encoding="utf-8")
    meta = {
        "rule": "corpus ∩ manifest status in {ok, truncated} ∩ load_table readable; "
        "sorted ids; pandas.Series.sample(n, random_state=seed)",
        "n": args.n,
        "seed": args.seed,
        "n_corpus": len(corpus),
        "n_downloaded": len(downloaded),
        "n_readable": len(readable),
        "unreadable": unreadable,
        "subset": subset,
        "subset_n_cols": {d: manifest[d].get("n_cols") for d in subset},
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"corpus={len(corpus)} downloaded={len(downloaded)} readable={len(readable)} "
          f"unreadable={len(unreadable)} -> {args.out} ({len(subset)} ids)")


if __name__ == "__main__":
    main()
