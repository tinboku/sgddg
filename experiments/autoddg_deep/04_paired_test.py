"""Paired comparison of two arms with the query as the unit.

Usage:
    python -I 04_paired_test.py --eval results/eval/main.json --a baseline__r1 --b bugfix__r1 \
        --noise-a2 baseline__r2 --test randomization --n-perm 10000 --bootstrap 10000

For every (view, retriever, pool, subset) setting present for both arms, on the
queries scored for both:
  * mean delta (B - A), wins / ties / losses over queries
  * two-sided paired randomization test: random sign flips of the per-query
    deltas (Smucker et al., CIKM 2007), p = (#{|mean*| >= |mean|} + 1) / (n_perm + 1)
  * percentile bootstrap 95% CI of the mean delta, resampling queries
  * with --noise-a2: noise floor = |mean delta(A vs A2)| for the same setting
    (two generation runs of the same arm); "effective" requires |delta| above the
    noise floor and wins > losses in the direction of the delta (IMPL §5)

Writes results/paired/<a>__vs__<b>.json and .csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR  # noqa: E402

TIE_EPS = 1e-9


def randomization_test(deltas: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    observed = abs(deltas.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, deltas.size))
    perm_means = np.abs((signs * deltas).mean(axis=1))
    return float((np.sum(perm_means >= observed - 1e-15) + 1) / (n_perm + 1))


def bootstrap_ci(deltas: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05):
    idx = rng.integers(0, deltas.size, size=(n_boot, deltas.size))
    means = deltas[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def compare(sa: dict, sb: dict, metric: str, args, rng) -> dict:
    common = sorted(q for q in set(sa) & set(sb) if not np.isnan(sa[q][metric]) and not np.isnan(sb[q][metric]))
    a = np.array([sa[q][metric] for q in common])
    b = np.array([sb[q][metric] for q in common])
    d = b - a
    if d.size == 0:
        return {"n_queries": 0}
    wins = int(np.sum(d > TIE_EPS))
    losses = int(np.sum(d < -TIE_EPS))
    lo, hi = bootstrap_ci(d, args.bootstrap, rng)
    return {
        "n_queries": int(d.size),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta": float(d.mean()),
        "wins": wins,
        "ties": int(d.size - wins - losses),
        "losses": losses,
        "p_randomization": randomization_test(d, args.n_perm, rng),
        "ci95_low": lo,
        "ci95_high": hi,
        "per_query_delta": dict(zip(common, map(float, d))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--eval", type=Path, required=True, help="output of 03_index_eval.py")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--noise-a2", default=None, help="second run of arm A for the noise floor")
    ap.add_argument("--metrics", default="ndcg@10,ndcg@20")
    ap.add_argument("--test", default="randomization", choices=["randomization"])
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    per_query = json.loads(args.eval.read_text(encoding="utf-8"))["per_query"]

    def settings_of(arm: str) -> dict[str, dict]:
        out = {}
        for key, scores in per_query.items():
            name, view, *rest = key.split("|")
            if name == arm:
                out["|".join([view, *rest])] = scores
        return out

    A, B = settings_of(args.a), settings_of(args.b)
    A2 = settings_of(args.noise_a2) if args.noise_a2 else {}
    if args.a == "human" or args.b == "human":
        # the human arm has a single view; compare it against every view of the other arm
        if args.a == "human":
            A = {k: next(iter(A.values())) for k in B} if A else {}
        else:
            B = {k: next(iter(B.values())) for k in A} if B else {}
    shared = sorted(set(A) & set(B))
    if not shared:
        raise SystemExit(f"no shared settings for {args.a} and {args.b} in {args.eval}")

    rows = []
    for setting in shared:
        for metric in args.metrics.split(","):
            rng = np.random.default_rng(args.seed)
            res = compare(A[setting], B[setting], metric, args, rng)
            if res["n_queries"] == 0:
                continue
            noise = None
            if setting in A2:
                noise_res = compare(A[setting], A2[setting], metric, args, np.random.default_rng(args.seed))
                noise = abs(noise_res.get("delta", 0.0))
            direction_ok = res["wins"] > res["losses"] if res["delta"] > 0 else res["losses"] > res["wins"]
            res.update(
                setting=setting,
                metric=metric,
                noise_floor=noise,
                effective=(None if noise is None else bool(abs(res["delta"]) > noise and direction_ok)),
            )
            rows.append(res)
            print(
                f"{setting:42s} {metric:8s} n={res['n_queries']:2d} "
                f"A={res['mean_a']:.4f} B={res['mean_b']:.4f} d={res['delta']:+.4f} "
                f"W/T/L={res['wins']}/{res['ties']}/{res['losses']} p={res['p_randomization']:.4f} "
                f"CI=[{res['ci95_low']:+.4f},{res['ci95_high']:+.4f}]"
                + (f" noise={noise:.4f} effective={res['effective']}" if noise is not None else "")
            )

    out_dir = RESULTS_DIR / "paired"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.a}__vs__{args.b}"
    meta = {"eval": str(args.eval), "a": args.a, "b": args.b, "noise_a2": args.noise_a2,
            "n_perm": args.n_perm, "bootstrap": args.bootstrap, "seed": args.seed}
    (out_dir / f"{stem}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1), encoding="utf-8")
    fields = ["setting", "metric", "n_queries", "mean_a", "mean_b", "delta", "wins", "ties", "losses",
              "p_randomization", "ci95_low", "ci95_high", "noise_floor", "effective"]
    with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_dir / (stem + '.json')}")


if __name__ == "__main__":
    main()
