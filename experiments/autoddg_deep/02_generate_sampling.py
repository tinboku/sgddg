"""Direction 3: same-budget sample selection for the Semantic Profiler (5 values / column).

Usage:
    python -I 02_generate_sampling.py --strategies orig,freq,dist --run-id r1 --provider openai
    python -I 02_generate_sampling.py --strategies orig,freq,dist --run-id r1 --provider openai --offline  # cache only

Only the values the SemanticProfiler sees per column change; everything else is the
bug-fix arm (configs/arm_bugfix.yaml): topic, 100-row dataset sample, basic/structural
profile, UFD and SFD prompts.

    orig  AutoDDG: 5 rows of the 100-row sample, random_state=9 (= bugfix r1, cache hits)
    freq  the 5 most frequent non-blank values of the full column, as "value (n=count)"
    dist  full-column, interpretable, no model:
            numeric (profiling_v2.stats measure_range) -> real values at min/25/50/75/max
            mixed formats (stats.value_format, >= 2 shapes with >= 5%, one with digits) -> one example
                per shape, then frequent values
            other -> top-3 frequent + 2 rarest values (ties: random_state=9)
          columns with <= 5 distinct values get all of them in freq and dist

The topic is always read from run r1 (it does not depend on the strategy), so a second
run id replicates only semantic -> UFD -> SFD (noise floor at T=0).

Outputs
    cache/gen/S3_<strategy>/<run_id>/<dataset_id>.json   per column samples, raw semantic
                                                         JSON, strategy, prompts, UFD, SFD, calls
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, HERE, MANIFEST, TABLES_DIR, read_jsonl, sha256_text  # noqa: E402
from llm_cache import CachingClient, make_inner_client  # noqa: E402
from tables import load_table  # noqa: E402

from autoddg.profiling_v2 import stats  # noqa: E402
from autoddg.utils import get_sample  # noqa: E402

gen = __import__("02_generate")  # build_pipe, config_hash, ... of the bug-fix arm

N_VALUES = 5
SEED = 9
MIXED_MIN_RATE = 0.05
STRATEGIES = ("orig", "freq", "dist")


# --- sample selection ---------------------------------------------------------------

def _values(s: pd.Series) -> pd.Series:
    nonblank = s[~stats._blank(s)]
    return nonblank.astype(str).str.strip()


def numeric_range(name: str, s: pd.Series) -> dict | None:
    for f in stats.column_facts(name, s):
        if f.kind == "measure_range" and not f.value.get("mixed_units"):
            return f.value
    return None


def _parsed(s: pd.Series, vals: pd.Series) -> pd.Series:
    if stats.physical_type(s) in ("integer", "float"):
        return pd.to_numeric(s.loc[vals.index], errors="coerce")
    return stats._numeric_from_affixed(vals)


def mixed_shapes(s: pd.Series) -> list[dict]:
    fmt = stats.value_format(s)
    if not fmt:
        return []
    shapes = [p for p in fmt["patterns"] if p["rate"] >= MIXED_MIN_RATE]
    covered = sum(p["rate"] for p in fmt["patterns"])
    # word-count shapes of free text are not formats: at least one shape must hold digits
    coded = any(r"\d" in p["pattern"] for p in shapes)
    return shapes if len(shapes) >= 2 and covered >= 0.8 and coded else []


def column_profile(name: str, s: pd.Series) -> dict:
    """Facts about the full column used both for selection and for the checks."""
    vals = _values(s)
    vc = vals.value_counts()
    rng = numeric_range(name, s)
    out = {"n_nonblank": int(len(vals)), "n_distinct": int(vc.size),
           "top": [[v, int(c)] for v, c in vc.head(10).items()],
           "numeric": rng is not None, "mixed_shapes": len(mixed_shapes(s))}
    if rng is not None and "quantiles" in rng:
        q = rng["quantiles"]
        iqr = q["q75"] - q["q25"]
        out["heavy_tail"] = bool(rng["max"] > q["q75"] + 3 * iqr or rng["min"] < q["q25"] - 3 * iqr)
    else:
        out["heavy_tail"] = False
    top5_share = float(vc.head(5).sum() / len(vals)) if len(vals) else 1.0
    is_key = vc.size == len(vals)
    out["long_tail"] = bool(not out["numeric"] and not is_key and vc.size > 10 and top5_share < 0.8)
    out["complex"] = bool(out["heavy_tail"] or out["long_tail"] or out["mixed_shapes"])
    return out


def _fill(picked: list[str], pool: list[str], n: int = N_VALUES) -> list[str]:
    for v in pool:
        if len(picked) >= n:
            break
        if v not in picked:
            picked.append(v)
    return picked


def select_freq(name: str, s: pd.Series) -> tuple[list[str], str]:
    vc = _values(s).value_counts()
    if vc.empty:
        return [], "all_blank"
    return [f"{v} (n={c})" for v, c in vc.head(N_VALUES).items()], "top_frequency"


def select_dist(name: str, s: pd.Series) -> tuple[list[str], str]:
    s = s.reset_index(drop=True)  # some tables have duplicate index labels
    vals = _values(s)
    vc = vals.value_counts()
    if vc.empty:
        return [], "all_blank"
    if vc.size <= N_VALUES:
        return list(vc.index), "small_domain"
    rng = numeric_range(name, s)
    if rng is not None and "quantiles" in rng:
        num = _parsed(s, vals).dropna()
        picked = []
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            target = float(np.quantile(num, q, method="inverted_cdf"))
            idx = (num - target).abs().idxmin()  # a real cell at that quantile
            if vals.loc[idx] not in picked:
                picked.append(vals.loc[idx])
        return _fill(picked, list(vc.index)), "numeric_quantiles"
    shapes = mixed_shapes(s)
    if shapes:
        picked = [p["example"] for p in shapes]
        return _fill(picked, list(vc.index)), "mixed_format"
    head = list(vc.index[:3])
    rest = vc.iloc[3:]
    rarest = rest[rest == rest.min()]
    tail = list(rarest.sample(min(2, rarest.size), random_state=SEED).index)
    if len(tail) < 2:  # only one value at the minimum count: take the next rarest
        tail += [v for v in rest.sort_values(kind="stable").index if v not in tail][: 2 - len(tail)]
    return head + tail, "head3_tail2"


def select_orig(sample_df: pd.DataFrame) -> dict[str, list[str]]:
    """SemanticProfiler.analyze_dataframe's own sample (5 rows, random_state=9)."""
    frame = sample_df.sample(N_VALUES, random_state=SEED) if N_VALUES < len(sample_df) else sample_df
    return {c: frame[c].fillna("").astype(str).tolist() for c in sample_df.columns}


# --- semantic profiling with given samples -----------------------------------------

def profile_semantics(profiler, columns: list[str], samples: dict[str, list[str]]) -> tuple[str, dict]:
    """Sequential mode of SemanticProfiler.analyze_dataframe with the samples supplied."""
    raw = {}
    summary = []
    for col in columns:
        desc = None
        for _ in range(3):
            desc = profiler.get_semantic_type(col, samples[col])
            if desc is not None:
                break
        raw[col] = desc
        if desc is not None:
            summary.append(profiler._create_column_summary(col, desc))
    text = "The key semantic information for this dataset includes:\n" + "\n".join(summary)
    return text, raw


# --- per table ----------------------------------------------------------------------

def run_one(dataset_id: str, entry_title: str, entry_desc: str | None, manifest_row: dict,
            cfg: dict, strategy: str, args, meta: dict) -> dict:
    out = CACHE_DIR / "gen" / f"S3_{strategy}" / args.run_id / f"{dataset_id}.json"
    if out.exists() and not args.force:
        cached = json.loads(out.read_text(encoding="utf-8"))
        if cached.get("status") == "ok" and cached.get("config_hash") == meta["config_hash"]:
            return cached
    provider = meta["provider"]
    inner = None if args.offline else make_inner_client(provider)
    topic_client = CachingClient(inner, CACHE_DIR / "llm", run_id="r1", provider=provider, offline=args.offline)
    client = CachingClient(inner, CACHE_DIR / "llm", run_id=args.run_id, provider=provider, offline=args.offline)
    rec = {"dataset_id": dataset_id, "arm": f"S3_{strategy}", "strategy": strategy, "run_id": args.run_id,
           "config_hash": meta["config_hash"], "autoddg_commit": meta["autoddg_commit"],
           "table_sha256": manifest_row["sha256"], "calls": {}}
    try:
        pipe = gen.build_pipe(cfg, topic_client)
        # the profile the bug-fix arm actually used (re-running it would rewrite its cache)
        prof = json.loads((CACHE_DIR / "gen" / "bugfix" / "r1" / f"{dataset_id}.json").read_text(encoding="utf-8"))
        if prof.get("status") != "ok" or prof.get("table_sha256") != manifest_row["sha256"]:
            raise ValueError("no usable bugfix r1 record for this table")
        df, _ = load_table(TABLES_DIR / f"{dataset_id}.csv", cfg.get("read_max_rows"))
        sample_df, dataset_sample = get_sample(df, int(cfg["sample_size"]), int(cfg["sample_random_state"]))
        topic = pipe.generate_topic(entry_title or "", entry_desc, dataset_sample)
        rec["calls"]["topic"] = topic_client.drain_log()
        for comp in (pipe.description_generator, pipe.semantic_profiler, pipe.search_description):
            comp.llm_client = client

        columns = list(sample_df.columns)
        orig = select_orig(sample_df)
        samples, strat = {}, {}
        for col in columns:
            if strategy == "orig":
                samples[col], strat[col] = orig[col], "random5"
            else:
                pick = select_freq if strategy == "freq" else select_dist
                samples[col], strat[col] = pick(str(col), df[col])
                if strat[col] == "all_blank":
                    samples[col] = orig[col]  # nothing to choose from: same input as orig
        semantic_details, raw = profile_semantics(pipe.semantic_profiler, columns, samples)
        rec["calls"]["semantic"] = client.drain_log()
        semantic_profile = "\n".join(x for x in [prof["structural_profile"], semantic_details] if x)
        ufd_prompt, ufd = pipe.describe_dataset(
            dataset_sample=dataset_sample, dataset_profile=prof["basic_profile"], use_profile=True,
            semantic_profile=semantic_profile, use_semantic_profile=True, data_topic=topic, use_topic=True,
        )
        rec["calls"]["ufd"] = client.drain_log()
        sfd_prompt, sfd = pipe.expand_description_for_search(ufd, topic)
        rec["calls"]["sfd"] = client.drain_log()
        rec.update(
            status="ok", topic=topic, columns=[str(c) for c in columns],
            samples={str(c): samples[c] for c in columns}, column_strategy={str(c): strat[c] for c in columns},
            column_stats={str(c): column_profile(str(c), df[c]) for c in columns},
            semantic_raw={str(c): raw[c] for c in columns}, semantic_details=semantic_details,
            ufd_prompt=ufd_prompt, sfd_prompt=sfd_prompt, ufd=ufd, sfd=sfd,
            ufd_prompt_sha256=sha256_text(ufd_prompt),
        )
    except Exception as e:
        rec.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-1500:])
    gen._write_json(out, rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--strategies", default="orig,freq,dist")
    ap.add_argument("--run-id", default="r1")
    ap.add_argument("--provider", default="none", choices=["none", "openai"])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild records (LLM calls still come from cache)")
    ap.add_argument("--dataset-ids", type=Path, default=HERE / "configs" / "subset30.txt")
    ap.add_argument("--arm", type=Path, default=HERE / "configs" / "arm_bugfix.yaml")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if any(s not in STRATEGIES for s in strategies):
        raise SystemExit(f"strategies must be among {STRATEGIES}")

    cfg = yaml.safe_load(args.arm.read_text(encoding="utf-8"))
    meta = {"provider": args.provider, "config_hash": gen.config_hash(cfg),
            "autoddg_commit": gen.autoddg_commit()}
    _, _, corpus = gen.select_benchmark(gen.DEFAULT_NTCIR_DIR, "test", 3, 1, csv_only=True)
    manifest = {r["dataset_id"]: r for r in read_jsonl(MANIFEST)}
    ids = [l.strip() for l in args.dataset_ids.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [i for i in ids if manifest.get(i, {}).get("status") in {"ok", "truncated"}]
    print(f"strategies={strategies} run={args.run_id} tables={len(ids)} autoddg={meta['autoddg_commit']}")

    jobs = [(s, i) for s in strategies for i in ids]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for s, i in jobs:
            e = corpus[i]
            desc = e.get("description") if cfg["topic_uses_original_description"] else None
            futs[pool.submit(run_one, i, e.get("title"), desc, manifest[i], cfg, s, args, meta)] = (s, i)
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            print(f"[{n}/{len(jobs)}] {r['arm']} {r['dataset_id'][:8]} {r['status']} {r.get('error', '')[:120]}",
                  flush=True)


if __name__ == "__main__":
    main()
