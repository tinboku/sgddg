"""Generate UFD/SFD for one arm with AutoDDG (original package or AutoDDGDeep).

Usage:
    python -I 02_generate.py --arm configs/arm_baseline.yaml --run-id r1 --provider openai   # needs OPENAI_API_KEY
    python -I 02_generate.py --arm configs/arm_bugfix.yaml --stages profile          # offline, no LLM
    python -I 02_generate.py --arm configs/arm_baseline.yaml --dry-run --limit 5     # fake LLM, offline
    python -I 02_generate.py --arm configs/arm_bugfix.yaml --run-id r1 --provider openai \
        --dataset-ids configs/subset30.txt                                           # fixed small subset

Per dataset the call sequence follows examples/quick_start.ipynb:
    sample (100 rows, random_state=9) -> profile_dataframe(full table) -> generate_topic
    -> analyze_semantics(sample, sequential) -> describe_dataset (UFD) -> expand_description_for_search (SFD)

F1 / F2 arms (config key ``facts``): the profile stage also builds the factset
(profiling_v2.pipeline) and renders the UFD / SFD fact blocks; describe_dataset gets
``ufd_facts`` and, with ``facts.mode: ufd_sfd``, expand_description_for_search gets
``sfd_facts``. Everything else is the bug-fix arm.

Outputs
    cache/profile/<arm>/<dataset_id>.json          profile stage (no LLM, shared by run ids)
    cache/gen/<arm>/<run_id>/<dataset_id>.json     all stages, prompts, per-call temperatures
    cache/llm/                                     content-addressed LLM responses
    results/descriptions/<arm>__<run_id>.jsonl     one row per corpus dataset

Datasets without a usable table get status "no_table" and no generation; 03_index_eval
indexes title + original description for them in every arm (same degraded input).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CACHE_DIR,
    DEFAULT_NTCIR_DIR,
    MANIFEST,
    RESULTS_DIR,
    TABLES_DIR,
    read_jsonl,
    select_benchmark,
    sha256_text,
    write_jsonl,
)
from llm_cache import CachingClient, make_inner_client  # noqa: E402
from tables import load_table  # noqa: E402

import autoddg  # noqa: E402
from autoddg import AutoDDG  # noqa: E402
from autoddg.autoddg_deep import AutoDDGDeep  # noqa: E402
from autoddg.profiling_v2.pipeline import build_factset, fact_texts  # noqa: E402
from autoddg.utils import get_sample  # noqa: E402

LLM_STAGES = ("topic", "semantic", "ufd", "sfd")
AUTODDG_SRC = Path(autoddg.__file__).resolve().parent


def autoddg_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "-C", str(AUTODDG_SRC), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(AUTODDG_SRC), "status", "--porcelain", "."],
            capture_output=True,
            text=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def prompt_hash() -> str:
    conf = AUTODDG_SRC / "configurations"
    texts = [(conf / name).read_text(encoding="utf-8") for name in ("prompts.yaml", "prompts_v2.yaml")]
    return sha256_text("\n".join(texts))[:16]


def config_hash(cfg: dict) -> str:
    return sha256_text(json.dumps(cfg, sort_keys=True))[:16]


def fact_mode(cfg: dict) -> str:
    return (cfg.get("facts") or {}).get("mode", "none")


class _NoClient:
    """Placeholder passed to AutoDDG's constructor; replaced before any call."""


def build_pipe(cfg: dict, client: CachingClient) -> AutoDDG:
    kwargs = {
        "description_temperature": float(cfg["description_temperature"]),
        "description_words": int(cfg["description_words"]),
        "topic_temperature": float(cfg["topic_temperature"]),
    }
    if cfg["pipeline"] == "autoddg":
        if fact_mode(cfg) != "none":
            raise ValueError("the original pipeline has no fact slots")
        if cfg.get("sfd_temperature") is not None:
            raise ValueError("the original pipeline cannot pass an SFD temperature")
        pipe = AutoDDG(_NoClient(), cfg["model"], **kwargs)
    elif cfg["pipeline"] == "autoddg_deep":
        pipe = AutoDDGDeep(
            _NoClient(), cfg["model"], sfd_temperature=float(cfg["sfd_temperature"]),
            fact_mode=fact_mode(cfg), **kwargs
        )
    else:
        raise ValueError(f"unknown pipeline {cfg['pipeline']!r}")
    for component in (
        pipe.description_generator,
        pipe.semantic_profiler,
        pipe.topic_generator,
        pipe.search_description,
    ):
        component.llm_client = client
    return pipe


def run_profile(pipe: AutoDDG, cfg: dict, dataset_id: str, table_sha: str, cfg_hash: str) -> dict:
    out = CACHE_DIR / "profile" / cfg["arm"] / f"{dataset_id}.json"
    if out.exists():
        cached = json.loads(out.read_text(encoding="utf-8"))
        if cached.get("table_sha256") == table_sha and cached.get("config_hash") == cfg_hash:
            return cached
    df, read_info = load_table(TABLES_DIR / f"{dataset_id}.csv", cfg.get("read_max_rows"))
    started = time.time()
    basic_profile, structural_profile = pipe.profile_dataframe(df)
    record = {
        "dataset_id": dataset_id,
        "arm": cfg["arm"],
        "config_hash": cfg_hash,
        "table_sha256": table_sha,
        "read": read_info,
        "basic_profile": basic_profile,
        "structural_profile": structural_profile,
        "profile_seconds": round(time.time() - started, 2),
    }
    if fact_mode(cfg) != "none":
        started = time.time()
        fs = build_factset(df, dataset_id, cfg["facts"].get("select"),
                           table={"sha256": table_sha, "read_status": read_info.get("read_status")})
        record.update(fact_texts(fs))
        record["factset"] = fs.to_dict()
        record["facts_seconds"] = round(time.time() - started, 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def generate_one(
    entry: dict, manifest_row: dict | None, cfg: dict, args: argparse.Namespace, meta: dict
) -> dict:
    dataset_id = entry["id"]
    base = {"dataset_id": dataset_id, "arm": meta["arm_tag"], "run_id": args.run_id}
    if not manifest_row or manifest_row["status"] not in {"ok", "truncated"}:
        return {**base, "status": "no_table", "download_status": (manifest_row or {}).get("status")}

    out = CACHE_DIR / "gen" / meta["arm_tag"] / args.run_id / f"{dataset_id}.json"
    want_llm = any(s in args.stages for s in LLM_STAGES)
    if out.exists():
        cached = json.loads(out.read_text(encoding="utf-8"))
        if (
            cached.get("config_hash") == meta["config_hash"]
            and cached.get("prompt_hash") == meta["prompt_hash"]
            and cached.get("table_sha256") == manifest_row["sha256"]
            and cached.get("status") == "ok"
            and (cached.get("ufd") is not None or not want_llm)
        ):
            return cached

    client = CachingClient(
        make_inner_client(meta["provider"]),
        CACHE_DIR / "llm",
        run_id=args.run_id,
        provider=meta["provider"],
        offline=args.offline,
    )
    record: dict[str, Any] = {
        **base,
        "config": cfg,
        "config_hash": meta["config_hash"],
        "prompt_hash": meta["prompt_hash"],
        "autoddg_commit": meta["autoddg_commit"],
        "provider": meta["provider"],
        "table_sha256": manifest_row["sha256"],
        "download_status": manifest_row["status"],
        "calls": {},
    }
    try:
        pipe = build_pipe(cfg, client)
        prof = run_profile(pipe, cfg, dataset_id, manifest_row["sha256"], meta["config_hash"])
        record.update(read=prof["read"], basic_profile=prof["basic_profile"],
                      structural_profile=prof["structural_profile"], fact_mode=fact_mode(cfg),
                      ufd_facts=prof.get("ufd_facts"), sfd_facts=prof.get("sfd_facts"))
        if not want_llm:
            record["status"] = "ok"
            return record

        df, _ = load_table(TABLES_DIR / f"{dataset_id}.csv", cfg.get("read_max_rows"))
        sample_df, dataset_sample = get_sample(df, int(cfg["sample_size"]), int(cfg["sample_random_state"]))

        original_description = entry.get("description") if cfg["topic_uses_original_description"] else None
        topic = pipe.generate_topic(entry.get("title") or "", original_description, dataset_sample)
        record["calls"]["topic"] = client.drain_log()

        if cfg["semantic_mode"] != "sequential":
            raise ValueError("only sequential semantic mode is wired")
        semantic_details = pipe.analyze_semantics(sample_df)
        record["calls"]["semantic"] = client.drain_log()
        semantic_profile = "\n".join(
            s for s in [prof["structural_profile"], semantic_details] if s
        )

        ufd_prompt, ufd = pipe.describe_dataset(
            dataset_sample=dataset_sample,
            dataset_profile=prof["basic_profile"],
            use_profile=True,
            semantic_profile=semantic_profile,
            use_semantic_profile=True,
            data_topic=topic,
            use_topic=True,
            **({"ufd_facts": prof.get("ufd_facts")} if fact_mode(cfg) != "none" else {}),
        )
        record["calls"]["ufd"] = client.drain_log()

        sfd_prompt, sfd = pipe.expand_description_for_search(
            ufd, topic, **({"sfd_facts": prof.get("sfd_facts")} if fact_mode(cfg) != "none" else {})
        )
        record["calls"]["sfd"] = client.drain_log()

        sfd_calls = record["calls"]["sfd"]
        record.update(
            status="ok",
            topic=topic,
            semantic_details=semantic_details,
            ufd_prompt_sha256=sha256_text(ufd_prompt),
            sfd_prompt_sha256=sha256_text(sfd_prompt),
            ufd_prompt=ufd_prompt,
            sfd_prompt=sfd_prompt,
            ufd=ufd,
            sfd=sfd,
            sfd_temperature_passed=sfd_calls[-1]["temperature_passed"] if sfd_calls else None,
            sfd_temperature_effective=sfd_calls[-1]["temperature_effective"] if sfd_calls else None,
        )
    except Exception as e:
        record.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--run-id", default="r1", help="replicate id; part of the LLM cache key")
    ap.add_argument("--stages", default="profile,topic,semantic,ufd,sfd")
    ap.add_argument("--dry-run", action="store_true", help="fake LLM; writes under <arm>__dryrun")
    ap.add_argument("--offline", action="store_true", help="LLM cache only; fail on miss")
    ap.add_argument(
        "--provider",
        default="none",
        choices=["none", "openai", "fake"],
        help="none = cache only (default; never calls an API); openai needs OPENAI_API_KEY",
    )
    ap.add_argument("--ntcir-dir", type=Path, default=DEFAULT_NTCIR_DIR)
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-relevant", type=int, default=3)
    ap.add_argument("--min-total", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated dataset ids (debug)")
    ap.add_argument("--dataset-ids", type=Path, default=None,
                    help="file with one dataset id per line (e.g. configs/subset30.txt); only these "
                         "are run, output goes to <arm>__<run_id>__<file stem>.jsonl")
    args = ap.parse_args()
    args.stages = tuple(s.strip() for s in args.stages.split(","))

    cfg = yaml.safe_load(args.arm.read_text(encoding="utf-8"))
    provider = "fake" if args.dry_run else args.provider
    meta = {
        "arm_tag": cfg["arm"] + ("__dryrun" if args.dry_run else ""),
        "provider": provider,
        "config_hash": config_hash(cfg),
        "prompt_hash": prompt_hash(),
        "autoddg_commit": autoddg_commit(),
    }
    if any(s in args.stages for s in LLM_STAGES) and provider == "openai" and not args.offline:
        make_inner_client(provider)  # fail fast when the API key is missing

    _, _, corpus = select_benchmark(
        args.ntcir_dir, args.split, args.min_relevant, args.min_total, csv_only=True
    )
    manifest = {r["dataset_id"]: r for r in read_jsonl(MANIFEST)}
    ids = sorted(corpus)
    if args.ids:
        ids = [i for i in args.ids.split(",") if i in corpus]
    if args.dataset_ids:
        listed = [l.strip() for l in args.dataset_ids.read_text(encoding="utf-8").splitlines() if l.strip()]
        missing = [i for i in listed if i not in corpus]
        if missing:
            raise SystemExit(f"{len(missing)} ids in {args.dataset_ids} are not in the corpus: {missing[:3]}")
        ids = [i for i in ids if i in set(listed)]
    if args.limit:
        usable = [i for i in ids if manifest.get(i, {}).get("status") in {"ok", "truncated"}]
        ids = usable[: args.limit]
    print(f"arm={meta['arm_tag']} run={args.run_id} stages={args.stages} datasets={len(ids)} "
          f"autoddg={meta['autoddg_commit']} config={meta['config_hash']} prompts={meta['prompt_hash']}")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_one, corpus[i], manifest.get(i), cfg, args, meta): i for i in ids
        }
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            rows.append(r)
            extra = r.get("error", "") if r["status"] == "error" else ""
            print(f"[{n}/{len(ids)}] {r['dataset_id']} {r['status']} {extra}", flush=True)

    rows.sort(key=lambda r: r["dataset_id"])
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"status counts: {counts}")
    if not any(s in args.stages for s in LLM_STAGES):
        return  # profile-only runs never overwrite a descriptions file
    keep = ("dataset_id", "arm", "run_id", "status", "download_status", "topic", "ufd", "sfd",
            "sfd_temperature_passed", "sfd_temperature_effective", "fact_mode", "ufd_facts", "sfd_facts",
            "config_hash", "prompt_hash",
            "autoddg_commit", "error")
    suffix = f"__{args.dataset_ids.stem}" if args.dataset_ids else ""
    out = RESULTS_DIR / "descriptions" / f"{meta['arm_tag']}__{args.run_id}{suffix}.jsonl"
    write_jsonl(out, [{k: r.get(k) for k in keep} for r in rows])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
