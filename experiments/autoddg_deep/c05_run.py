"""C05 paired diagnostic: original NL profile input vs structured key-value input.

Both arms reuse the F2/r1 cache (same table sample, topic, semantic profile, factset,
prompt template, model, temperature, run id). Only the representation of the profile
evidence in the UFD prompt and of the facts in the SFD prompt differs
(c05_structured.py). The original arm is re-executed through the same generator code;
its prompts and outputs must equal the F2/r1 cache byte for byte (LLM cache hits).

    python -I c05_run.py --provider openai      # generation (structured arm = new calls)
    python -I c05_run.py --stage analyse        # offline audit -> results/c05/
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, HERE, RESULTS_DIR, TABLES_DIR  # noqa: E402
from llm_cache import CachingClient, make_inner_client  # noqa: E402
from tables import load_table  # noqa: E402

import c05_structured as S  # noqa: E402
from autoddg.description import fact_coverage as cov  # noqa: E402
from autoddg.utils import get_sample  # noqa: E402

spec = importlib.util.spec_from_file_location("gen02", HERE / "02_generate.py")
gen02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen02)

OUT = RESULTS_DIR / "c05"
PRICE = {"in": 0.15 / 1e6, "out": 0.60 / 1e6}  # gpt-4o-mini, USD per token
ARMS = ("orig", "struct")


def ids() -> list[str]:
    return (HERE / "configs/c05_tables.txt").read_text().split()


def load_source(i: str):
    src = json.loads((CACHE_DIR / "gen/F2/r1" / f"{i}.json").read_text())
    prof = json.loads((HERE / src["profile_cache"]).read_text())
    assert src["status"] == "ok" and prof["profile_key"] == src["profile_key"]
    assert prof["table_sha256"] == src["table_sha256"]
    assert S.original_fact_block(prof["factset"], "ufd") == src["ufd_facts"] == prof["ufd_facts"]
    assert S.original_fact_block(prof["factset"], "sfd") == src["sfd_facts"] == prof["sfd_facts"]
    return src, prof


def generate(args) -> None:
    cfg = yaml.safe_load((HERE / "configs/arm_F2.yaml").read_text())
    inner = None if args.provider == "none" else make_inner_client("openai")
    for i in ids():
        src, prof = load_source(i)
        client = CachingClient(inner, CACHE_DIR / "llm", run_id="r1", provider="openai",
                               offline=args.provider == "none")
        pipe = gen02.build_pipe(cfg, client)
        df, _ = load_table(TABLES_DIR / f"{i}.csv", cfg.get("read_max_rows"))
        _, sample = get_sample(df, int(cfg["sample_size"]), int(cfg["sample_random_state"]))
        semantic_profile = "\n".join(s for s in [prof["structural_profile"], src["semantic_details"]] if s)
        block, stats = S.structured_block(prof["basic_profile"], prof["structural_profile"],
                                          src["semantic_details"], prof["factset"], "ufd")
        sfd_block, _ = S.structured_block(prof["basic_profile"], "", "", prof["factset"], "sfd",
                                          with_profile=False)
        inputs = {
            "orig": dict(profile=prof["basic_profile"], semantic=semantic_profile,
                         ufd_facts=src["ufd_facts"], sfd_facts=src["sfd_facts"]),
            "struct": dict(profile=block, semantic=S.SEMANTIC_POINTER,
                           ufd_facts=S.UFD_FACTS_POINTER, sfd_facts=sfd_block),
        }
        rec = {"dataset_id": i, "topic": src["topic"], "struct_stats": stats,
               "shared_calls": {"topic": src["calls"]["topic"], "semantic": src["calls"]["semantic"]},
               "lossless_ufd": S.check_lossless(prof["basic_profile"], prof["structural_profile"],
                                                src["semantic_details"], prof["factset"], block, "ufd"),
               "lossless_sfd": S.check_lossless(prof["basic_profile"], "", "", prof["factset"], sfd_block,
                                                "sfd", with_profile=False)}
        for arm in ARMS:
            x = inputs[arm]
            up, ufd = pipe.describe_dataset(
                dataset_sample=sample, dataset_profile=x["profile"], use_profile=True,
                semantic_profile=x["semantic"], use_semantic_profile=True, data_topic=src["topic"],
                use_topic=True, ufd_facts=x["ufd_facts"])
            ufd_calls = client.drain_log()
            sp, sfd = pipe.expand_description_for_search(ufd, src["topic"], sfd_facts=x["sfd_facts"])
            sfd_calls = client.drain_log()
            if arm == "orig":  # the original arm must be the F2/r1 run itself
                assert up == src["ufd_prompt"] and ufd == src["ufd"], i
                assert sp == src["sfd_prompt"] and sfd == src["sfd"], i
            rec[arm] = {"ufd_prompt": up, "ufd": ufd, "sfd_prompt": sp, "sfd": sfd,
                        "calls": {"ufd": ufd_calls, "sfd": sfd_calls}}
        out = CACHE_DIR / "gen/C05" / f"{i}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        hits = {a: [c["cache_hit"] for s in ("ufd", "sfd") for c in rec[a]["calls"][s]] for a in ARMS}
        print(i[:8], "lossless missing", rec["lossless_ufd"]["n_missing"], rec["lossless_sfd"]["n_missing"],
              "cache hits", hits, flush=True)


# --- analysis -------------------------------------------------------------------------

def _num(s: str) -> float:
    return float(s.replace(",", ""))


def evidence(prof: dict, src: dict, sample_df) -> dict[str, list[float]]:
    """column -> numbers in that column's evidence (profile, semantic, facts, sample values)."""
    order, _ = S.parse_basic(prof["basic_profile"])
    block, _ = S.structured_block(prof["basic_profile"], prof["structural_profile"],
                                  src["semantic_details"], prof["factset"], "ufd")
    blocks = S._blocks(block)
    ev = {}
    for col in order:
        nums = [_num(n) for n in cov.numbers_in(blocks.get(col, "").replace(f"[{col}]", ""))]
        if col in sample_df.columns:
            for v in sample_df[col].dropna().astype(str).unique()[:200]:
                nums += [_num(n) for n in cov.numbers_in(v)]
        ev[col] = nums
    ev["TABLE"] = [_num(n) for n in cov.numbers_in(blocks.get("TABLE", ""))]
    return ev


def mentions(sentence: str, cols: list[str]) -> set[str]:
    s = sentence.lower()
    out = set()
    for c in cols:
        for form in {c.lower(), c.lower().replace("_", " ")}:
            if len(form) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", s):
                out.add(c)
    return out


def audit_numbers(text: str, ev: dict[str, list[float]]) -> list[dict]:
    """Every number >= 32 or with decimals: which columns' evidence supports it, and which
    columns the sentence names. Candidate column confusion = sentence names columns, the
    number is supported only by other columns."""
    cols = [c for c in ev if c != "TABLE"]
    rows = []
    for sent in re.split(r"(?<=[.!?])\s+|\n", text):
        named = mentions(sent, cols)
        for n in cov.numbers_in(sent):
            if "." not in n and abs(float(n)) < 32:
                continue
            sup = sorted(c for c, nums in ev.items() if any(cov.number_matches(a, n) for a in nums))
            status = ("unsupported" if not sup else
                      "candidate_confusion" if named and not (set(sup) & (named | {"TABLE"})) else "ok")
            rows.append({"number": n, "supported_by": sup, "named_columns": sorted(named),
                         "status": status, "sentence": sent.strip()[:300]})
    return rows


def cost(calls: list[dict]) -> dict:
    i = sum((c["usage"] or {}).get("prompt_tokens", 0) for c in calls)
    o = sum((c["usage"] or {}).get("completion_tokens", 0) for c in calls)
    new = [c for c in calls if not c["cache_hit"]]
    ni = sum((c["usage"] or {}).get("prompt_tokens", 0) for c in new)
    no = sum((c["usage"] or {}).get("completion_tokens", 0) for c in new)
    return {"in": i, "out": o, "usd": i * PRICE["in"] + o * PRICE["out"],
            "new_usd": ni * PRICE["in"] + no * PRICE["out"]}


def analyse(args) -> None:
    cfg = yaml.safe_load((HERE / "configs/arm_F2.yaml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows, audits, sheet, key = [], {}, [], {}
    rng = random.Random(20261022)
    for i in ids():
        src, prof = load_source(i)
        rec = json.loads((CACHE_DIR / "gen/C05" / f"{i}.json").read_text())
        df, _ = load_table(TABLES_DIR / f"{i}.csv", cfg.get("read_max_rows"))
        sample_df, _ = get_sample(df, int(cfg["sample_size"]), int(cfg["sample_random_state"]))
        ev = evidence(prof, src, sample_df)
        facts = {f["fact_id"]: f for f in prof["factset"]["facts"]}
        shared = cost(rec["shared_calls"]["topic"] + rec["shared_calls"]["semantic"])
        row = {"dataset_id": i, "shared": shared, "lossless_ufd": rec["lossless_ufd"],
               "lossless_sfd": rec["lossless_sfd"], "struct_stats": rec["struct_stats"]}
        for arm in ARMS:
            r = rec[arm]
            kept_u = [cov.fact_kept(facts[k], r["ufd"])[0] for k in src["selected_ufd"]]
            kept_s = [cov.fact_kept(facts[k], r["sfd"])[0] for k in src["selected_sfd"]]
            au = {"ufd": audit_numbers(r["ufd"], ev), "sfd": audit_numbers(r["sfd"], ev)}
            audits[f"{i}:{arm}"] = au
            row[arm] = {
                "ufd_cost": cost(r["calls"]["ufd"]), "sfd_cost": cost(r["calls"]["sfd"]),
                "ufd_words": len(r["ufd"].split()), "sfd_words": len(r["sfd"].split()),
                "ufd_facts_kept": sum(kept_u), "sfd_facts_kept": sum(kept_s),
                "n_ufd_facts": len(kept_u), "n_sfd_facts": len(kept_s),
                **{f"{t}_{k}": sum(1 for x in au[t] if x["status"] == k)
                   for t in ("ufd", "sfd") for k in ("ok", "unsupported", "candidate_confusion")},
            }
        rows.append(row)
        order = list(ARMS)
        rng.shuffle(order)
        key[i] = {"A": order[0], "B": order[1]}
        sheet.append(f"## {i}\n\n### Profile (original form)\n\n```\n{prof['basic_profile']}\n"
                     f"{prof['structural_profile']}\n{src['ufd_facts']}\n```\n\n"
                     f"### Description A\n\n{rec[order[0]]['ufd']}\n\n### Description B\n\n{rec[order[1]]['ufd']}\n\n"
                     "| criterion | A | B |\n|---|---|---|\n"
                     "| column confusion (value attributed to the wrong column), count | | |\n"
                     "| statements not supported by the profile, count | | |\n"
                     "| key facts missing (coverage, units, measures) | | |\n"
                     "| readability / usefulness 1-5 | | |\n| preferred (A/B/tie) | | |\n")
    (OUT / "per_table.json").write_text(json.dumps(rows, indent=1))
    (OUT / "number_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=1))
    (OUT / "blind_sheet_ufd.md").write_text(
        "# C05 blind evaluation sheet (UFD)\n\nFill in without opening `blind_key.json`.\n\n" + "\n".join(sheet))
    (OUT / "blind_key.json").write_text(json.dumps(key, indent=1))
    for r in rows:
        o, s = r["orig"], r["struct"]
        print(r["dataset_id"][:8], "| ufd in", o["ufd_cost"]["in"], s["ufd_cost"]["in"],
              "| sfd in", o["sfd_cost"]["in"], s["sfd_cost"]["in"],
              "| kept u", f"{o['ufd_facts_kept']}/{o['n_ufd_facts']}", f"{s['ufd_facts_kept']}/{s['n_ufd_facts']}",
              "s", f"{o['sfd_facts_kept']}/{o['n_sfd_facts']}", f"{s['sfd_facts_kept']}/{s['n_sfd_facts']}",
              "| unsup u", o["ufd_unsupported"], s["ufd_unsupported"],
              "| conf u", o["ufd_candidate_confusion"], s["ufd_candidate_confusion"],
              "s", o["sfd_candidate_confusion"], s["sfd_candidate_confusion"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["generate", "analyse"], default="generate")
    ap.add_argument("--provider", choices=["none", "openai"], default="none")
    a = ap.parse_args()
    (generate if a.stage == "generate" else analyse)(a)
