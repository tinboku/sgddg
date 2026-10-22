"""C05 position-aligned control: same slots, same slot wording, only the representation
of the column evidence inside each slot changes (sentences -> key=value grouped by column).

    orig  = F2/r1 (from cache/gen/C05, unchanged)
    pos   = basic-profile slot, semantic slot and fact slot each keep their original
            position and surrounding instructions; their column lines are rewritten by
            c05_structured.aligned_* (no pointers, no header, no source keys, no VALUES).
The SFD fact slot uses the same aligned fact text. A check confirms that removing the three
slot contents leaves the orig and pos prompts byte-identical.

    python -I c05_pos_run.py --provider openai     # generation (pos arm = new calls)
    python -I c05_pos_run.py --stage analyse       # results/c05_pos/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c05_run as R  # noqa: E402
import c05_structured as S  # noqa: E402
from common import CACHE_DIR, HERE, RESULTS_DIR, TABLES_DIR  # noqa: E402
from llm_cache import CachingClient, make_inner_client  # noqa: E402
from tables import load_table  # noqa: E402
from autoddg.description import fact_coverage as cov  # noqa: E402
from autoddg.utils import get_sample  # noqa: E402

OUT = RESULTS_DIR / "c05_pos"
GEN = CACHE_DIR / "gen/C05pos"
# meta talk about the input itself (not about the data); fixed before reading the outputs
META = re.compile(r"\b(profile[sd]?|key[- ]value|structured (?:data|profile|format)|entries|"
                  r"semantic information|the (?:facts|information) (?:provided|above|given))\b", re.I)


def slots(prof: dict, src: dict) -> dict:
    order, _ = S.parse_basic(prof["basic_profile"])
    return dict(
        profile=S.aligned_profile(prof["basic_profile"]),
        semantic=S.aligned_semantic(prof["basic_profile"], prof["structural_profile"], src["semantic_details"]),
        ufd_facts=S.aligned_facts(prof["factset"], "ufd", order),
        sfd_facts=S.aligned_facts(prof["factset"], "sfd", order))


def skeleton(prompt: str, parts: list[str]) -> str:
    for p in parts:
        assert prompt.count(p) == 1, p[:80]
        prompt = prompt.replace(p, "<SLOT>")
    return prompt


def generate(args) -> None:
    cfg = yaml.safe_load((HERE / "configs/arm_F2.yaml").read_text())
    inner = None if args.provider == "none" else make_inner_client("openai")
    GEN.mkdir(parents=True, exist_ok=True)
    for i in R.ids():
        src, prof = R.load_source(i)
        old = json.loads((CACHE_DIR / "gen/C05" / f"{i}.json").read_text())
        client = CachingClient(inner, CACHE_DIR / "llm", run_id="r1", provider="openai",
                               offline=args.provider == "none")
        pipe = R.gen02.build_pipe(cfg, client)
        df, _ = load_table(TABLES_DIR / f"{i}.csv", cfg.get("read_max_rows"))
        _, sample = R.get_sample(df, int(cfg["sample_size"]), int(cfg["sample_random_state"]))
        x = slots(prof, src)
        semantic_orig = "\n".join(s for s in [prof["structural_profile"], src["semantic_details"]] if s)
        # lossless: the three UFD slots together carry every original atom of each column
        combined = "\n".join([x["profile"], x["semantic"], x["ufd_facts"]])
        loss_u = S.check_lossless(prof["basic_profile"], prof["structural_profile"], src["semantic_details"],
                                  prof["factset"], combined, "ufd")
        loss_s = S.check_lossless(prof["basic_profile"], "", "", prof["factset"], x["sfd_facts"], "sfd",
                                  with_profile=False)
        if args.dry:
            up = old["orig"]["ufd_prompt"]
            print(i[:8], "lossless", loss_u["n_missing"], loss_s["n_missing"])
            if i == R.ids()[args.show]:
                print(up.replace(prof["basic_profile"], x["profile"]).replace(semantic_orig, x["semantic"])
                      .replace(src["ufd_facts"], x["ufd_facts"])[-3500:])
            continue
        up, ufd = pipe.describe_dataset(
            dataset_sample=sample, dataset_profile=x["profile"], use_profile=True,
            semantic_profile=x["semantic"], use_semantic_profile=True, data_topic=src["topic"],
            use_topic=True, ufd_facts=x["ufd_facts"])
        ufd_calls = client.drain_log()
        sp, sfd = pipe.expand_description_for_search(ufd, src["topic"], sfd_facts=x["sfd_facts"])
        sfd_calls = client.drain_log()
        # structure check: with the slot contents cut out, both prompts are the same bytes
        o = old["orig"]
        assert skeleton(up, [x["profile"], x["semantic"], x["ufd_facts"]]) == \
            skeleton(o["ufd_prompt"], [prof["basic_profile"], semantic_orig, src["ufd_facts"]]), i
        assert skeleton(sp, [ufd, x["sfd_facts"]]) == skeleton(o["sfd_prompt"], [o["ufd"], src["sfd_facts"]]), i
        rec = {"dataset_id": i, "slots": x, "lossless_ufd": loss_u, "lossless_sfd": loss_s,
               "skeleton_identical": True,
               "pos": {"ufd_prompt": up, "ufd": ufd, "sfd_prompt": sp, "sfd": sfd,
                       "calls": {"ufd": ufd_calls, "sfd": sfd_calls}}}
        (GEN / f"{i}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        print(i[:8], "lossless", loss_u["n_missing"], loss_s["n_missing"], "cache hits",
              [c["cache_hit"] for s in ("ufd", "sfd") for c in rec["pos"]["calls"][s]], flush=True)


def analyse(args) -> None:
    cfg = yaml.safe_load((HERE / "configs/arm_F2.yaml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows, audits = [], {}
    for i in R.ids():
        src, prof = R.load_source(i)
        old = json.loads((CACHE_DIR / "gen/C05" / f"{i}.json").read_text())
        new = json.loads((GEN / f"{i}.json").read_text())
        arms = {"orig": old["orig"], "struct": old["struct"], "pos": new["pos"]}
        df, _ = load_table(TABLES_DIR / f"{i}.csv", cfg.get("read_max_rows"))
        sample_df, _ = get_sample(df, int(cfg["sample_size"]), int(cfg["sample_random_state"]))
        ev = R.evidence(prof, src, sample_df)
        facts = {f["fact_id"]: f for f in prof["factset"]["facts"]}
        row = {"dataset_id": i, "lossless_ufd": new["lossless_ufd"]["n_missing"],
               "lossless_ufd_atoms": new["lossless_ufd"]["n_atoms"],
               "lossless_sfd": new["lossless_sfd"]["n_missing"],
               "lossless_sfd_atoms": new["lossless_sfd"]["n_atoms"]}
        for arm, r in arms.items():
            au = {"ufd": R.audit_numbers(r["ufd"], ev), "sfd": R.audit_numbers(r["sfd"], ev)}
            audits[f"{i}:{arm}"] = au
            row[arm] = {
                "ufd_cost": R.cost(r["calls"]["ufd"]), "sfd_cost": R.cost(r["calls"]["sfd"]),
                "ufd_facts_kept": sum(cov.fact_kept(facts[k], r["ufd"])[0] for k in src["selected_ufd"]),
                "sfd_facts_kept": sum(cov.fact_kept(facts[k], r["sfd"])[0] for k in src["selected_sfd"]),
                "n_ufd_facts": len(src["selected_ufd"]), "n_sfd_facts": len(src["selected_sfd"]),
                "ufd_meta_hits": sorted({m.group(0).lower() for m in META.finditer(r["ufd"])}),
                **{f"{t}_{k}": sum(1 for x in au[t] if x["status"] == k)
                   for t in ("ufd", "sfd") for k in ("unsupported", "candidate_confusion")},
            }
        rows.append(row)
    (OUT / "per_table.json").write_text(json.dumps(rows, indent=1))
    (OUT / "number_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=1))
    tot = {a: {k: 0 for k in ("ui", "si", "uk", "sk", "nu", "ns", "uu", "uc", "meta", "usd", "new")}
           for a in ("orig", "struct", "pos")}
    for r in rows:
        line = [r["dataset_id"][:8], f"loss {r['lossless_ufd']}/{r['lossless_ufd_atoms']}"]
        for a in tot:
            x, t = r[a], tot[a]
            t["ui"] += x["ufd_cost"]["in"]; t["si"] += x["sfd_cost"]["in"]
            t["uk"] += x["ufd_facts_kept"]; t["sk"] += x["sfd_facts_kept"]
            t["nu"] += x["n_ufd_facts"]; t["ns"] += x["n_sfd_facts"]
            t["uu"] += x["ufd_unsupported"]; t["uc"] += x["ufd_candidate_confusion"]
            t["meta"] += bool(x["ufd_meta_hits"])
            t["usd"] += x["ufd_cost"]["usd"] + x["sfd_cost"]["usd"]
            t["new"] += x["ufd_cost"]["new_usd"] + x["sfd_cost"]["new_usd"]
            line.append(f"{a}: ui {x['ufd_cost']['in']} si {x['sfd_cost']['in']} "
                        f"ku {x['ufd_facts_kept']}/{x['n_ufd_facts']} ks {x['sfd_facts_kept']}/{x['n_sfd_facts']} "
                        f"meta {x['ufd_meta_hits']}")
        print(" | ".join(line))
    print(json.dumps(tot, indent=1))
    (OUT / "totals.json").write_text(json.dumps(tot, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["generate", "analyse"], default="generate")
    ap.add_argument("--provider", choices=["none", "openai"], default="none")
    ap.add_argument("--dry", action="store_true", help="build slots and check lossless only, no LLM")
    ap.add_argument("--show", type=int, default=9)
    a = ap.parse_args()
    (generate if a.stage == "generate" else analyse)(a)
