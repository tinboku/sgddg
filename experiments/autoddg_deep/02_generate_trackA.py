"""Track A pilot runner: four arms, frozen held-out tables, two repeats.

This is intentionally small and reproducible; all mutable outputs live under
``cache/trackA`` and ``results/trackA``.  Use ``--dry-run`` for zero API calls.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import importlib.util
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO / "external/AutoDDG/src"))
sys.path.insert(0, str(ROOT))
from tables import load_table
_tr = importlib.util.spec_from_file_location("track_a_table_reader", REPO/"research/track_a/pilot/table_reader.py")
_tm = importlib.util.module_from_spec(_tr); _tr.loader.exec_module(_tm)
load_table_fixed = _tm.load_table_fixed
from autoddg.autoddg_deep import AutoDDGDeep
from autoddg.profiling_v2.pipeline import build_factset, fact_texts
from autoddg.profiling_v2.select import SelectConfig

ARMS = ("B0", "C", "Cprime", "A")

def ids():
    return [x.strip() for x in (REPO/"research/track_a/heldout_ids.txt").read_text().splitlines() if x.strip()]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arm", action="append", choices=ARMS)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    selected = args.arm or list(ARMS)
    table_ids = ids()[:args.limit]
    out = ROOT/"results"/"trackA"; out.mkdir(parents=True, exist_ok=True)
    cache = ROOT/"cache"/"trackA"; cache.mkdir(parents=True, exist_ok=True)
    manifest = {json.loads(x)["dataset_id"]: json.loads(x) for x in (ROOT/"data/manifest.jsonl").read_text().splitlines()}
    if args.dry_run:
        print(json.dumps({"dry_run": True, "arms": selected, "tables": len(table_ids), "repeats": 2,
                          "api_calls": 0, "heldout_sha256": sha(REPO/"research/track_a/heldout_ids.txt")}, ensure_ascii=False))
        return
    from openai import OpenAI
    client = OpenAI()
    for arm in selected:
        cfg = yaml.safe_load((ROOT/f"configs/arm_{arm}.yaml").read_text())
        fact_cfg = cfg.get("facts", {})
        mode = fact_cfg.get("mode", "none")
        scfg = SelectConfig.from_dict(fact_cfg.get("select", {})) if mode != "none" else None
        import os
        for repeat in (("r1",) if os.environ.get("REPEATS1") else ("r1", "r2")):
            records=[]
            for did in table_ids:
                p = ROOT/"data/tables"/f"{did}.csv"
                if not p.exists(): records.append({"dataset_id":did,"status":"no_table"}); continue
                reader = load_table_fixed if arm == "B0" else load_table
                df, read_info = reader(p, cfg.get("read_max_rows"))
                # profile cache: profile_dataframe depends only on (df, reader); reuse across arms/repeats
                gen = AutoDDGDeep(client=client, model_name=cfg["model"], description_words=cfg["description_words"],
                                  description_temperature=cfg["description_temperature"], topic_temperature=cfg["topic_temperature"],
                                  sfd_temperature=cfg["sfd_temperature"], fact_mode=mode)
                pcache = cache/"profiles"; pcache.mkdir(parents=True, exist_ok=True)
                reader_cls = "B0" if arm == "B0" else "orig"
                pkey = pcache/f"{reader_cls}_{did}.json"
                if pkey.exists():
                    _pc = json.loads(pkey.read_text())
                    profile, semantic = _pc["profile"], _pc["semantic"]
                else:
                    # cap profiling rows for speed (generation sample + factset still use full df)
                    df_prof = df if len(df) <= 50000 else df.sample(50000, random_state=9)
                    profile, semantic = gen.profile_dataframe(df_prof)
                    pkey.write_text(json.dumps({"profile": profile, "semantic": semantic}, ensure_ascii=False, default=str))
                sample = df.sample(min(len(df), cfg.get("sample_size",100)), random_state=cfg.get("sample_random_state",9)).to_csv(index=False)
                facts = fact_texts(build_factset(df, did, scfg, table=read_info)) if scfg else {"ufd_facts":None,"sfd_facts":None}
                mrow = manifest.get(did, {})
                topic = gen.generate_topic(mrow.get("title", did), mrow.get("description"), sample)
                ufd_prompt, ufd = gen.describe_dataset(sample, profile, True, semantic, True, topic, True, facts["ufd_facts"])
                sfd_prompt, sfd = gen.expand_description_for_search(ufd, topic, facts["sfd_facts"])
                records.append({"dataset_id":did,"status":"ok","arm":arm,"repeat":repeat,"read":read_info,
                                "ufd":ufd,"sfd":sfd,"prompts":{"ufd":ufd_prompt,"sfd":sfd_prompt},
                                "facts":facts,"table_sha256":sha(p)})
            (out/f"{arm}__{repeat}.json").write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print(json.dumps({"completed":True,"arms":selected,"tables":len(table_ids),"repeats":2},ensure_ascii=False))

if __name__ == "__main__": main()
