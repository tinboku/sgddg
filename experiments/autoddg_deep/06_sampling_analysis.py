"""Programmatic checks for direction 3 (02_generate_sampling.py), no LLM judge.

Usage:
    python -I 06_sampling_analysis.py --runs r1,r2

Per strategy (orig / freq / dist) and run id:
  labels      per column, the semantic fields (entity type, domain type, usage context,
              isTemporal, isSpatial) that differ from orig in the same run; orig r1 vs
              orig r2 is the T=0 noise floor. Reported for all and for "complex"
              columns (heavy-tailed numeric, long-tailed categorical, mixed formats).
  tail        categorical columns with > 5 distinct values: share whose sample holds a
              value outside the column's top-5; tail values named in the semantic
              notes / UFD / SFD
  frequency   UFD/SFD sentences with a "most common / mostly / majority / predominantly"
              cue that name a category value: consistent if the value is its column's
              top-1, otherwise inconsistent
  faithful    05_fact_audit.unsupported (years / numbers absent from the call's prompt);
              omission of the important F1 facts (table_size, temporal_span,
              spatial_places, verified) via fact_coverage.fact_kept
  tokens      prompt / completion tokens per stage, new (non-cached) calls and cost at
              gpt-4o-mini list price

Output: results/sampling/s3_analysis.json and a summary on stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, HERE, RESULTS_DIR  # noqa: E402

from autoddg.description.fact_coverage import fact_kept  # noqa: E402

audit = __import__("05_fact_audit")

STRATEGIES = ("orig", "freq", "dist")
PRICE_IN, PRICE_OUT = 0.15e-6, 0.60e-6  # gpt-4o-mini $/token
IMPORTANT_KINDS = ("table_size", "temporal_span", "spatial_places")
FREQ_CUE = re.compile(r"\b(most common|most frequent|most often|mostly|majority|predominant\w*|primarily|"
                      r"the bulk of|dominated by|frequently)\b", re.I)


def norm(x) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip().lower()


def labels(desc: dict | None) -> dict:
    if not isinstance(desc, dict):
        return {k: None for k in ("entity", "domain", "usage", "temporal", "spatial")}
    t, s = desc.get("Temporal"), desc.get("Spatial")
    yes = lambda d, k: isinstance(d, dict) and str(d.get(k)).strip().lower() in ("true", "yes")  # noqa: E731
    return {"entity": norm(desc.get("Entity Type")), "domain": norm(desc.get("Domain-Specific Types")),
            "usage": norm(desc.get("Function/Usage Context")), "temporal": yes(t, "isTemporal"),
            "spatial": yes(s, "isSpatial")}


def load(strategy: str, run: str, i: str) -> dict | None:
    p = CACHE_DIR / "gen" / f"S3_{strategy}" / run / f"{i}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    return r if r.get("status") == "ok" else None


def mentions(text: str, value: str) -> bool:
    v = value.strip()
    if len(v) < 3 or re.fullmatch(r"[\d.,\-]+", v):
        return False  # short codes and bare numbers match too easily
    return re.search(r"(?<!\w)" + re.escape(v.lower()) + r"(?!\w)", text.lower()) is not None


def label_diff(a: dict, b: dict) -> dict:
    out = {"cols": 0, "any": 0, "complex_cols": 0, "complex_any": 0,
           "by_field": {k: 0 for k in ("entity", "domain", "usage", "temporal", "spatial")}}
    for c in a["columns"]:
        la, lb = labels(a["semantic_raw"].get(c)), labels(b["semantic_raw"].get(c))
        d = [k for k in la if la[k] != lb[k]]
        cx = a["column_stats"][c]["complex"]
        out["cols"] += 1
        out["complex_cols"] += cx
        out["any"] += bool(d)
        out["complex_any"] += bool(d) and cx
        for k in d:
            out["by_field"][k] += 1
    return out


def tail_check(r: dict) -> dict:
    out = {"cat_cols": 0, "tail_in_sample": 0, "tail_named_semantic": 0, "tail_named_ufd": 0,
           "tail_named_sfd": 0}
    for c in r["columns"]:
        st = r["column_stats"][c]
        if st["numeric"] or st["n_distinct"] <= 5 or st["n_distinct"] == st["n_nonblank"]:
            continue
        out["cat_cols"] += 1
        top5 = {v for v, _ in st["top"][:5]}
        vals = [re.sub(r" \(n=\d+\)$", "", v).strip() for v in r["samples"][c]]
        tail = [v for v in vals if v and v not in top5]
        out["tail_in_sample"] += bool(tail)
        for key, field in (("semantic_details", "tail_named_semantic"), ("ufd", "tail_named_ufd"),
                           ("sfd", "tail_named_sfd")):
            out[field] += any(mentions(r[key], v) for v in tail)
    return out


def freq_check(r: dict) -> dict:
    out = {"cue_sentences": 0, "with_value": 0, "consistent": 0, "inconsistent": [], }
    cats = {}
    for c in r["columns"]:
        st = r["column_stats"][c]
        if not st["numeric"] and st["top"] and st["n_distinct"] < st["n_nonblank"]:
            cats[c] = [v for v, _ in st["top"]]
    for field in ("ufd", "sfd"):
        for sent in re.split(r"(?<=[.!?])\s+", r[field] or ""):
            if not FREQ_CUE.search(sent):
                continue
            out["cue_sentences"] += 1
            hits = [(c, v, k) for c, top in cats.items() for k, v in enumerate(top) if mentions(sent, v)]
            if not hits:
                continue
            out["with_value"] += 1
            if any(k == 0 for _, _, k in hits):
                out["consistent"] += 1
            else:
                out["inconsistent"].append({"field": field, "sentence": sent[:200],
                                            "named": [(c, v, k + 1) for c, v, k in hits][:3]})
    return out


def important_facts(i: str) -> list[dict]:
    p = CACHE_DIR / "profile" / "F1" / f"{i}.json"
    if not p.exists():
        return []
    fs = json.loads(p.read_text(encoding="utf-8")).get("factset") or {}
    return [f for f in fs.get("facts", []) if f["kind"] in IMPORTANT_KINDS and f["status"] == "verified"
            and (f.get("render") or {}).get("ufd")]


def faithful(r: dict) -> dict:
    out = {}
    for t in ("ufd", "sfd"):
        u = audit.unsupported(r[t], r[f"{t}_prompt"])
        out[f"{t}_unsupported"] = len(u["unsupported_years"]) + len(u["unsupported_numbers"])
        out[f"{t}_numbers"] = u["n_years"] + u["n_numbers"]
    facts = important_facts(r["dataset_id"])
    out["important"] = len(facts)
    for t in ("ufd", "sfd"):
        out[f"{t}_important_missed"] = sum(not fact_kept(f, r[t])[0] for f in facts)
    return out


def tokens(r: dict) -> dict:
    out = {}
    for stage, calls in r["calls"].items():
        s = out.setdefault(stage, {"calls": 0, "new_calls": 0, "prompt": 0, "completion": 0,
                                   "new_prompt": 0, "new_completion": 0})
        for c in calls:
            u = c.get("usage") or {}
            p, q = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            s["calls"] += 1
            s["prompt"] += p
            s["completion"] += q
            if not c["cache_hit"]:
                s["new_calls"] += 1
                s["new_prompt"] += p
                s["new_completion"] += q
    return out


def add(acc: dict, x: dict) -> None:
    for k, v in x.items():
        if isinstance(v, dict):
            add(acc.setdefault(k, {}), v)
        elif isinstance(v, list):
            acc.setdefault(k, []).extend(v)
        else:
            acc[k] = acc.get(k, 0) + v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", default="r1,r2")
    ap.add_argument("--dataset-ids", type=Path, default=HERE / "configs" / "subset30.txt")
    args = ap.parse_args()
    runs = [x for x in args.runs.split(",") if x]
    ids = [l.strip() for l in args.dataset_ids.read_text(encoding="utf-8").splitlines() if l.strip()]

    recs = {(s, run, i): load(s, run, i) for s in STRATEGIES for run in runs for i in ids}
    usable = [i for i in ids if all(recs[(s, run, i)] for s in STRATEGIES for run in runs)]
    report = {"runs": runs, "n_tables": len(usable), "excluded": [i for i in ids if i not in usable],
              "per_arm": {}, "label_diff": {}, "strategy_mix": {}, "ufd_identical": {}}
    for s in STRATEGIES:
        for run in runs:
            acc: dict = {}
            for i in usable:
                r = recs[(s, run, i)]
                add(acc, {"tail": tail_check(r), "freq": freq_check(r), "faithful": faithful(r),
                          "tokens": tokens(r)})
            report["per_arm"][f"{s}/{run}"] = acc
            if s != "orig":
                mix: dict = {}
                for i in usable:
                    for v in recs[(s, run, i)]["column_strategy"].values():
                        mix[v] = mix.get(v, 0) + 1
                report["strategy_mix"][f"{s}/{run}"] = mix
    pairs = [(("orig", run), (s, run)) for run in runs for s in ("freq", "dist")]
    pairs += [((s, runs[0]), (s, run)) for s in STRATEGIES for run in runs[1:]]
    pairs += [(("freq", run), ("dist", run)) for run in runs]
    for a, b in pairs:
        acc = {}
        same_ufd = 0
        for i in usable:
            ra, rb = recs[(*a, i)], recs[(*b, i)]
            add(acc, label_diff(ra, rb))
            same_ufd += ra["ufd"] == rb["ufd"]
        key = f"{'/'.join(a)} vs {'/'.join(b)}"
        report["label_diff"][key] = acc
        report["ufd_identical"][key] = same_ufd
    out = RESULTS_DIR / "sampling" / "s3_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"tables={len(usable)} excluded={report['excluded']}")
    for k, d in report["label_diff"].items():
        print(f"labels {k}: changed {d['any']}/{d['cols']} cols ({d['any'] / d['cols']:.1%}); complex "
              f"{d['complex_any']}/{d['complex_cols']}; by field {d['by_field']}; UFD identical "
              f"{report['ufd_identical'][k]}/{len(usable)}")
    for k, a in report["per_arm"].items():
        t, f, fa = a["tail"], a["freq"], a["faithful"]
        tok = a["tokens"]
        new_cost = sum(v["new_prompt"] * PRICE_IN + v["new_completion"] * PRICE_OUT for v in tok.values())
        sem = tok.get("semantic", {})
        print(f"[{k}] tail-in-sample {t['tail_in_sample']}/{t['cat_cols']} named sem/ufd/sfd "
              f"{t['tail_named_semantic']}/{t['tail_named_ufd']}/{t['tail_named_sfd']} | freq cue "
              f"{f['cue_sentences']} w/value {f['with_value']} consistent {f['consistent']} "
              f"inconsistent {len(f['inconsistent'])} | unsupported UFD {fa['ufd_unsupported']}/{fa['ufd_numbers']} "
              f"SFD {fa['sfd_unsupported']}/{fa['sfd_numbers']} | important missed UFD "
              f"{fa['ufd_important_missed']}/{fa['important']} SFD {fa['sfd_important_missed']}/{fa['important']} | "
              f"semantic tokens {sem.get('prompt', 0)}+{sem.get('completion', 0)} | new cost ${new_cost:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
