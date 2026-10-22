"""Listwise LLM rerank of each query's judged pool: title+SFD vs title+SFD+fact card.

Usage:
    python -I 09_llm_rerank.py --provider openai --max-cost 0.5
    python -I 09_llm_rerank.py --provider none          # cache only

One call per (query, condition). The pool is the query's judged CSV datasets
(per_query protocol), shown in the bugfix single-field BM25 (raw tokenizer) order
under neutral labels D01..Dnn; the order and labels are the same in every
condition, only the subset30 document texts differ:
    ctrl     title + bugfix SFD                (other datasets: title + original description)
    facts    title + bugfix SFD + "Verified facts:" + F2 sfd_facts
    ctrl run r2 is a second, independent call of ctrl for the noise floor.
gpt-4o-mini, temperature 0, JSON output {"ranking": [...]}. Labels missing from the
answer are appended in input order; duplicates/unknown labels are dropped and counted.
Cost is estimated before the run (chars/4) and the run stops when the recorded
usage would exceed --max-cost.

Writes results/eval/rerank.json and results/paired/rerank.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CACHE_DIR, MANIFEST, RESULTS_DIR, read_jsonl, select_benchmark  # noqa: E402
from llm_cache import CachingClient, make_inner_client  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ORACLE = _load("07_oracle_tokenizer")
EVAL, PAIRED = ORACLE.EVAL, ORACLE.PAIRED
MODEL = "gpt-4o-mini"
PRICE_IN, PRICE_OUT = 0.15 / 1e6, 0.60 / 1e6
PROMPT = """You are ranking datasets for a dataset search engine.

Query: {query}

Candidate datasets:
{docs}

Rank ALL {n} candidate datasets from most to least relevant to the query.
Return only JSON: {{"ranking": ["D01", "D02", ...]}} containing every label exactly once."""


def build_docs(corpus, listed, sfd_arm, facts_arm):
    sfd = {r["dataset_id"]: r for r in read_jsonl(RESULTS_DIR / "descriptions" / f"{sfd_arm}.jsonl")}
    facts = {r["dataset_id"]: r for r in read_jsonl(RESULTS_DIR / "descriptions" / f"{facts_arm}.jsonl")}
    ctrl, exp = {}, {}
    for d, e in corpus.items():
        title = e.get("title") or ""
        row = sfd.get(d, {}) if d in listed else {}
        body = row["sfd"] if row.get("status") == "ok" and row.get("sfd") else (e.get("description") or "")
        ctrl[d] = f"Title: {title}\n{body}"
        card = facts.get(d, {}).get("sfd_facts") if d in listed and facts.get(d, {}).get("status") == "ok" else ""
        exp[d] = f"{ctrl[d]}\nVerified facts:\n{card}" if card else ctrl[d]
    return {"ctrl": ctrl, "facts": exp}


def parse_ranking(content: str, labels: list[str]) -> tuple[list[str], dict]:
    found = re.findall(r"D\d{2,3}", content or "")
    seen, order, dup, unknown = set(), [], 0, 0
    for lab in found:
        if lab not in labels:
            unknown += 1
        elif lab in seen:
            dup += 1
        else:
            seen.add(lab)
            order.append(lab)
    missing = [lab for lab in labels if lab not in seen]
    return order + missing, {"missing": len(missing), "duplicates": dup, "unknown": unknown}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--provider", default="openai", choices=["openai", "none"])
    ap.add_argument("--max-cost", type=float, default=0.5)
    ap.add_argument("--sfd-arm", default="bugfix__r1__subset30")
    ap.add_argument("--facts-arm", default="F2__r1__subset30")
    ap.add_argument("--dataset-ids", type=Path, default=HERE / "configs" / "subset30.txt")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    queries, qrels, corpus = select_benchmark(csv_only=True)
    manifest = {r["dataset_id"]: r for r in read_jsonl(MANIFEST)}
    with_table = {d for d in corpus if manifest.get(d, {}).get("status") in {"ok", "truncated"}}
    listed = {l.strip() for l in args.dataset_ids.read_text(encoding="utf-8").splitlines() if l.strip()} & set(corpus)
    subsets = {"all": set(corpus), "with_table": with_table, "listed": listed}
    qids = sorted(q for q in queries if any(r > 0 for r in qrels[q].values()))
    texts = build_docs(corpus, listed, args.sfd_arm, args.facts_arm)
    bm25_docs = EVAL.arm_documents(args.sfd_arm, "sfd", corpus)[0]

    jobs = []  # (condition, run_id, qid, labels->ids, prompt)
    for q in qids:
        order = ORACLE.rank(ORACLE.tok_raw, queries[q], sorted(qrels[q]), bm25_docs)
        labels = {f"D{i + 1:02d}": d for i, d in enumerate(order)}
        for cond, run_id in (("ctrl", "r1"), ("facts", "r1"), ("ctrl", "r2")):
            docs = "\n\n".join(f"[{lab}]\n{texts[cond][d]}" for lab, d in labels.items())
            jobs.append((cond, run_id, q, labels, PROMPT.format(query=queries[q], docs=docs, n=len(labels))))
    est_in = sum(len(j[4]) for j in jobs) / 4
    est_cost = est_in * PRICE_IN + len(jobs) * 250 * PRICE_OUT
    print(f"jobs={len(jobs)} est_input_tokens={est_in:.0f} est_cost=${est_cost:.3f} (cap ${args.max_cost})")
    if est_cost > args.max_cost:
        raise SystemExit("estimated cost exceeds cap; not running")

    inner = make_inner_client(args.provider)
    clients = {r: CachingClient(inner, CACHE_DIR / "llm", f"rerank_{r}", args.provider if inner else "openai",
                                offline=inner is None) for r in ("r1", "r2")}
    spent = {"input_tokens": 0, "output_tokens": 0, "calls_live": 0, "calls_cached": 0}

    def run(job):
        cond, run_id, q, labels, prompt = job
        client = clients[run_id]
        resp = client.chat_completions_create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0,
            response_format={"type": "json_object"}, max_tokens=600)
        content = resp["choices"][0]["message"]["content"]
        ranking, issues = parse_ranking(content, list(labels))
        return job, [labels[lab] for lab in ranking], issues

    results = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for job, ranked, issues in pool.map(run, jobs):
            results[(job[0], job[1], job[2])] = (ranked, issues)
    for client in clients.values():
        for entry in client.drain_log():
            u = entry.get("usage") or {}
            if entry["cache_hit"]:
                spent["calls_cached"] += 1
            else:
                spent["calls_live"] += 1
                spent["input_tokens"] += u.get("prompt_tokens", 0)
                spent["output_tokens"] += u.get("completion_tokens", 0)
    spent["cost_usd"] = spent["input_tokens"] * PRICE_IN + spent["output_tokens"] * PRICE_OUT
    print("spent", spent)

    per_query, issues_out = {}, {}
    for (cond, run_id, q), (ranked, issues) in results.items():
        issues_out[f"{cond}|{run_id}|{q}"] = issues
        for sname, allowed in subsets.items():
            judgments = {d: r for d, r in qrels[q].items() if d in allowed}
            if not any(r > 0 for r in judgments.values()):
                continue
            sub = [d for d in ranked if d in allowed]
            per_query.setdefault(f"{cond}_{run_id}|{sname}", {})[q] = {
                f"ndcg@{k}": EVAL.ndcg_at_k(sub, judgments, k) for k in (10, 20)}

    ref = json.loads((RESULTS_DIR / "eval" / "oracle_tok.json").read_text(encoding="utf-8"))["per_query"]
    cmp_args = argparse.Namespace(bootstrap=10000, n_perm=10000)
    rows = []
    for sname in subsets:
        for a_key, b_key in ((f"ctrl_r1|{sname}", f"facts_r1|{sname}"), (f"ctrl_r1|{sname}", f"ctrl_r2|{sname}"),
                             (f"bugfix__r1__subset30|raw|{sname}", f"ctrl_r1|{sname}")):
            sa = ref.get(a_key) or per_query[a_key]
            for metric in ("ndcg@10", "ndcg@20"):
                res = PAIRED.compare(sa, per_query[b_key], metric, cmp_args, np.random.default_rng(args.seed))
                if res["n_queries"] == 0:
                    continue
                res.update(a=a_key, b=b_key, metric=metric)
                rows.append(res)
                print(f"{b_key:22s} - {a_key:36s} {metric} n={res['n_queries']:2d} A={res['mean_a']:.4f} "
                      f"B={res['mean_b']:.4f} d={res['delta']:+.4f} W/T/L={res['wins']}/{res['ties']}/{res['losses']} "
                      f"p={res['p_randomization']:.4f} CI=[{res['ci95_low']:+.4f},{res['ci95_high']:+.4f}]")
    bad = {k: v for k, v in issues_out.items() if any(v.values())}
    print(f"parse issues in {len(bad)}/{len(issues_out)} answers")

    (RESULTS_DIR / "eval" / "rerank.json").write_text(json.dumps(
        {"settings": {"model": MODEL, "temperature": 0.0, "input_order": "bugfix BM25 raw", "estimate_usd": est_cost},
         "spent": spent, "parse_issues": issues_out, "per_query": per_query}, indent=1), encoding="utf-8")
    (RESULTS_DIR / "paired" / "rerank.json").write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8")
    print("wrote results/eval/rerank.json results/paired/rerank.json")


if __name__ == "__main__":
    main()
