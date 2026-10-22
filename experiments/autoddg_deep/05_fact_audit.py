"""Fact audit for the F1 / F2 arms: retention, unsupported numbers, budget crowding.

Usage:
    python -I 05_fact_audit.py --arms F1,F2 --run-id r1 --dataset-ids configs/smoke3.txt
    python -I 05_fact_audit.py --arms F1,F2,F2c --run-id r1 --dataset-ids configs/subset30.txt

Reads cache/gen/<arm>/<run_id>/<id>.json and the profile cache it points to
(``profile_cache``), and reports per table and per arm:

retention   for every fact in the common selection, whether the UFD (all
            selected facts) and the SFD (the verified / inferred subset) still
            state it. A fact's anchors are the values its rendered text names
            (span endpoints, places / category values, range bounds, units,
            interval). Writing variants count as kept: "2023-Q1" = "2023 Q1" =
            "Q1 2023", "2016-06-17" = "June 17, 2016", "161.1" = "161" (a
            rounded form of the value). Scalar facts need every anchor; list
            facts (places, categories) need min(2, n) of their shown values.
unsupported years (1800-2100) and other numbers in the UFD / SFD that do not
            occur in that call's own prompt (sample, profile, facts, topic,
            UFD for the SFD): the generator had no input for them.
crowding    promptable facts left out of the common selection by the token
            budget or a per-kind cap, with their kinds and priorities.
identity    with several arms: per table, whether topic, semantic notes, UFD
            prompt, UFD text and selected fact ids are identical (E3 requires
            this for F1 vs F2).

Output: results/audit/<arms>__<run_id>[__<stem>].json and a summary on stdout.
Retention measures literal restatement only; whether a restated fact is used
correctly still needs the manual check of the selected facts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, HERE, RESULTS_DIR  # noqa: E402

LIST_KINDS = ("spatial_places", "category_values")
MAX_LIST_ANCHORS = 5  # render.UFD_TOP_VALUES
INTERVAL_WORDS = {
    "year": ["year", "annual"], "half-year": ["half year", "semiannual", "semi annual"],
    "quarter": ["quarter"], "month": ["month"], "week": ["week"], "day": ["day", "daily"],
    "hour": ["hour"], "minute": ["minute"],
}
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")
_YEAR = re.compile(r"(?<!\d)(1[89]\d\d|20\d\d|2100)(?!\d)")


# --- normalisation ---------------------------------------------------------------

def norm(text: str) -> str:
    """Lower case; thousands commas dropped; - _ / , ; : ( ) as spaces; one space."""
    t = str(text).lower()
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = re.sub(r"[-_/,;:()–—]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _contains(text_n: str, phrase: str) -> bool:
    p = norm(phrase)
    return bool(p) and re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", text_n) is not None


def time_variants(value: str) -> list[str]:
    """Writings of a span endpoint that count as the same value."""
    v = str(value)
    out = [v]
    m = re.fullmatch(r"(\d{4})-Q([1-4])", v)
    if m:
        y, q = m.groups()
        ordinal = ["first", "second", "third", "fourth"][int(q) - 1]
        return out + [f"{y} Q{q}", f"Q{q} {y}", f"{ordinal} quarter of {y}"]
    if re.fullmatch(r"\d{4}", v):
        return out
    try:
        ts = pd.Timestamp(v)
    except (ValueError, TypeError):
        return out
    month, mon = ts.strftime("%B"), ts.strftime("%b")
    if re.fullmatch(r"\d{4}-\d{2}", v):
        return out + [f"{month} {ts.year}", f"{mon} {ts.year}"]
    day = f"{ts.day}"
    out += [f"{month} {day} {ts.year}", f"{mon} {day} {ts.year}", f"{day} {month} {ts.year}",
            f"{ts.month}/{ts.day}/{ts.year}"]
    return out


def numbers_in(text: str) -> list[str]:
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(text))
    return _NUM.findall(t)


def number_matches(anchor: float, found: str) -> bool:
    """``found`` is ``anchor`` or a rounded writing of it."""
    x = float(found)
    decimals = len(found.split(".")[1]) if "." in found else 0
    if decimals == 0 and not float(anchor).is_integer() and abs(anchor) < 10:
        return False  # "2" is too loose a writing of 1.81
    return abs(x - anchor) <= 0.5 * 10 ** -decimals + 1e-9


def has_number(text: str, anchor: float) -> bool:
    return any(number_matches(anchor, n) for n in numbers_in(text))


# --- anchors ---------------------------------------------------------------------

def anchors(fact: dict) -> list[dict]:
    """[{"kind": "text"|"number", "alts" | "value"}] the fact's rendered text names."""
    k, v = fact["kind"], fact["value"]
    out: list[dict] = []

    def text(*alts):
        out.append({"kind": "text", "alts": [a for a in alts if a]})

    def num(x):
        if x is not None:
            out.append({"kind": "number", "value": float(x)})

    if k == "temporal_span":
        for end in (v["start"], v["end"]):
            text(*time_variants(end))
    elif k == "temporal_resolution" and v.get("basis") == "observed_interval":
        text(*INTERVAL_WORDS.get(v.get("interval"), [v.get("interval")]))
    elif k == "spatial_places":
        if v.get("confirmed"):
            for p in v["places"][:MAX_LIST_ANCHORS]:
                text(p["value"])
        else:
            text(v["level"])
    elif k == "spatial_level":
        text(v["level"])
    elif k == "category_values":
        for t in v["top"][:MAX_LIST_ANCHORS]:
            text(t["value"])
    elif k == "measure_range":
        if v.get("trimmed"):
            num(v.get("trimmed_min")), num(v.get("trimmed_max"))
        elif v.get("mixed_units"):
            for g in v["by_unit"]:
                num(g["min"]), num(g["max"])
        else:
            num(v.get("min")), num(v.get("max"))
    elif k == "measure_unit":
        if v.get("basis") == "unit_column":
            for u in v["units"][:MAX_LIST_ANCHORS]:
                text(u["value"])
        elif v.get("unit") == "mixed":
            for u in v.get("units", []):
                text(u["value"])
        else:
            text(v["unit"])
    elif k == "table_size":
        num(v.get("n_rows"))
    elif k == "missingness":
        num(round(100 * v["null_rate"]))
    return out


def anchor_found(a: dict, text: str, text_n: str) -> bool:
    if a["kind"] == "number":
        return has_number(text, a["value"])
    for alt in a["alts"]:
        alt = str(alt)
        if alt and not alt[0].isalnum() and alt in text:  # "$", "%"
            return True
        if _contains(text_n, alt):
            return True
    return False


def fact_kept(fact: dict, text: str) -> tuple[bool, int, int]:
    """(kept, anchors found, anchors)."""
    anc = anchors(fact)
    if not anc:
        return True, 0, 0
    text_n = norm(text)
    hits = sum(anchor_found(a, text, text_n) for a in anc)
    need = min(2, len(anc)) if fact["kind"] in LIST_KINDS else len(anc)
    return hits >= need, hits, len(anc)


# --- unsupported numbers -----------------------------------------------------------

def unsupported(output: str, prompt: str) -> dict:
    """Years and other numbers in ``output`` that do not occur in ``prompt``."""
    prompt_nums = numbers_in(prompt)
    years = sorted(set(_YEAR.findall(output)))
    bad_years = [y for y in years if y not in prompt_nums]
    others = sorted({n for n in numbers_in(output) if n not in years and len(n.lstrip("-").replace(".", "")) >= 2})
    bad_nums = [n for n in others if not any(number_matches(float(p), n) or p == n for p in prompt_nums)]
    return {"n_years": len(years), "unsupported_years": bad_years,
            "n_numbers": len(others), "unsupported_numbers": bad_nums}


# --- per table -----------------------------------------------------------------------

def load_gen(arm: str, run_id: str, dataset_id: str) -> dict | None:
    p = CACHE_DIR / "gen" / arm / run_id / f"{dataset_id}.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    return rec if rec.get("status") == "ok" and rec.get("ufd") is not None else None


def load_factset(rec: dict) -> dict | None:
    rel = rec.get("profile_cache")
    if not rel:
        return None
    prof = json.loads((HERE / rel).read_text(encoding="utf-8"))
    if prof.get("profile_key") != rec.get("profile_key"):
        raise SystemExit(f"{rec['dataset_id']}: profile cache changed since generation; regenerate first")
    return prof.get("factset")


def crowding(fs: dict) -> dict:
    by_id = {f["fact_id"]: f for f in fs["facts"]}
    b = fs.get("budget") or {}
    out = {"used_tokens": b.get("used_ufd_tokens"), "max_tokens": b.get("ufd_max_tokens"),
           "n_selected": len(b.get("selected_ufd", [])), "dropped": []}
    for d in b.get("dropped", []):
        if d["reason"] in ("budget", "kind_cap"):
            f = by_id[d["id"]]
            out["dropped"].append({"id": d["id"], "reason": d["reason"], "kind": f["kind"],
                                   "priority": f["priority"], "status": f["status"]})
    out["n_dropped"] = len(out["dropped"])
    n_cand = out["n_selected"] + out["n_dropped"]
    out["crowded_share"] = out["n_dropped"] / n_cand if n_cand else 0.0
    return out


def audit_table(rec: dict, fs: dict) -> dict:
    by_id = {f["fact_id"]: f for f in fs["facts"]}
    b = fs.get("budget") or {}
    res = {"dataset_id": rec["dataset_id"], "arm": rec["arm"]}
    for target in ("ufd", "sfd"):
        ids = b.get(f"selected_{target}", [])
        rows = []
        for i in ids:
            kept, hits, n = fact_kept(by_id[i], rec[target])
            rows.append({"id": i, "kind": by_id[i]["kind"], "kept": kept, "anchors_found": hits, "anchors": n,
                         "text": (by_id[i].get("render") or {}).get("ufd")})
        res[f"{target}_retention"] = {
            "n_facts": len(rows),
            "n_kept": sum(r["kept"] for r in rows),
            "rate": sum(r["kept"] for r in rows) / len(rows) if rows else None,
            "missed": [r for r in rows if not r["kept"]],
        }
        res[f"{target}_unsupported"] = unsupported(rec[target], rec.get(f"{target}_prompt") or "")
    res["crowding"] = crowding(fs)
    res["unresolved"] = fs.get("unresolved", [])
    return res


IDENTITY_KEYS = ("topic", "semantic_details", "ufd_facts", "ufd_prompt", "ufd", "selected_ufd")


def identity(recs: dict[str, dict]) -> dict:
    """Per arm pair: the keys whose values differ from the first arm's."""
    arms = list(recs)
    ref = recs[arms[0]]
    return {f"{arms[0]}=={arm}": [k for k in IDENTITY_KEYS if recs[arm].get(k) != ref.get(k)]
            for arm in arms[1:]}


def _summary(rows: list[dict]) -> dict:
    def pooled(key):
        n = sum(r[key]["n_facts"] for r in rows)
        k = sum(r[key]["n_kept"] for r in rows)
        return {"n_facts": n, "n_kept": k, "rate": k / n if n else None}

    def unsup(key):
        ny = sum(r[key]["n_years"] for r in rows)
        nn = sum(r[key]["n_numbers"] for r in rows)
        by = sum(len(r[key]["unsupported_years"]) for r in rows)
        bn = sum(len(r[key]["unsupported_numbers"]) for r in rows)
        return {"years": ny, "unsupported_years": by, "numbers": nn, "unsupported_numbers": bn,
                "rate": (by + bn) / (ny + nn) if ny + nn else None}

    sel = sum(r["crowding"]["n_selected"] for r in rows)
    drop = sum(r["crowding"]["n_dropped"] for r in rows)
    return {
        "n_tables": len(rows),
        "ufd_retention": pooled("ufd_retention"),
        "sfd_retention": pooled("sfd_retention"),
        "ufd_unsupported": unsup("ufd_unsupported"),
        "sfd_unsupported": unsup("sfd_unsupported"),
        "crowding": {"n_selected": sel, "n_dropped": drop,
                     "share": drop / (sel + drop) if sel + drop else None,
                     "tables_with_drops": sum(r["crowding"]["n_dropped"] > 0 for r in rows)},
    }


def _fmt_rate(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arms", default="F1,F2")
    ap.add_argument("--run-id", default="r1")
    ap.add_argument("--dataset-ids", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ids = [l.strip() for l in args.dataset_ids.read_text(encoding="utf-8").splitlines() if l.strip()]

    per_arm: dict[str, list[dict]] = {a: [] for a in arms}
    ident, missing = {}, []
    for i in ids:
        recs = {}
        for arm in arms:
            rec = load_gen(arm, args.run_id, i)
            fs = load_factset(rec) if rec else None
            if rec is None or fs is None:
                missing.append(f"{arm}/{i}")
                continue
            recs[arm] = rec
            per_arm[arm].append(audit_table(rec, fs))
        if len(recs) > 1:
            ident[i] = identity(recs)

    report = {"arms": arms, "run_id": args.run_id, "dataset_ids": str(args.dataset_ids),
              "missing": missing, "summary": {a: _summary(r) for a, r in per_arm.items() if r},
              "identity": ident, "tables": per_arm}
    stem = f"__{args.dataset_ids.stem}"
    out = args.out or RESULTS_DIR / "audit" / f"{'_'.join(arms)}__{args.run_id}{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    for arm, s in report["summary"].items():
        u, f = s["ufd_unsupported"], s["sfd_unsupported"]
        print(f"[{arm}] tables={s['n_tables']}  "
              f"UFD kept {s['ufd_retention']['n_kept']}/{s['ufd_retention']['n_facts']} "
              f"({_fmt_rate(s['ufd_retention']['rate'])})  "
              f"SFD kept {s['sfd_retention']['n_kept']}/{s['sfd_retention']['n_facts']} "
              f"({_fmt_rate(s['sfd_retention']['rate'])})")
        print(f"      unsupported UFD years {u['unsupported_years']}/{u['years']} numbers "
              f"{u['unsupported_numbers']}/{u['numbers']}; SFD years {f['unsupported_years']}/{f['years']} "
              f"numbers {f['unsupported_numbers']}/{f['numbers']}")
        c = s["crowding"]
        print(f"      crowding: {c['n_dropped']} of {c['n_selected'] + c['n_dropped']} promptable facts "
              f"left out ({_fmt_rate(c['share'])}), {c['tables_with_drops']} tables")
        for r in per_arm[arm]:
            miss = [f"{m['id']}:{m['kind']}" for m in r["sfd_retention"]["missed"]]
            print(f"      {r['dataset_id'][:8]} UFD {r['ufd_retention']['n_kept']}/{r['ufd_retention']['n_facts']} "
                  f"SFD {r['sfd_retention']['n_kept']}/{r['sfd_retention']['n_facts']} "
                  f"SFD missed {miss} unsupported UFD {r['ufd_unsupported']['unsupported_years'] + r['ufd_unsupported']['unsupported_numbers']} "
                  f"SFD {r['sfd_unsupported']['unsupported_years'] + r['sfd_unsupported']['unsupported_numbers']}")
    for i, d in ident.items():
        bad = {k: v for k, v in d.items() if v}
        print(f"identity {i[:8]}: {'identical' if not bad else bad}")
    if missing:
        print(f"missing (no ok generation or no profile): {missing}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
