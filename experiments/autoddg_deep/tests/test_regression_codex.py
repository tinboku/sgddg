"""Regression tests for the seven must-fix items in REVIEW_codex_M2.md.

Each test reproduces the counterexample Codex ran in memory (numbered as in
the review) and pins the fixed behaviour.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

from autoddg.description.search_factaware import FactAwareSearchDescription
from autoddg.llm import LLMClient
from autoddg.profiling_v2 import select, stats
from autoddg.profiling_v2.facts import SCHEMA_VERSION
from autoddg.profiling_v2.pipeline import build_factset, fact_texts

HERE = Path(__file__).resolve().parents[1]


def _facts(fs, kind, col=None, computed=True):
    return [f for f in fs.facts if f.kind == kind and (col is None or f.columns == [col])
            and (not computed or f.evidence.source != "datamart")]


def _one(fs, kind, col):
    out = _facts(fs, kind, col)
    assert len(out) == 1, out
    return out[0]


def _all_text(fs) -> str:
    t = fact_texts(fs)
    return t["ufd_facts"] + "\n" + t["sfd_facts"]


# --- 1. percentile trimming must not drop real endpoints ----------------------


def test_c1_real_endpoint_is_kept_in_span():
    fs = build_factset(pd.DataFrame({"Year": [1970] + [2020] * 999}), "c1")
    span = _one(fs, "temporal_span", "Year")
    assert (span.value["start"], span.value["end"]) == ("1970", "2020")
    # the robust core is reported separately, never as the span
    assert (span.value["core"]["start"], span.value["core"]["end"]) == ("2020", "2020")
    assert span.value["suspect"]["n_far"] == 1
    ufd = fact_texts(fs)["ufd_facts"]
    assert "Year: 1970 to 2020" in ufd
    assert "2020 to 2020" not in ufd


def test_c1_implausible_years_are_named_in_the_fact_not_hidden(toy):
    fs = build_factset(toy, "toy")
    span = _one(fs, "temporal_span", "year")
    # 1900 is a plausible calendar year: it stays in the span, flagged as far
    assert (span.value["start"], span.value["end"]) == ("1900", "2020")
    assert span.value["suspect"]["n_implausible"] == 3
    assert span.status != "verified"  # the span leaves 3 values out
    text = span.render["ufd"]
    assert "excluding 3 values outside 1800-2100" in text and "9999" in text


# --- 2. format precision is not an observation frequency -----------------------


FREQ_WORDS = ("daily", "annual", "yearly", "monthly", "weekly", "hourly", "quarterly")


def test_c2_two_dates_get_no_frequency_word():
    fs = build_factset(pd.DataFrame({"date": ["2020-01-02", "2021-05-03"] * 5}), "c2")
    text = _all_text(fs).lower()
    assert not any(w in text for w in FREQ_WORDS), text
    for f in fs.facts:
        if f.kind == "temporal_resolution" and f.value.get("basis") != "observed_interval" and f.render:
            assert f.render["sfd_terms"] == []


def test_c2_second_precision_endpoints_keep_seconds():
    fs = build_factset(pd.DataFrame({"ts": [f"2020-01-01 12:00:{s:02d}" for s in range(1, 60)]}), "c2s")
    span = _one(fs, "temporal_span", "ts")
    assert (span.value["start"], span.value["end"]) == ("2020-01-01 12:00:01", "2020-01-01 12:00:59")
    assert "12:00 to 2020-01-01 12:00\n" not in fact_texts(fs)["ufd_facts"] + "\n"


def test_c2_observed_interval_is_about_time_points_not_collection():
    fs = build_factset(pd.DataFrame({"Year": list(range(2001, 2021)) * 3}), "c2y")
    gap = [f for f in fs.facts if f.kind == "temporal_resolution"
           and f.value.get("basis") == "observed_interval"][0]
    assert "time points" in gap.render["ufd"]
    for w in ("collected", "frequency", "updated"):
        assert w not in gap.render["ufd"]
    assert all("time points" in t for t in gap.render["sfd_terms"])


# --- 3. place heuristics are candidates, not verified geography ----------------


def test_c3_state_open_closed_is_not_geography():
    fs = build_factset(pd.DataFrame({"State": ["Open"] * 60 + ["Closed"] * 40}), "c3")
    assert all(f.status == "suppressed" for f in _facts(fs, "spatial_places", "State"))
    assert all(f.status == "suppressed" for f in _facts(fs, "spatial_level", "State"))
    cat = _one(fs, "category_values", "State")
    assert cat.status != "suppressed"
    t = fact_texts(fs)
    assert t["ufd_slots"]["geographic_scope"] == "" and t["sfd_slots"]["geographic_scope"] == ""
    assert "Open" in t["ufd_slots"]["representative_categories"]
    assert "state-level" not in t["sfd_facts"]


def test_c3_unconfirmed_places_stay_candidates_and_keep_categories():
    fs = build_factset(pd.DataFrame({"County": ["Kings", "Queens", "Erie", "Albany"] * 25}), "c3c")
    places = _one(fs, "spatial_places", "County")
    assert places.status not in select.SFD_STATUSES
    assert places.value["confirmed"] is None
    assert _one(fs, "category_values", "County").status != "suppressed"
    t = fact_texts(fs)
    assert t["sfd_slots"]["geographic_scope"] == ""
    assert "county-level" not in t["sfd_facts"]


def test_c3_states_confirmed_by_vocabulary_are_verified():
    fs = build_factset(pd.DataFrame({"State": ["New York", "NY", "California", "TX"] * 25}), "c3v")
    places = _one(fs, "spatial_places", "State")
    assert places.status == "verified"
    assert places.value["confirmed"]["basis"] == "vocabulary:us_states"
    assert _one(fs, "category_values", "State").status == "suppressed"


# --- 4. units: no bracket annotations, no mixed-unit range ---------------------


@pytest.mark.parametrize("header", ["Case (Y/N)", "Report (MM/DD/YYYY)", "Flag (A/B/C)"])
def test_c4_bracket_annotations_are_not_units(header):
    assert stats.header_units(header) == []


@pytest.mark.parametrize("header, units", [
    ("Amount ($)", ["$"]),
    ("Rate (per 1,000)", ["per 1,000"]),
    ("BMI (kg/m2)", ["kg/m2"]),
    ("Price ($/lb)", ["$/lb"]),
])
def test_c4_real_bracket_units_still_found(header, units):
    assert stats.header_units(header) == units


def test_c4_mixed_currency_gives_per_unit_ranges():
    fs = build_factset(pd.DataFrame({"price": ["$1"] * 95 + ["€10000"] * 5}), "c4")
    rng = _one(fs, "measure_range", "price")
    assert rng.value["mixed_units"] is True
    assert "min" not in rng.value and "max" not in rng.value
    by = {g["unit"]: g for g in rng.value["by_unit"]}
    assert (by["$"]["min"], by["$"]["max"], by["$"]["n"]) == (1, 1, 95)
    assert (by["€"]["min"], by["€"]["max"], by["€"]["n"]) == (10000, 10000, 5)
    units = [f for f in _facts(fs, "measure_unit", "price")]
    assert [u.value["unit"] for u in units] == ["mixed"]
    text = rng.render["ufd"]
    assert "1 to 10000" not in text and "$" in text and "€" in text


# --- 5. qualifiers are part of the fact text -----------------------------------


def test_c5_partially_parseable_dates_are_qualified_and_not_verified():
    years = [y for y in range(2001, 2011) if y != 2005]
    dates = [f"{years[i % 9]}-06-01" for i in range(80)] + ["unknown"] * 20
    fs = build_factset(pd.DataFrame({"report_date": dates}), "c5")
    span = _one(fs, "temporal_span", "report_date")
    assert span.status == "inferred"
    text = span.render["ufd"]
    assert "80% parseable" in text
    # "no rows for 2005" would be a full-table claim: it is either dropped or qualified
    assert "no rows" not in text
    assert span.render["limitation"] is None
    assert fact_texts(fs)["ufd_slots"]["limitations"] == ""


def test_c5_category_share_names_its_denominator():
    fs = build_factset(pd.DataFrame({"flag": ["True"] * 70 + [None] * 30}), "c5c")
    text = _one(fs, "category_values", "flag").render["ufd"]
    assert "True (100%)" not in text
    assert "of 70 non-blank" in text


def test_c5_no_partial_fact_is_verified():
    fs = build_factset(pd.DataFrame({
        "report_date": [f"200{i % 9 + 1}-06-01" for i in range(80)] + ["unknown"] * 20,
        "amount": [str(i) for i in range(90)] + ["n/a"] * 10,
    }), "c5p")
    for f in fs.facts:
        if f.value.get("partial"):
            assert f.status != "verified", f


# --- 6. caches are bound to the extractor version and the fact input ----------


def _load_generate():
    spec = importlib.util.spec_from_file_location("gen02", HERE / "02_generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_c6_profile_cache_needs_matching_extractor_version(tmp_path, monkeypatch):
    gen = _load_generate()
    cfg = yaml.safe_load((HERE / "configs" / "arm_F1.yaml").read_text())
    ver = gen.extractor_version(cfg)
    assert ver["schema"] == SCHEMA_VERSION and ver["commit"] and ver["src_sha256"]
    key = gen.profile_key(cfg)
    # a pre-fix cache entry (table + config hash only) is stale
    assert not gen.profile_cache_valid({"table_sha256": "t", "config_hash": gen.config_hash(cfg)}, "t", key)
    assert gen.profile_cache_valid({"table_sha256": "t", "profile_key": key}, "t", key)
    assert not gen.profile_cache_valid({"table_sha256": "u", "profile_key": key}, "t", key)
    # editing an extractor source file changes the key
    src = tmp_path / "x.py"
    src.write_text("A = 1\n")
    monkeypatch.setattr(gen, "extractor_sources", lambda cfg: [src])
    k1 = gen.profile_key(cfg)
    src.write_text("A = 2\n")
    assert gen.profile_key(cfg) != k1


def test_c6_generation_cache_depends_on_fact_input():
    gen = _load_generate()
    meta = {"config_hash": "c", "prompt_hash": "p"}
    prof_a = {"ufd_facts": "- Temporal scope: Year: 2001 to 2020", "sfd_facts": None}
    prof_b = {"ufd_facts": "- Temporal scope: Year: 1970 to 2020", "sfd_facts": None}
    rec = {"config_hash": "c", "prompt_hash": "p", "table_sha256": "t", "status": "ok", "ufd": "x",
           "fact_input_sha256": gen.fact_input_sha(prof_a)}
    assert gen.gen_cache_valid(rec, meta, "t", gen.fact_input_sha(prof_a), want_llm=True)
    assert not gen.gen_cache_valid(rec, meta, "t", gen.fact_input_sha(prof_b), want_llm=True)
    old = {k: v for k, v in rec.items() if k != "fact_input_sha256"}
    assert not gen.gen_cache_valid(old, meta, "t", gen.fact_input_sha(prof_a), want_llm=True)


# --- 7. E3: one common fact set, F2 only adds the SFD channel -----------------


def test_c7_sfd_selection_is_the_ufd_selection(toy, birth_rates):
    for df in (toy, birth_rates):
        fs = build_factset(df, "c7")
        status = {f.fact_id: f.status for f in fs.facts}
        assert fs.budget["selected_sfd"] == [i for i in fs.budget["selected_ufd"]
                                             if status[i] in select.SFD_STATUSES]


def test_c7_f1_f2_configs_share_facts_and_constraints():
    f1 = yaml.safe_load((HERE / "configs" / "arm_F1.yaml").read_text())
    f2 = yaml.safe_load((HERE / "configs" / "arm_F2.yaml").read_text())
    assert f1["facts"]["select"] == f2["facts"]["select"]
    assert f2["facts"].get("profile_from") == "F1"
    assert not f1["facts"].get("sfd_constraints") and not f2["facts"].get("sfd_constraints")
    drop = ("arm", "facts")
    assert {k: v for k, v in f1.items() if k not in drop} == {k: v for k, v in f2.items() if k not in drop}
    ab = yaml.safe_load((HERE / "configs" / "arm_F2c.yaml").read_text())
    assert ab["facts"]["sfd_constraints"] is True and ab["facts"]["profile_from"] == "F1"


class _Rec(LLMClient):
    def chat_completions_create(self, model, messages, temperature=None, **kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": "x"}}]}


def test_c7_sfd_channel_adds_no_expansion_constraints():
    base = FactAwareSearchDescription(_Rec(), "m").build_prompt("D.", "T").lower()
    plain = FactAwareSearchDescription(_Rec(), "m").build_prompt("D.", "T", verified_facts="- F: 1")
    low = plain.lower()
    # the channel adds the facts, no wording beyond the parent's expansion instructions
    for w in ("synonym", "do not extend", "do not change", "variants", "verbatim"):
        assert low.count(w) == base.count(w), w
    constrained = FactAwareSearchDescription(_Rec(), "m", fact_constraints=True).build_prompt(
        "D.", "T", verified_facts="- F: 1")
    assert constrained.lower().count("synonym") > base.count("synonym")
    assert "do not change or extend" in constrained.lower()
    assert constrained.startswith(plain[: plain.index("- F: 1")])


def test_c7_tight_ufd_budget_never_lets_sfd_pick_more():
    df = pd.DataFrame({f"cat{i}": [f"v{j}" for j in range(5)] * 20 for i in range(6)})
    cfg = select.SelectConfig(ufd_max_tokens=30)
    fs = build_factset(df, "c7b", cfg)
    assert fs.budget["selected_ufd"]
    assert set(fs.budget["selected_sfd"]) <= set(fs.budget["selected_ufd"])
