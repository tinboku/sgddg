"""select + render: status thresholds, datamart/full-scan dedupe, budget, empty slots."""

from __future__ import annotations

from autoddg.profiling_v2 import render, select
from autoddg.profiling_v2.facts import Evidence, Fact, FactSet, support
from autoddg.profiling_v2.pipeline import build_factset, fact_texts
from autoddg.profiling_v2.select import SelectConfig


def _fact(kind, cols, value, status="verified", source="computed", sup=None, **ev):
    return Fact(kind=kind, scope="column" if len(cols) == 1 else ("table" if not cols else "column_set"),
                columns=cols, value=value, status=status,
                evidence=Evidence(method="t", source=source, support=sup or {}, **ev))


def _run(facts, cfg=None) -> FactSet:
    cfg = cfg or SelectConfig()
    fs = FactSet("t")
    fs.add(facts)
    select.select(fs, cfg)
    render.render_facts(fs)
    select.budget(fs, cfg)
    assert fs.validate() == []
    return fs


SPAN = {"start": "2001", "end": "2020", "resolution": "year", "kind": "year", "n_distinct_times": 20}


# --- status -------------------------------------------------------------------

def test_rate_thresholds_only_lower_status():
    cfg = SelectConfig()
    facts = [
        _fact("temporal_span", ["a"], dict(SPAN), sup=support(99, 100)),
        _fact("temporal_span", ["b"], dict(SPAN), sup=support(90, 100)),
        _fact("temporal_span", ["c"], dict(SPAN), sup=support(60, 100)),
        _fact("temporal_span", ["d"], dict(SPAN), sup=support(10, 100)),
        # module said inferred; a perfect rate must not upgrade it
        _fact("measure_unit", ["e"], {"unit": "$", "basis": "value_affix"}, status="inferred",
              sup=support(100, 100)),
        # suppressed is sticky
        _fact("temporal_span", ["f"], dict(SPAN), status="suppressed", sup=support(100, 100)),
    ]
    select.assign_status(facts, cfg)
    assert [f.status for f in facts] == ["verified", "inferred", "uncertain", "suppressed",
                                         "inferred", "suppressed"]


def test_coverage_share_is_not_a_correctness_rate():
    # top-10 covering 5% of a high-cardinality column is still a true fact
    f = _fact("category_values", ["x"], {"basis": "top_values", "top": [{"value": "a", "count": 5}]},
              sup=support(5, 100))
    select.assign_status([f], SelectConfig())
    assert f.status == "verified"


# --- dedupe -------------------------------------------------------------------

def test_full_scan_wins_over_datamart(birth_rates):
    fs = build_factset(birth_rates, "2a3cc123")
    for kind, col in [("category_values", "Age Group"), ("measure_range", "Birth Rate"),
                      ("temporal_span", "Year"), ("missingness", "Birth Rate")]:
        dm = [f for f in fs.by_kind(kind) if f.columns == [col] and f.evidence.source == "datamart"]
        full = [f for f in fs.by_kind(kind) if f.columns == [col] and f.evidence.source != "datamart"]
        assert len(dm) == 1 and len(full) == 1, (kind, col)
        assert dm[0].status == "suppressed"
        if kind != "missingness":  # both are below the missingness threshold
            assert dm[0].value["_select"][-1] == {"reason": "duplicate_full_scan",
                                                  "superseded_by": full[0].fact_id}
            assert full[0].status != "suppressed"
    # the trimmed datamart span (1975) never reaches a prompt; the full scan (1970) does
    texts = fact_texts(fs)
    assert "Year: 1970 to 2015" in texts["ufd_facts"]
    assert "1975" not in texts["ufd_facts"] and "1975" not in texts["sfd_facts"]


def test_trimmed_without_full_scan_is_uncertain():
    f = _fact("measure_range", ["x"], {"trimmed_min": 1, "trimmed_max": 9, "trimmed": True},
              status="inferred", source="datamart")
    fs = _run([f])
    assert f.status == "uncertain"
    assert f.fact_id in fs.budget["selected_ufd"] and f.fact_id not in fs.budget["selected_sfd"]
    assert render.slots(fs, "ufd")["measures_and_units"] == "likely x: mostly 1 to 9"


def test_restated_facts_suppressed():
    facts = [
        _fact("temporal_span", ["year"], dict(SPAN), sup=support(100, 100)),
        _fact("measure_range", ["year"], {"min": 2001, "max": 2020}, sup=support(100, 100)),
        _fact("measure_range", ["county_fips"], {"min": 1001, "max": 56045}, sup=support(100, 100)),
        _fact("spatial_places", ["county"], {"level": "county", "n_distinct": 2, "complete_list": True,
                                             "places": [{"value": "Kings", "count": 3},
                                                        {"value": "Erie", "count": 1}], "claim": "includes",
                                             "confirmed": {"basis": "test", "match_rate": 1.0}}),
        _fact("category_values", ["county"], {"basis": "top_values", "top": [{"value": "Kings", "count": 3}]}),
    ]
    _run(facts)
    assert [f.status for f in facts] == ["verified", "suppressed", "suppressed", "verified", "suppressed"]


# --- budget / status filter -----------------------------------------------------

def _mixed() -> list[Fact]:
    return [
        _fact("table_size", [], {"n_rows": 100, "n_cols": 3}),
        _fact("temporal_span", ["year"], dict(SPAN), sup=support(100, 100)),
        _fact("category_values", ["kind"], {"basis": "top_values", "n_distinct": 2,
                                            "top": [{"value": "SECRETVALUE", "count": 9}]},
              status="suppressed"),
        _fact("category_values", ["grp"], {"basis": "top_values", "n_distinct": 2,
                                           "top": [{"value": "maybe", "count": 9}]}, status="uncertain"),
        _fact("measure_unit", ["amt"], {"unit": "$", "basis": "header"}, status="inferred"),
    ]


def test_suppressed_never_prompted_uncertain_ufd_only():
    fs = _run(_mixed())
    ids = {f.columns[0] if f.columns else "table": f.fact_id for f in fs.facts}
    assert ids["kind"] not in fs.budget["selected_ufd"] + fs.budget["selected_sfd"]
    assert ids["grp"] in fs.budget["selected_ufd"] and ids["grp"] not in fs.budget["selected_sfd"]
    texts = fact_texts(fs)
    assert "SECRETVALUE" not in texts["ufd_facts"] and "SECRETVALUE" not in texts["sfd_facts"]
    assert "likely grp: maybe" in texts["ufd_facts"]
    assert "maybe" not in texts["sfd_facts"]
    assert "amt unit: $" in texts["ufd_facts"] and "amt unit: $" in texts["sfd_facts"]
    assert {"id": ids["kind"], "reason": "status:suppressed"} in fs.budget["dropped"]


def test_budget_respects_tokens_priority_and_caps():
    facts = [_fact("category_values", [f"c{i}"], {"basis": "top_values", "n_distinct": 1,
                                                  "top": [{"value": "v" * 40, "count": 1}]})
             for i in range(6)]
    facts.append(_fact("temporal_span", ["year"], dict(SPAN), sup=support(100, 100)))
    cfg = SelectConfig(ufd_max_tokens=40)
    fs = _run(facts, cfg)
    by_id = {f.fact_id: f for f in fs.facts}
    chosen = fs.budget["selected_ufd"]
    assert by_id[chosen[0]].kind == "temporal_span"  # priority 2 before category_values (5)
    assert sum(by_id[i].tokens["ufd"] for i in chosen) == fs.budget["used_ufd_tokens"] <= 40
    assert any(d["reason"] == "budget" for d in fs.budget["dropped"])
    # with a large budget the per-kind cap (4 category facts) binds
    fs = _run([Fact.from_dict({**f.to_dict(), "fact_id": "", "render": None}) for f in facts],
              SelectConfig(ufd_max_tokens=10_000))
    assert sum(fs.facts[int(i[1:]) - 1].kind == "category_values" for i in fs.budget["selected_ufd"]) == 4
    assert sum(d["reason"] == "kind_cap" for d in fs.budget["dropped"]) == 2


# --- render: empty slots --------------------------------------------------------

def test_empty_factset_gives_seven_empty_slots():
    fs = _run([])
    for target in ("ufd", "sfd"):
        s = render.slots(fs, target)
        assert list(s) == list(render.SLOTS) and all(v == "" for v in s.values())
        assert render.format_slots(s) == ""
    # no budget at all (render called before select.budget) is not an error either
    assert all(v == "" for v in render.slots(FactSet("t"), "ufd").values())


def test_missing_slots_stay_empty():
    fs = _run([_fact("table_size", [], {"n_rows": 100, "n_cols": 3})])
    s = render.slots(fs, "ufd")
    assert s["identity"] == "100 rows, 3 columns"
    assert all(s[k] == "" for k in render.SLOTS if k != "identity")
    assert render.format_slots(s) == "- Identity: 100 rows, 3 columns"


def test_places_wording_and_sfd_terms():
    value = {"level": "county", "n_distinct": 62, "complete_list": False, "claim": "includes",
             "places": [{"value": v, "count": 1} for v in ["Kings", "Queens", "Erie", "Bronx", "Albany", "Monroe"]],
             "confirmed": {"basis": "test", "match_rate": 1.0}}
    fs = _run([_fact("spatial_places", ["County"], value)])
    assert render.slots(fs, "ufd")["geographic_scope"] == (
        "County (county) includes Kings, Queens, Erie, Bronx, Albany and others (62 distinct)")
    assert render.slots(fs, "sfd")["geographic_scope"].endswith("(county-level, by county)")
    # an unconfirmed candidate names only the likely level, UFD only
    fs = _run([_fact("spatial_places", ["County"], {**value, "confirmed": None}, status="uncertain")])
    assert render.slots(fs, "ufd")["geographic_scope"] == "County: probably county names (not confirmed)"
    assert render.slots(fs, "sfd")["geographic_scope"] == ""


def test_toy_frame_limitations(toy):
    fs = build_factset(toy, "toy")
    lim = render.slots(fs, "ufd")["limitations"]
    # exclusions are part of the span fact itself, not a limitation line
    assert "year" not in lim
    assert "excluding 3 values outside 1800-2100" in render.slots(fs, "ufd")["temporal_scope"]
    assert "county: 15% missing" not in lim  # 30/200 = 15% < missingness_min_rate
    assert fs.validate() == []


def test_build_factset_on_real_tables(real_table):
    dataset_id, df = real_table
    fs = build_factset(df, dataset_id)
    assert fs.validate() == []
    assert FactSet.from_json(fs.to_json()).to_dict() == fs.to_dict()
    texts = fact_texts(fs)
    assert texts["ufd_facts"].startswith("- ")
    size = [f.fact_id for f in fs.by_kind("table_size")]
    assert size[0] in fs.budget["selected_ufd"] or {"id": size[0], "reason": "budget"} in fs.budget["dropped"]
    assert fs.budget["used_ufd_tokens"] <= fs.budget["ufd_max_tokens"]
    assert set(fs.budget["selected_sfd"]) <= set(fs.budget["selected_ufd"])
    prompted = set(fs.budget["selected_ufd"]) | set(fs.budget["selected_sfd"])
    assert not any(f.status == "suppressed" for f in fs.facts if f.fact_id in prompted)
    assert all(fs.facts[int(i[1:]) - 1].status in select.SFD_STATUSES for i in fs.budget["selected_sfd"])
