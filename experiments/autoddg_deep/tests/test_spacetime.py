"""spacetime: sentinel-proof spans, two kinds of resolution, place wording."""

from __future__ import annotations

import pandas as pd
import pytest

from autoddg.profiling_v2 import spacetime
from autoddg.profiling_v2.facts import FactSet


def _factset(df: pd.DataFrame, **kw) -> FactSet:
    fs = FactSet("t")
    fs.add(spacetime.facts(df, **kw))
    return fs


def _get(fs: FactSet, kind: str, col: str, basis: str | None = None):
    hits = [f for f in fs.by_kind(kind) if f.columns == [col]
            and (basis is None or f.value.get("basis") == basis)]
    assert len(hits) == 1, (kind, col, basis, hits)
    return hits[0]


def test_real_tables_valid_and_roundtrip(real_table):
    _, df = real_table
    fs = _factset(df)
    assert fs.validate() == []
    assert FactSet.from_json(fs.to_json()).to_dict() == fs.to_dict()
    assert fs.by_kind("temporal_span")


def test_year_span_ignores_sentinels(toy):
    fs = _factset(toy)
    span = _get(fs, "temporal_span", "year")
    assert (span.value["start"], span.value["end"]) == ("2001", "2020")
    v = span.evidence.violations
    assert (v["n_sentinel"], v["n_outside_fence"], v["n_rows"]) == (3, 1, 4)
    assert {e["value"] for e in v["examples"]} == {"9999", "1900"}
    assert span.evidence.support == {"n_ok": 196, "n_checked": 200, "rate": 0.98}


def test_date_string_sentinels():
    dates = list(pd.date_range("2015-01-01", "2016-12-31", freq="D").strftime("%Y-%m-%d"))
    dates += ["9999-12-31"] * 5 + ["1900-01-01"] * 3 + ["0001-01-01"]
    fs = _factset(pd.DataFrame({"closed_on": dates}))
    span = _get(fs, "temporal_span", "closed_on")
    assert (span.value["start"], span.value["end"]) == ("2015-01-01", "2016-12-31")
    v = span.evidence.violations
    assert v["n_sentinel"] == 6 and v["n_outside_fence"] == 3 and v["n_unparsed"] == 0
    assert span.value["years_present"] == [2015, 2016]


def test_mixed_formats_and_two_resolutions(toy):
    fs = _factset(toy)
    span = _get(fs, "temporal_span", "report_date")
    assert (span.value["start"], span.value["end"]) == ("2019-01", "2020-12")
    assert span.evidence.violations["n_unparsed"] == 1  # "not a date"
    assert _get(fs, "temporal_resolution", "report_date", "format").value["resolution"] == "month"
    gap = _get(fs, "temporal_resolution", "report_date", "observed_interval").value
    assert gap["interval"] == "month" and gap["share"] == 1.0


def test_daily_format_with_weekly_observations():
    # written as full dates (format: day) but only observed every Monday
    d = pd.date_range("2020-01-06", periods=52, freq="7D")
    fs = _factset(pd.DataFrame({"week_of": d.strftime("%m/%d/%Y")}))
    assert _get(fs, "temporal_resolution", "week_of", "format").value["resolution"] == "day"
    gap = _get(fs, "temporal_resolution", "week_of", "observed_interval")
    assert gap.value["interval"] == "week" and gap.status == "verified"


def test_real_spans_are_full_scan(birth_rates, infant_mortality, lobbyists):
    # datamart reports 1975 (trimmed); the full scan sees the 1970 rows
    span = _get(_factset(birth_rates), "temporal_span", "Year")
    assert span.value["start"] == "1970" and span.value["end"] == "2015"
    assert 1971 in span.value["missing_years"]
    q = _get(_factset(infant_mortality), "temporal_span", "Year and Quarter")
    assert q.value["resolution"] == "quarter" and q.value["start"].endswith("-Q1")
    filed = _get(_factset(lobbyists), "temporal_span", "Date Filed")
    parsed = pd.to_datetime(lobbyists["Date Filed"], format="mixed")
    assert filed.value["start"] == parsed.min().strftime("%Y-%m-%d")
    assert filed.value["end"] == parsed.max().strftime("%Y-%m-%d")


def test_places_claim_includes_not_covers(toy):
    fs = _factset(toy)
    county = _get(fs, "spatial_places", "county").value
    assert county["claim"] == "includes" and county["complete_list"] is True
    assert county["n_distinct"] == 4 and sum(p["count"] for p in county["places"]) == 170
    assert _get(fs, "spatial_level", "county").value["level"] == "county"
    z = _get(fs, "spatial_places", "zip").value
    assert z["complete_list"] is False and len(z["places"]) == spacetime.TOP_PLACES
    assert z["n_distinct"] > spacetime.MAX_PLACES_ALL


@pytest.mark.parametrize(
    "name, values, level",
    [
        ("State", ["NY", "CA"], "state"),
        ("business_city", ["SF"], "city"),
        ("ZipCode", [10001, 10002], "zip code"),
        ("State FIPS Code", [1, 2], None),
        ("County", [1, 2], None),
        ("district_id", ["a", "b"], None),
        ("Statement", ["x"], None),
    ],
)
def test_place_level(name, values, level):
    assert spacetime.place_level(name, pd.Series(values)) == level


@pytest.mark.parametrize(
    "name, values, kind",
    [
        ("Year", [2001, 2002, 2003], "year"),
        ("count", [2001, 2002, 2003], None),  # year-like numbers without a year header
        ("Month", ["January", "February", "March"], None),  # bare words
        ("zip", ["10001", "10002", "10003"], None),
        ("period", ["2023 Q1", "2023 Q2", "2023Q3"], "quarter"),
        ("ts", ["2020-01-01 10:00", "2020-01-02 11:30", "2020-01-03 09:15"], "datetime"),
    ],
)
def test_temporal_kind(name, values, kind):
    assert spacetime.temporal_kind(name, pd.Series(values)) == kind


def test_year_hint_from_other_profiler():
    df = pd.DataFrame({"yyyy": [2001, 2002, 2004]})
    assert not _factset(df).by_kind("temporal_span")
    assert _factset(df, temporal_hint=["yyyy"]).by_kind("temporal_span")
