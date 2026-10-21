"""stats: missingness, top-k, quantiles, formats and verbatim units."""

from __future__ import annotations

import pandas as pd
import pytest

from autoddg.profiling_v2 import stats
from autoddg.profiling_v2.facts import FactSet


def _factset(df: pd.DataFrame) -> FactSet:
    fs = FactSet("t", columns=stats.column_records(df))
    fs.add(stats.facts(df))
    return fs


def _one(fs: FactSet, kind: str, col: str):
    hits = [f for f in fs.by_kind(kind) if f.columns == [col]]
    assert len(hits) == 1, (kind, col, hits)
    return hits[0]


def _units(fs: FactSet, col: str) -> list:
    return [f.value.get("unit") for f in fs.by_kind("measure_unit") if f.columns == [col]]


def test_real_tables_valid_and_roundtrip(real_table):
    _, df = real_table
    fs = _factset(df)
    assert fs.validate() == []
    assert FactSet.from_json(fs.to_json()).to_dict() == fs.to_dict()
    assert [c["name"] for c in fs.columns] == [str(c) for c in df.columns]


def test_toy_missing_categorical(toy):
    fs = _factset(toy)
    miss = _one(fs, "missingness", "county")
    assert miss.value["n_missing"] == 30 and miss.value["null_rate"] == 30 / 200
    cat = _one(fs, "category_values", "county")
    assert {t["value"] for t in cat.value["top"]} == {"Kings", "Queens", "Bronx", "Erie"}
    assert sum(t["count"] for t in cat.value["top"]) == 170
    assert cat.value["complete"] is True


def test_blank_strings_count_as_missing():
    df = pd.DataFrame({"c": ["a", " ", "", None, "b"] * 4})
    miss = _one(_factset(df), "missingness", "c")
    assert miss.value == {"null_rate": 0.6, "n_missing": 12, "n_blank_strings": 8}


def test_quantiles_show_sentinel_instead_of_hiding_it(toy):
    rng = _one(_factset(toy), "measure_range", "year")
    # raw stats keep the 9999 sentinel visible; spacetime is what drops it
    assert rng.value["max"] == 9999 and rng.value["min"] == 1900
    assert 2001 <= rng.value["quantiles"]["q50"] <= 2020


def test_units_are_verbatim(toy):
    fs = _factset(toy)
    assert _units(fs, "Amount ($)") == ["$"]
    assert _units(fs, "price") == ["$"]
    # 0-100 values never become a percentage
    assert _units(fs, "score") == []
    cols = {c["name"]: c for c in fs.columns}
    assert cols["Amount ($)"]["unit"] == "$" and cols["score"]["unit"] is None
    text = fs.to_json()
    assert "USD" not in text and "dollar" not in text.lower()
    assert "percent" not in text.lower()
    price = _one(fs, "measure_range", "price")
    assert price.value["parsed_from_text"] is True and price.value["min"] >= 1


@pytest.mark.parametrize(
    "header, units",
    [
        ("Amount ($)", ["$"]),
        ("Weight (kg)", ["kg"]),
        ("Concentration [mg/L]", ["mg/L"]),
        ("% of Annual Goal Completed", ["%"]),
        ("Rate per 100,000 population", ["per 100,000 population"]),
        ("Budget (in thousands)", ["in thousands"]),
        ("cost_usd", ["usd"]),
        ("score", []),
        ("Birth Rate", []),
        ("Year (2010)", []),
    ],
)
def test_header_units(header, units):
    assert stats.header_units(header) == units


def test_formats(toy):
    cols = {c["name"]: c for c in _factset(toy).columns}
    z = cols["zip"]["format"]
    assert z["pattern"] == r"^\d{5}$" and z["match_rate"] == 1.0
    assert z["named"]["name"] == "zip_code"
    # zip codes stay text: no range, but top-k values
    assert cols["zip"]["physical"] == "string"
    dates = cols["report_date"]["format"]
    assert len(dates["patterns"]) == 3 and all(p["example"] for p in dates["patterns"])
    assert cols["year"]["format"] is None


def test_real_unit_column(infant_mortality):
    fs = _factset(infant_mortality)
    unit = _one(fs, "measure_unit", "Unit")
    assert {u["value"] for u in unit.value["units"]} == {"per 1,000 births", "per 100,000 births"}
