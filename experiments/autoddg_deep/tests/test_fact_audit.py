"""05_fact_audit: writing variants count as kept, unsupported numbers, crowding."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from autoddg.profiling_v2.pipeline import build_factset

HERE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("audit05", HERE / "05_fact_audit.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def _span(start, end):
    return {"kind": "temporal_span", "value": {"start": start, "end": end}}


@pytest.mark.parametrize("text", [
    "Covers 2023-Q1 to 2025-Q4.", "from 2023 Q1 through 2025 Q4", "between Q1 2023 and Q4 2025",
    "from the first quarter of 2023 to the fourth quarter of 2025",
])
def test_quarter_writings_are_kept(text):
    assert audit.fact_kept(_span("2023-Q1", "2025-Q4"), text)[0]


def test_dates_and_months():
    assert audit.fact_kept(_span("2016-06-17", "2018-12-13"), "filed June 17, 2016 to Dec 13, 2018")[0]
    assert audit.fact_kept(_span("2019-01", "2020-12"), "January 2019 - December 2020")[0]
    assert not audit.fact_kept(_span("2016-06-17", "2018-12-13"), "filed 2016 to 2018")[0]


def test_numbers_rounded_but_not_loose():
    rng = {"kind": "measure_range", "value": {"min": 1.8, "max": 161.1}}
    assert audit.fact_kept(rng, "rates from 1.8 to 161 per 1,000")[0]
    assert not audit.fact_kept(rng, "rates from 2 to 161")[0]
    assert audit.fact_kept({"kind": "table_size", "value": {"n_rows": 1068}}, "1,068 rows")[0]


def test_list_facts_need_two_values():
    cat = {"kind": "category_values", "value": {"top": [{"value": v} for v in ["Kings", "Queens", "Erie"]]}}
    assert not audit.fact_kept(cat, "records for Kings")[0]
    assert audit.fact_kept(cat, "records for Kings and Erie")[0]


def test_unsupported_numbers_against_prompt():
    u = audit.unsupported("Data from 1970 to 2016, rates up to 161 and 12.5%.",
                          "Year: 1970 to 2015; Birth Rate: 1.8 to 161.1")
    assert u["unsupported_years"] == ["2016"]
    assert u["unsupported_numbers"] == ["12.5"]


def test_crowding_on_built_factset():
    df = pd.DataFrame({f"cat{i}": [f"value number {j}" for j in range(5)] * 20 for i in range(8)})
    from autoddg.profiling_v2.select import SelectConfig
    fs = build_factset(df, "crowd", SelectConfig(ufd_max_tokens=60)).to_dict()
    c = audit.crowding(fs)
    assert c["n_dropped"] > 0 and 0 < c["crowded_share"] < 1
    assert {d["reason"] for d in c["dropped"]} <= {"budget", "kind_cap"}
