"""factset/1.0 containers: required fields, enums, validation, JSON round trip."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from autoddg.profiling_v2.facts import (
    FACT_KINDS,
    SCHEMA_VERSION,
    STATUSES,
    Evidence,
    Fact,
    FactSchemaError,
    FactSet,
    jsonable,
    support,
)

FACT_KEYS = {
    "fact_id", "kind", "scope", "columns", "value", "render",
    "evidence", "confidence", "status", "priority", "tokens",
}
EVIDENCE_KEYS = {"method", "source", "scan", "support", "violations", "upstream"}
FACTSET_KEYS = {
    "schema_version", "dataset_id", "table", "provenance",
    "domain", "columns", "facts", "budget",
}


def _fact(**kw) -> Fact:
    base = dict(
        kind="table_size",
        scope="table",
        columns=[],
        value={"n_rows": 10, "n_cols": 2},
        evidence=Evidence(method="test", support=support(10, 10)),
    )
    base.update(kw)
    return Fact(**base)


def test_enums_match_impl_section_4():
    assert len(FACT_KINDS) == 16
    assert STATUSES == ("verified", "inferred", "uncertain", "suppressed")


def test_ids_priority_confidence_defaults():
    fs = FactSet("ds")
    a, b = fs.add([_fact(), _fact(kind="observation_unit", scope="column_set",
                                  columns=["a", "b"], value={})])
    assert (a.fact_id, b.fact_id) == ("F001", "F002")
    assert b.priority < a.priority
    assert a.confidence == {"value": 1.0, "type": "support_rate", "calibrator": None}


def test_roundtrip_and_keys():
    fs = FactSet("ds", table={"n_rows_total": 10})
    fs.add([_fact(), _fact(kind="missingness", scope="column", columns=["x"],
                           value={"null_rate": 0.5},
                           evidence=Evidence(method="m", upstream=["F001"]))])
    d = json.loads(fs.to_json())
    assert set(d) == FACTSET_KEYS and d["schema_version"] == SCHEMA_VERSION
    for f in d["facts"]:
        assert set(f) == FACT_KEYS and set(f["evidence"]) == EVIDENCE_KEYS
    back = FactSet.from_json(fs.to_json())
    assert back.to_dict() == fs.to_dict()


@pytest.mark.parametrize(
    "bad, msg",
    [
        (dict(kind="nope"), "unknown kind"),
        (dict(status="maybe"), "unknown status"),
        (dict(scope="column", columns=[]), "exactly 1 column"),
        (dict(scope="table", columns=["a"]), "lists columns"),
        (dict(evidence=Evidence(method="m", support={"n_ok": 5, "n_checked": 2, "rate": 2.5})),
         "n_ok outside"),
        (dict(evidence=Evidence(method="m", violations={"examples": []})), "int n_rows"),
        (dict(value={"x": np.float64("nan")}), "JSON"),
        (dict(evidence=Evidence(method="m", upstream=["F999"])), "unknown"),
    ],
)
def test_validation_rejects(bad, msg):
    fs = FactSet("ds")
    fs.add([_fact(**bad)])
    errs = fs.validate()
    assert any(msg in e for e in errs), errs
    with pytest.raises(FactSchemaError):
        FactSet.from_json(json.dumps(fs.to_dict(), allow_nan=True))


def test_jsonable_numpy_pandas():
    out = jsonable({"a": np.int64(3), "b": np.float32(0.5), "c": [np.nan, np.inf],
                    "d": pd.Timestamp("2020-01-02"), "e": pd.NaT, "f": np.bool_(True)})
    assert out == {"a": 3, "b": 0.5, "c": [None, None], "d": "2020-01-02T00:00:00",
                   "e": None, "f": True}
    json.dumps(out, allow_nan=False)
