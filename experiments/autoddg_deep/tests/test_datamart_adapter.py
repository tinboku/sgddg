"""datamart_adapter: every datamart field becomes a schema-valid Fact."""

from __future__ import annotations

import json

import pandas as pd

from autoddg.profiling_v2 import datamart_adapter
from autoddg.profiling_v2.facts import FactSet


def _factset(df: pd.DataFrame, meta: dict | None = None) -> FactSet:
    fs = FactSet("t")
    fs.add(datamart_adapter.facts(df, meta=meta))
    return fs


def test_real_tables_valid_and_roundtrip(real_table):
    ds_id, df = real_table
    fs = _factset(df)
    assert fs.validate() == []
    assert FactSet.from_json(fs.to_json()).to_dict() == fs.to_dict()
    size = fs.by_kind("table_size")[0].value
    assert size["n_rows"] == len(df) and size["n_cols"] == df.shape[1]
    # one column_semantic fact per column carries num_distinct_values
    assert {f.columns[0] for f in fs.by_kind("column_semantic")} == set(df.columns)


def test_birth_rates_year_span_is_trimmed(birth_rates):
    fs = _factset(birth_rates)
    span = fs.by_kind("temporal_span")[0]
    assert span.columns == ["Year"] and span.value["resolution"] == "year"
    # datamart trims k-means clusters to 5-95%: the 6 rows of 1970 drop out
    assert birth_rates["Year"].min() == 1970
    assert span.value["start"] == "1975" and span.value["end"] == "2015"
    assert span.value["trimmed"] is True
    rng = next(f for f in fs.by_kind("measure_range") if f.columns == ["Birth Rate"])
    assert rng.value["trimmed_min"] >= birth_rates["Birth Rate"].min()
    assert rng.value["trimmed_max"] <= birth_rates["Birth Rate"].max()


def test_categorical_histogram_sorted_by_count(birth_rates):
    fs = _factset(birth_rates)
    cat = next(f for f in fs.by_kind("category_values") if f.columns == ["Race or Hispanic Origin"])
    counts = [t["count"] for t in cat.value["top"]]
    assert counts == sorted(counts, reverse=True)
    assert cat.status == "verified" and 0 < cat.evidence.support["rate"] <= 1


def test_spatial_bbox_and_admin_level_from_meta():
    # admin areas need datamart-geo data, which is not installed: feed a meta
    meta = {
        "nb_rows": 3, "nb_profiled_rows": 3, "nb_columns": 3,
        "columns": [
            {"name": "lat", "structural_type": "http://schema.org/Float",
             "semantic_types": ["http://schema.org/latitude"]},
            {"name": "lon", "structural_type": "http://schema.org/Float",
             "semantic_types": ["http://schema.org/longitude"]},
            {"name": "state", "structural_type": "http://schema.org/Text",
             "semantic_types": ["http://schema.org/AdministrativeArea"], "admin_area_level": 1},
        ],
        "spatial_coverage": [{
            "type": "latlong", "column_names": ["lat", "lon"], "number": 3,
            "ranges": [{"range": {"type": "envelope", "coordinates": [[-74.1, 40.9], [-73.7, 40.5]]}},
                       {"range": {"type": "envelope", "coordinates": [[-122.5, 37.8], [-122.3, 37.7]]}}],
        }],
    }
    fs = _factset(None, meta=meta)
    assert fs.validate() == []
    bbox = fs.by_kind("spatial_bbox")[0].value
    assert (bbox["west"], bbox["east"], bbox["north"], bbox["south"]) == (-122.5, -73.7, 40.9, 37.7)
    assert fs.by_kind("spatial_level")[0].value["level"] == "admin_level_1"


def test_toy_frame(toy):
    fs = _factset(toy)
    assert fs.validate() == []
    json.loads(fs.to_json())
    # datamart reports the gaps in the county column
    miss = next(f for f in fs.by_kind("missingness") if f.columns == ["county"])
    assert abs(miss.value["null_rate"] - 30 / 200) < 1e-9
