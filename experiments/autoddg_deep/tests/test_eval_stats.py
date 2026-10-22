"""Sanity checks for the NDCG, fusion and paired-test helpers."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("index_eval", "03_index_eval.py")
pt = _load("paired_test", "04_paired_test.py")


def test_ndcg_perfect_and_known_value():
    j = {"a": 2, "b": 1, "c": 0}
    assert ev.ndcg_at_k(["a", "b", "c"], j, 10) == 1.0
    # b first: dcg = 1/log2(2) + 3/log2(3); idcg = 3 + 1/log2(3)
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert math.isclose(ev.ndcg_at_k(["b", "a", "c"], j, 10), expected)
    # cut-off: relevant item below k contributes nothing
    assert ev.ndcg_at_k(["c", "a"], {"a": 1, "c": 0}, 1) == 0.0
    assert math.isnan(ev.ndcg_at_k(["a"], {"a": 0}, 10))


def test_rrf_prefers_consensus():
    assert ev.rrf(["x", "y", "z"], ["y", "x", "z"])[-1] == "z"
    assert ev.rrf(["x", "y"], ["x", "y"]) == ["x", "y"]


def test_bm25_ranks_matching_doc_first():
    docs = {"d1": "county crime statistics", "d2": "school lunch menu", "d3": "river flow"}
    assert ev.rank_bm25("crime by county", sorted(docs), docs)[0] == "d1"


def test_randomization_test_null_and_strong_effect():
    rng = np.random.default_rng(0)
    assert pt.randomization_test(np.zeros(32), 2000, rng) == 1.0
    p = pt.randomization_test(np.full(20, 0.1), 5000, np.random.default_rng(0))
    assert p < 0.001


def test_bootstrap_ci_brackets_mean():
    d = np.random.default_rng(1).normal(0.05, 0.1, size=32)
    lo, hi = pt.bootstrap_ci(d, 4000, np.random.default_rng(2))
    assert lo < d.mean() < hi
