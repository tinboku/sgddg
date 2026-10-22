"""Coverage-range bug: autoddg.profiling.summary vs autoddg.profiling_v2.summary_fixed.

The positive / negative / cross-zero cases run the real datamart_profiler on
small frames. datamart_profiler never emits an empty coverage list for a frame,
so the empty case wraps the real profiler and overwrites the coverage field.
"""

from __future__ import annotations

import datamart_profiler
import pandas as pd
import pytest

from autoddg.profiling.summary import profile_dataset as profile_original
from autoddg.profiling_v2.summary_fixed import profile_dataset as profile_fixed


def _column_line(profile_text: str, name: str) -> str:
    return next(line for line in profile_text.splitlines() if line.startswith(f"**{name}**"))


def _both(df: pd.DataFrame, name: str) -> tuple[str, str]:
    return _column_line(profile_original(df)[0], name), _column_line(profile_fixed(df)[0], name)


def test_all_positive_range():
    original, fixed = _both(pd.DataFrame({"v": [5, 6, 7, 8, 9, 10] * 5}), "v")
    assert original.endswith("Coverage spans from 0 to 10.0. ")
    assert fixed.endswith("Coverage spans from 5.0 to 10.0. ")


def test_all_negative_range():
    original, fixed = _both(pd.DataFrame({"v": [-10, -9, -8, -7, -6, -5] * 5}), "v")
    assert original.endswith("Coverage spans from -10.0 to 0. ")
    assert fixed.endswith("Coverage spans from -10.0 to -5.0. ")


def test_cross_zero_range():
    # 0 lies inside the true range, so the original is right here and both agree
    original, fixed = _both(pd.DataFrame({"v": [-3, -2, -1, 0, 2, 4, 7] * 5}), "v")
    assert original.endswith("Coverage spans from -3.0 to 7.0. ")
    assert fixed == original


@pytest.mark.parametrize("coverage", [[], [{"range": {}}]], ids=["empty_list", "no_range_bounds"])
def test_empty_coverage(monkeypatch, coverage):
    real_process = datamart_profiler.process_dataset

    def process_with_coverage(df):
        metadata = real_process(df)
        metadata["columns"][0]["coverage"] = coverage
        return metadata

    monkeypatch.setattr(datamart_profiler, "process_dataset", process_with_coverage)
    original, fixed = _both(pd.DataFrame({"v": [1.5, 2.5, 3.5] * 10}), "v")
    assert original.endswith("Coverage spans from 0 to 0. ")
    assert "Coverage spans" not in fixed
    assert fixed == original.replace("Coverage spans from 0 to 0. ", "")


def test_only_coverage_text_differs():
    df = pd.DataFrame(
        {
            "pos": [5, 6, 7, 8, 9, 10] * 5,
            "neg": [-10, -9, -8, -7, -6, -5] * 5,
            "name": ["a", "b", "c"] * 10,
            "date": pd.date_range("2015-01-01", periods=30, freq="D").astype(str),
        }
    )
    original_profile, original_semantic = profile_original(df)
    fixed_profile, fixed_semantic = profile_fixed(df)
    assert original_semantic == fixed_semantic
    strip = lambda line: line.split("Coverage spans")[0]  # noqa: E731
    assert list(map(strip, original_profile.splitlines())) == list(
        map(strip, fixed_profile.splitlines())
    )
    assert original_profile != fixed_profile
