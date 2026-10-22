"""Shared frames for the profiling_v2 fact-module tests.

Real tables come from subset30 under data/tables (tests skip if a file is
missing). The toy frame packs the edge cases the fact layer must handle:
sentinel 9999 years, mixed date formats, a categorical column with gaps,
a '$' header and a 0-100 score without any unit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from tables import load_table  # noqa: E402

TABLE_DIR = HERE / "data" / "tables"

# small subset30 tables with temporal / categorical / unit columns
REAL_TABLES = {
    "birth_rates": "2a3cc123-f9f6-4e1d-a03f-3e33056e5021",  # Year, Age Group, Birth Rate
    "lobbyists": "dde84e96-8559-4a9c-9b96-d349f069dd65",  # 4 date columns, gaps
    "infant_mortality": "0cbb09dc-4012-4f18-900e-6d746bfc10d3",  # "2023 Q1", Unit column
}


def _real(key: str) -> pd.DataFrame:
    path = TABLE_DIR / f"{REAL_TABLES[key]}.csv"
    if not path.exists():
        pytest.skip(f"{path.name} not downloaded")
    return load_table(path)[0]


@pytest.fixture(scope="session", params=sorted(REAL_TABLES))
def real_table(request) -> tuple[str, pd.DataFrame]:
    return REAL_TABLES[request.param], _real(request.param)


@pytest.fixture(scope="session")
def birth_rates() -> pd.DataFrame:
    return _real("birth_rates")


@pytest.fixture(scope="session")
def lobbyists() -> pd.DataFrame:
    return _real("lobbyists")


@pytest.fixture(scope="session")
def infant_mortality() -> pd.DataFrame:
    return _real("infant_mortality")


def make_toy() -> pd.DataFrame:
    """200 rows: years 2001-2020 with three 9999 sentinels and one 1900."""
    rng = np.random.default_rng(0)
    n = 200
    years = np.repeat(np.arange(2001, 2021), 10)
    years[[3, 50, 120]] = 9999
    years[77] = 1900
    # one date per month in 2019-2020 in three formats
    months = pd.date_range("2019-01-01", periods=24, freq="MS")
    fmts = ["%Y-%m-%d", "%m/%d/%Y", "%d %b %Y"]
    dates = [months[i % 24].strftime(fmts[i % 3]) for i in range(n)]
    dates[5] = "not a date"
    county = rng.choice(["Kings", "Queens", "Bronx", "Erie"], size=n).astype(object)
    county[rng.choice(n, 30, replace=False)] = None
    return pd.DataFrame({
        "year": years,
        "report_date": dates,
        "county": county,
        "Amount ($)": rng.integers(0, 5000, n),
        "price": [f"${v:,}" for v in rng.integers(1, 3000, n)],
        "score": rng.integers(0, 101, n),
        "zip": [f"{v:05d}" for v in rng.integers(10001, 10300, n)],
    })


@pytest.fixture(scope="session")
def toy() -> pd.DataFrame:
    return make_toy()
