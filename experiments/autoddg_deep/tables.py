"""Read downloaded CSVs into DataFrames with the same settings for every arm."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def sniff_delimiter(path: Path, encoding: str) -> str:
    with open(path, encoding=encoding, errors="replace", newline="") as f:
        sample = f.read(64 * 1024)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_table(path: Path, max_rows: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (frame, read_info). Raises ValueError if no encoding parses."""
    last_error = None
    for encoding in ENCODINGS:
        try:
            sep = sniff_delimiter(path, encoding)
            df = pd.read_csv(
                path,
                sep=sep,
                encoding=encoding,
                nrows=max_rows,
                low_memory=False,
                on_bad_lines="skip",
            )
            info = {
                "encoding": encoding,
                "sep": sep,
                "n_rows_read": int(len(df)),
                "n_cols": int(df.shape[1]),
                "max_rows": max_rows,
                "read_status": "truncated" if max_rows and len(df) >= max_rows else "ok",
            }
            return df, info
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            last_error = e
    raise ValueError(f"unreadable csv: {type(last_error).__name__}: {last_error}")
