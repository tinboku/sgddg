"""Download the CSV tables behind the NTCIR evaluation corpus and write a manifest.

Usage:
    python 00_fetch_tables.py --split test --csv-only --judged-only

Each dataset gets one manifest row:
    dataset_id, status (ok | truncated | not_csv | failed | no_csv_url), url, url_index,
    n_csv_urls, http_status, content_type, bytes, sha256, n_rows, n_cols, error, fetched_at

The downloaded bytes are treated as untrusted data: they are only written to
data/tables/<dataset_id>.csv and parsed with the csv module, never executed.
Datasets whose download fails keep their manifest row so every arm can give
them the same degraded (metadata-only) input.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_NTCIR_DIR,
    MANIFEST,
    TABLES_DIR,
    csv_urls,
    read_jsonl,
    select_benchmark,
    sha256_file,
    write_jsonl,
)

USER_AGENT = "Mozilla/5.0 (research; AutoDDG-Deep table fetcher)"
HTML_PREFIXES = (b"<!doctype", b"<html", b"<?xml", b"<head", b"<!--")
csv.field_size_limit(sys.maxsize)


def count_rows(path: Path) -> tuple[int | None, int | None]:
    """Data rows (excluding header) and header width, via the csv module."""
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            sample = f.read(64 * 1024)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(f, dialect)
            header = next(reader, None)
            n = sum(1 for _ in reader)
        return n, (len(header) if header else 0)
    except Exception:
        return None, None


def looks_like_markup(head: bytes) -> bool:
    stripped = head.lstrip().lower()
    return stripped.startswith(HTML_PREFIXES) or stripped.startswith((b"{", b"["))


def fetch_one(dataset_id: str, urls: list[str], args: argparse.Namespace) -> dict:
    row = {
        "dataset_id": dataset_id,
        "n_csv_urls": len(urls),
        "status": "no_csv_url",
        "url": None,
        "url_index": None,
        "http_status": None,
        "content_type": None,
        "bytes": 0,
        "sha256": None,
        "n_rows": None,
        "n_cols": None,
        "error": None,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    dest = TABLES_DIR / f"{dataset_id}.csv"
    part = dest.with_suffix(".part")
    errors = []
    for idx, url in enumerate(urls[: args.max_urls]):
        row.update(url=url, url_index=idx)
        try:
            with requests.get(
                url,
                stream=True,
                timeout=(args.connect_timeout, args.read_timeout),
                headers={"User-Agent": USER_AGENT},
                allow_redirects=True,
            ) as resp:
                row["http_status"] = resp.status_code
                row["content_type"] = resp.headers.get("Content-Type")
                if resp.status_code != 200:
                    errors.append(f"[{idx}] HTTP {resp.status_code}")
                    continue
                truncated = False
                size = 0
                started = time.time()
                with open(part, "wb") as out:
                    for chunk in resp.iter_content(1 << 16):
                        out.write(chunk)
                        size += len(chunk)
                        if size >= args.max_bytes:
                            truncated = True
                            break
                        if time.time() - started > args.max_seconds:
                            truncated = True
                            break
            with open(part, "rb") as f:
                head = f.read(2048)
            if size == 0:
                errors.append(f"[{idx}] empty body")
                part.unlink(missing_ok=True)
                continue
            if looks_like_markup(head):
                errors.append(f"[{idx}] body is markup/json, not csv")
                row["status"] = "not_csv"
                part.unlink(missing_ok=True)
                continue
            if truncated:
                # keep whole lines only so the prefix parses as a smaller table
                data = part.read_bytes()
                cut = data.rfind(b"\n")
                part.write_bytes(data[: cut + 1] if cut > 0 else data)
            part.replace(dest)
            n_rows, n_cols = count_rows(dest)
            row.update(
                status="truncated" if truncated else "ok",
                bytes=dest.stat().st_size,
                sha256=sha256_file(dest),
                n_rows=n_rows,
                n_cols=n_cols,
                error="; ".join(errors) or None,
            )
            return row
        except Exception as e:  # network errors, SSL, decode, ...
            errors.append(f"[{idx}] {type(e).__name__}: {str(e)[:200]}")
            part.unlink(missing_ok=True)
    if urls and row["status"] != "not_csv":
        row["status"] = "failed"
    row["error"] = "; ".join(errors) or None
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ntcir-dir", type=Path, default=DEFAULT_NTCIR_DIR)
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-relevant", type=int, default=3)
    ap.add_argument("--min-total", type=int, default=1)
    ap.add_argument("--csv-only", action="store_true", help="only datasets with a CSV file")
    ap.add_argument(
        "--judged-only",
        action="store_true",
        help="only the judged pool of the kept queries (always on; kept for CLI parity)",
    )
    ap.add_argument("--refilter-after-csv", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-urls", type=int, default=3, help="CSV urls tried per dataset")
    ap.add_argument("--max-bytes", type=int, default=100 * 1024 * 1024)
    ap.add_argument("--max-seconds", type=float, default=300.0)
    ap.add_argument("--connect-timeout", type=float, default=20.0)
    ap.add_argument("--read-timeout", type=float, default=60.0)
    ap.add_argument("--retry-failed", action="store_true", help="re-fetch failed/not_csv rows")
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N datasets")
    args = ap.parse_args()

    queries, qrels, corpus = select_benchmark(
        args.ntcir_dir,
        args.split,
        args.min_relevant,
        args.min_total,
        csv_only=args.csv_only,
        refilter_after_csv=args.refilter_after_csv,
    )
    ids = sorted(corpus)
    if args.limit:
        ids = ids[: args.limit]
    print(f"queries={len(queries)} corpus={len(ids)} (split={args.split}, csv_only={args.csv_only})")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    previous = {r["dataset_id"]: r for r in read_jsonl(MANIFEST)}
    done_ok = {"ok", "truncated"} | (set() if args.retry_failed else {"failed", "not_csv"})
    rows = {}
    todo = []
    for did in ids:
        prev = previous.get(did)
        if prev and prev["status"] in done_ok:
            path = TABLES_DIR / f"{did}.csv"
            if prev["status"] not in {"ok", "truncated"} or (
                path.exists() and sha256_file(path) == prev["sha256"]
            ):
                rows[did] = prev
                continue
        todo.append(did)
    print(f"cached={len(rows)} to_fetch={len(todo)}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, did, csv_urls(corpus[did]), args): did for did in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            rows[r["dataset_id"]] = r
            print(f"[{i}/{len(todo)}] {r['dataset_id']} {r['status']} {r['bytes']}B {r['error'] or ''}")
            if i % 25 == 0:
                write_jsonl(MANIFEST, [rows[d] for d in sorted(rows)])

    # keep rows for datasets outside the current selection (other splits) untouched
    merged = {**previous, **rows}
    write_jsonl(MANIFEST, [merged[d] for d in sorted(merged)])

    sel = [rows[d] for d in ids]
    counts: dict[str, int] = {}
    for r in sel:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    usable = counts.get("ok", 0) + counts.get("truncated", 0)
    print(f"status counts: {counts}")
    print(f"usable tables: {usable}/{len(sel)} = {usable / max(len(sel), 1):.1%}")


if __name__ == "__main__":
    main()
