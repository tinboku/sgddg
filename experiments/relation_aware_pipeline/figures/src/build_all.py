"""Rebuild every paper figure.

    python3 figures/src/build_all.py

Input  : the committed evaluation artifacts under out_santos/ and out_astro2/
Output : figures/*.pdf and figures/*.png, plus one .export.json manifest each
         and a build log per figure beside this file.

No script here re-runs an experiment or recomputes a metric.
"""
from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
SCRIPTS = ["make_fig2_and_figA1_retrieval.py", "make_fig3_case_study.py",
           "make_figA2_per_benchmark.py", "make_figA3_text_generation.py"]
EXPECTED = ["fig2_retrieval_main", "fig3_group_completion_case_study",
            "figA1_retrieval_appendix", "figA2_per_benchmark_summary",
            "figA3_text_generation_secondary"]


def main() -> int:
    bad = 0
    for name in SCRIPTS:
        t0 = time.time()
        try:
            runpy.run_path(str(HERE / name), run_name="__main__")
            print(f"OK    {name}  {time.time()-t0:.1f}s")
        except Exception as exc:
            bad += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print()
    for stem in EXPECTED:
        pdf, png = OUT / f"{stem}.pdf", OUT / f"{stem}.png"
        ok = pdf.exists() and png.exists()
        bad += 0 if ok else 1
        print(f"{'present' if ok else 'MISSING':>8}  {stem}"
              f"  {pdf.stat().st_size//1024 if pdf.exists() else 0} KB")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
