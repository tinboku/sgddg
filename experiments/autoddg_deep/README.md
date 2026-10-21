# AutoDDG-Deep experiments (M0 + M1)

Plan: `IMPL_claude_profiler.md` (repo root). This directory holds the experiment
scripts; the library code lives in `external/AutoDDG` (own git repo, branch
`autoddg-deep` on top of 3c59e1c; the original package files are untouched).

## Environment

```bash
/opt/homebrew/bin/python3.11 -m venv .venvs/autoddg-deep
.venvs/autoddg-deep/bin/pip install -c experiments/autoddg_deep/constraints_autoddg_3c59e1c.txt \
    -e external/AutoDDG desbordante==2.5.0 datamart-geo pytest
```

The shell profile sets `PYTHONPATH` to a py3.14 site-packages, which breaks the
venv. Run every script with `python -I` (ignores `PYTHONPATH`).

## Evaluation set

`data_search_2_e_test` topics with >= 3 relevant judgments = 32 queries
(`--min-relevant 3 --min-total 1`, the default everywhere). The corpus is the
judged pool of those queries restricted to datasets with a CSV file (483).

## Pipeline

```bash
PY=../../.venvs/autoddg-deep/bin/python
$PY -I 00_fetch_tables.py --split test --csv-only --judged-only      # data/tables, data/manifest.jsonl
$PY -I 02_generate.py --arm configs/arm_baseline.yaml --stages profile   # offline profile stage
$PY -I 02_generate.py --arm configs/arm_baseline.yaml --run-id r1 --provider openai
$PY -I 02_generate.py --arm configs/arm_baseline.yaml --run-id r2 --provider openai   # noise floor
$PY -I 02_generate.py --arm configs/arm_bugfix.yaml   --run-id r1 --provider openai
$PY -I 03_index_eval.py --arms baseline__r1,baseline__r2,bugfix__r1,human --tag m1
$PY -I 04_paired_test.py --eval results/eval/m1.json --a baseline__r1 --b bugfix__r1 --noise-a2 baseline__r2
```

`--provider` defaults to `none` (LLM cache only, never calls an API).
`--dry-run` runs everything with a fake LLM under `<arm>__dryrun`.

Tests: `$PY -I -m pytest tests` here and `$PY -I -m pytest tests` in `external/AutoDDG`.

## Smoke run (2026-10-21)

`OPENAI_API_KEY` is read by `openai.OpenAI()` in `llm_cache.make_inner_client`;
non-interactive shells need `source ~/.zshrc` first. Both arms, `--run-id smoke --limit 5`:
4/5 ok per arm. `01dca46b` (598 columns) fails at the UFD call with
`context_length_exceeded` (gpt-4o-mini, 128k) in both arms. 11 tables have >= 200
columns and 7 tables are unreadable by `load_table`; how to treat them in the full
run is still open.

Full run (baseline x2 noise floor + bugfix):

```bash
source ~/.zshrc
cd experiments/autoddg_deep
PY=../../.venvs/autoddg-deep/bin/python
$PY -I 02_generate.py --arm configs/arm_baseline.yaml --run-id r1 --provider openai 2>&1 | tee logs/gen_baseline_r1.log
$PY -I 02_generate.py --arm configs/arm_baseline.yaml --run-id r2 --provider openai 2>&1 | tee logs/gen_baseline_r2.log
$PY -I 02_generate.py --arm configs/arm_bugfix.yaml   --run-id r1 --provider openai 2>&1 | tee logs/gen_bugfix_r1.log
```

Re-running the same command resumes: finished datasets come from `cache/gen`,
LLM calls from `cache/llm`.
