# C05 frozen table list (frozen 2026-10-22, before any C05 generation)

`c05_tables.txt`: 12 tables from `subset30.txt`. Selection used only the F2/r1
profile cache (`cache/profile/F1`) and the F2/r1 run status; no NTCIR query, qrel or
retrieval result was read.

Profile flags (from the factset): long-tail category (> 50 distinct values),
partial parse / mixed format, sentinel or implausible values, a column >= 50% missing.

Rule (subset30 id order):
1. Eligible: F2/r1 status ok (drops fbb92a2a, context_length_exceeded) and
   4 <= columns <= 30 so a column-by-column audit is feasible (drops 19610d2e 68 cols,
   1bf0f2e9 47, b1efbf4b 228, c4ca6131 2 cols = an API error payload, not a data table).
2. Complex stratum: every eligible table with >= 2 flags, one table per near-duplicate
   schema (be2a86f4 kept, e83c4ac6 same schema dropped):
   10811fc1 1ff67fc2 3e23114a 4b5bd984 682835a9 be2a86f4 c4e2ce61 dcefd0c1 dde84e96 (9).
3. Typical stratum: eligible tables with no flag, first 3 by id:
   14f56ea0 1cf07054 2a208933 (4e4c5dbb 51ea24c0 7b64babd not taken).
4. Tables with exactly one flag (0cbb09dc, 201460a5, 20e29a5b, 2f1aabc5, 53acdac4,
   5f04aaa3, 62acddae, f359e457) are in neither stratum.

`c05_generate.py` re-derives the flags and fails if this list no longer matches the rule.
