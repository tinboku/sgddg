# Track A — Evidence Utilization (Issue #5)

**Status:** offline screen; agent temporary judgments; no new LLM calls, no API research, subset30 only. The requested Hermes audit files were absent at their stated path except `TRACK_B_CANDIDATE_BANK.md`; the audit bundle used here is the existing repository snapshot at `research/issue4-delivery/hcresearch/research/audit/` (same named artifacts). This path discrepancy is a reproducibility caveat.

## Decision summary

| class | count of distinct screened phenomena | disposition |
|---|---:|---|
| 1. source/profile error | 3 | 2 clear propagation phenomena (CSV shift; trimmed ranges), plus 1 source-semantic ambiguity (implausible/future dates); only the first two are actionable signals |
| 2. selection omission | 2 | Contractor Action and `5+ bd Units`: better/meaningful input was filtered or starved before the 100-word bottleneck |
| 3. LLM did not use reachable evidence | 2 | changed semantic labels reached UFD, but no stable downstream use; no positive quality claim |
| 4. SFD-stage loss | 1 | architectural property, not a separately observed new error: anything omitted by UFD is unrecoverable because SFD receives UFD+topic, not raw/profile/semantic evidence |
| 5. reasonable omission at ~100 words | 1 | six checked columns were recoverable from unchanged sample/header; omission is not automatically failure |

Counts are phenomenon counts, not prevalence estimates. A single case can support more than one class; the matrix below preserves that overlap.

## Updated five-column trace matrix

| raw truth (source CSV) | profile correct? | prompt arrival | UFD use | SFD use / classification |
|---|---|---|---|---|
| `4b5bd984`, `Date Created`: 2019-01-01–2026-10-20 (`applicability_checklist.md`, item 1) | **No**: basic profile says 2019-01-02–2023-03-06; profiler coverage is trimmed, not min/max (`dossiers.json` / flow audit) | Yes, wrong range arrives in both C03 arms | Both C03 arms state wrong span | Copied into SFD; **source/profile error**, not an architectural win |
| `be2a86f4`, trailing delimiter shifts columns; `rank`/names/date fields misaligned | **No**: DataFrame parse is shifted; basic profile and semantic evidence describe neighbours (`REPORT_information_flow_audit.md` §1/§4) | Yes, corrupted profile/sample/semantic inputs arrive | C05 examples include “1,200 unique last names” and cross-column date splice | SFD inherits UFD; **source/parser hygiene**, NO-GO as novelty |
| `Contractor Action` (4b5bd984): 95.2% blank in raw; random 100-row sample yields five blanks | Raw truth yes; baseline semantic input is uninformative; `dist` gives “status tracking” | Improved label reaches UFD prompt (`REPORT_information_flow_audit.md` §4) | No sentence uses it; **selection/starvation + LLM non-use** | No recovery channel; architectural only if selection is coupled to evidence utility, not a label rewrite |
| `5+ bd Units` (19610d2e): 185/190 zero, max 6; rare nonzero values matter | Baseline profile says 0–0; distribution-aware input exposes 0/1/2/6 but semantic summary drops values | Semantic label reaches prompt; values do not | No stable UFD use in S3; **selection/representation loss** | SFD cannot recover it |
| S3 orig/freq/dist: long-tail/category evidence is available in sampled/profile inputs | Profile/sample facts are not inherently wrong; `freq` removes tails, `dist` restores some | Labels change 49–61% vs noise 23–29%; sample values are not directly passed to UFD | Long-tail mentions 0; frequency claims 0; description metrics indistinguishable from noise (`REPORT_s3_sampling.md`) | No evidence of downstream use; **LLM non-use**, negative result |
| C03 six unchanged/control columns: facts visible through sample/header | Correct enough for the checked trace; no demonstrated hidden need | Evidence is present in prompt | Some columns not named, but paraphrase may still be valid; name count is only a proxy | SFD follows UFD; **reasonable omission** under ~100 words unless human blind review says otherwise |
| Any evidence dropped by UFD | N/A | SFD receives only UFD+topic (F2 adds fact slot; baseline does not) | UFD is the gate | SFD cannot reconstruct dropped profile/sample/semantic evidence; **SFD-stage loss mechanism** |

## What is and is not architectural

The strongest repeatable mechanism hypothesis is **budgeted cross-stage evidence gating**: upstream evidence has heterogeneous utility (truth/fidelity, applicability, rarity, units, grain, temporal coverage), but AutoDDG exposes much of it as undifferentiated profile/sample text; the ~100-word UFD is the first irreversible bottleneck, and SFD inherits that bottleneck. A useful intervention would therefore rank/pack evidence by verified user applicability and preserve provenance through UFD→SFD, with an explicit same-budget control. This is a mechanism hypothesis, not a result.

The observed range error and CSV shift are **NO-GO** as thesis mechanisms: they are bug/baseline hygiene. A plain fact slot or more flat fact injection is also **NO-GO** here because it restates F1/F2. S3 sampling-only and the tested C05 representations are negative/neutral and do not justify expansion. The only potentially architectural candidate is utility-aware evidence selection plus provenance-preserving cross-stage handoff; it remains **PLANNED**, not demonstrated.

## Minimal falsifiable experiment (PLANNED)

Freeze the same subset30 tables, model, topic, output budget, and source truth. Compare:

* **B0:** current bug-fixed baseline, with known parser/range contamination excluded or corrected;
* **fair control:** same verified candidate evidence and same token/word budget, but shuffled/flat packing with identical provenance labels;
* **A:** utility-aware selector (applicability checklist + dossier fields: grain, time span, units, category coverage, long-tail pattern, missingness, geography) and a compact provenance-preserving UFD→SFD evidence handoff.

Pre-register a small human blind sheet: factual span/grain/unit/category correctness, user-relevant coverage, unsupported claims, UFD/SFD retention, and tokens. A is supported only if it beats both B0 and fair control on a proximal evidence-use score without increasing unsupported claims/cost; otherwise stop. No generation was run; status is **PLANNED**.

## Prior-work boundary

TAP4LLM supports budgeted table sampling/packing in table reasoning, not dataset-description quality or this UFD→SFD gate. Puduppully is relevant to planning/content selection, but does not establish this AutoDDG evidence path. Doc2Query-style work concerns query/document generation and is not evidence-utilization validation for tabular descriptions. AutoDDG’s own profiler, fixed context assembly, UFD/SFD split, F1/F2 fact work, and the existing S3/C05 screens define the closer boundary. Transfer is a hypothesis only.

## Human-review queue

Agent judgments needing blind human review: whether six “unchanged/recoverable” omissions are user-relevant; whether coded/banded columns (especially `M_RLA00pctprof_0910`) are correctly interpreted; whether future/implausible dates are errors or legitimate planned values; and whether any semantic label was paraphrased rather than literally named. The existing `human_blind_sheet.md` should be used; no answer key is surfaced here.

## Reproducibility pointers

* `research/issue4-delivery/hcresearch/research/audit/dossiers.json`
* `research/issue4-delivery/hcresearch/research/audit/applicability_checklist.md`
* `research/issue4-delivery/hcresearch/research/REPORT_information_flow_audit.md`
* `experiments/autoddg_deep/REPORT_s3_sampling.md`, `REPORT_c05.md`
* `experiments/autoddg_deep/results/sampling/s3_analysis.json`, `results/c05/`, `cache/gen/S3_*`, `cache/gen/C05/`
* source CSVs: `experiments/autoddg_deep/data/tables/<dataset-id>.csv` (subset30 only)

