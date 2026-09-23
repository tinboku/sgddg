# Paper figures

Submission-ready. Every PDF in this directory is **true vector** — checked by
enumerating each page's `/Resources /XObject` for `/Subtype /Image`; none of
them contains a bitmap. They are written straight from Matplotlib's PDF
backend and never passed through a raster stage. The PNGs beside them are
600 dpi previews, not the deliverable.

Rebuild everything:

```bash
python3 figures/src/build_all.py
```

Input: the evaluation artifacts under `out_santos/` and `out_astro2/`.
Output: this directory. No script here re-runs an experiment or recomputes a
metric.

## Main text

| file | size | content |
|---|---|---|
| `fig2_retrieval_main.pdf` | 180.8 × 111.8 mm | A main comparison, B component ablation with same-size controls, C per-query distribution |
| `fig3_group_completion_case_study.pdf` | 180.8 × 62.0 mm | one query end to end: flat retrieval drifts into a neighbouring group, group completion returns the right one |

## Appendix

| file | size | content |
|---|---|---|
| `figA1_retrieval_appendix.pdf` | 180.8 × 167.9 mm | the same three panels with all 24 arms in panel A |
| `figA2_per_benchmark_summary.pdf` | 180.8 × 51.8 mm | absolute F1 per benchmark and pooled |
| `figA3_text_generation_secondary.pdf` | 88.9 × 46.0 mm | what the structure block does to the generated text; the overall difference is 0.000 |

`figA1` shares panels B and C with `fig2`; it is the main figure with panel A
extended, not a second measurement.

## Sources

`src/` holds the generator scripts and one build log each:

```
src/fig_style.py                      names, colours, markers, type scale, export
src/make_fig2_and_figA1_retrieval.py  -> fig2_retrieval_main, figA1_retrieval_appendix
src/make_fig3_case_study.py           -> fig3_group_completion_case_study
src/make_figA2_per_benchmark.py       -> figA2_per_benchmark_summary
src/make_figA3_text_generation.py     -> figA3_text_generation_secondary
src/build_all.py                      runs all of the above and checks the outputs

Earlier drafts are kept locally under `src/superseded/` and are not tracked;
`.gitignore` in this directory excludes them along with LaTeX aux files.
```

`fig_style.NAMES` is the only source of method names; a script that reaches
for an unlisted arm raises rather than inventing a spelling.

## Reading the figures

- Reference arm is **Flat-Adaptive** everywhere. Choosing how many datasets to
  return needs no structure, so **Flat-Fixed-k is an arm, never the
  denominator**.
- Metric is query-macro F1 over the returned set: per query, then averaged.
- Intervals are 95% bootstrap, 1000 resamples, seed 20260921, resampling whole
  (benchmark, parent task) clusters. Test is a paired sign test with ties
  discarded. **A filled marker means both agree; a hollow marker means they
  disagree.**
- Orange square = our methods, blue circle = prior/baseline, grey triangle =
  reference or ceiling. Colour is not the only cue: the palette audit puts the
  blue and the grey within about one unit of each other in grayscale, so
  category is carried by marker shape as well. That describes the design; it
  is not a claim of accessibility compliance.
- Panel C reports 105 better, 74 worse, 66 unchanged, median +0.000. The
  earlier "top 10% carry 90% of the net gain" annotation has been **removed
  from the figure**; the value is still recorded in
  `fig2_retrieval_main.export.json` together with the note that its
  denominator is the sum of all 245 *signed* differences.

## Two names that carry a caveat

- **D3L-Inspired** is our reading of the paper over our own profiles — name,
  format and value signals, **without** the distribution or embedding index.
  **D3L (authors' code)** is the cloned repository at `ce3874b` running four
  of its five indexes; `EmbeddingIndex` needs multi-GB fastText vectors and
  was not built.
- `figA1` omits **Starmie-Inspired**, **Lake Clustering** and **Lake
  Navigation**, which are scored in `discovery_per_query.csv` but were not on
  the requested arm list. Append their ids to `APPENDIX` in the script to
  include them.

## Not included

The method / concept overview figure was dropped. Figure numbering therefore
starts at 2; renumbering is a manuscript decision, not a file one.

## Pending

No venue is fixed, so the widths (IEEE two-column assumed), the minimum font
size and the accepted formats have **not** been checked against any
publisher's live guidance.
