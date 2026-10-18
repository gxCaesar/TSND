# Data interfaces and analysis settings

## Dated inputs

The historical oncology records were retrieved on 15 June 2026 with an annual cap of
15,000 retained works. The query selected primary-topic subfield 2730, with a PubMed
identifier and indexed abstract. The historical cardiology acquisition used subfield
2705 on 16 June 2026. The generic acquisition on 13 June 2026 used subfields 1702,
3104, and 2204. Historical membership counts need not count unique works across corpora.

The new oncology reference was acquired on 17 October 2026 for publication years
2010–2019. Its 163,616 records are a separately dated query result. Its full, API-order,
and ten random views vary feature/history observations through 2016 while sharing
the full-reference outcome observations. All three-year forward windows, including
those of the training cohorts, use the full reference.

Raw records, acquisition order, and preprocessing versions are needed for exact
reproduction. A new query cannot establish equality with a retained historical snapshot.
The small files supplied with the figure code are derived plotting inputs.

The companion `ESM_1.zip` supplies the dated raw inputs and saved predictions.
After extraction, its root is `source-data/`:

| Location | Included material |
| --- | --- |
| `snapshots/oncology_20260615/` | 15 annual Parquet tables, 2010–2024, 222,237 retained memberships |
| `snapshots/generic_20260613/` | 15 annual Parquet tables, 2010–2024, 619,554 retained memberships |
| `snapshots/cardiology_20260616/` | 10 annual Parquet tables, 2010–2019, 145,878 retained memberships |
| `snapshots/oncology_20261017/` | 10 annual Parquet tables, 2010–2019, 163,616 unique records |
| `predictions/june/` | 110 saved prediction tables for history, threshold and recombination comparisons |
| `predictions/october/` | 480 saved annotated prediction tables for the twelve reference views |
| `datasets.json` | File inventory, row counts, columns and uncompressed Parquet sizes |
| `resampling/` | All original mean draw gains and paired model-seed AP summaries |
| `replay.py` | Standard-library replay of saved means, SDs, quantiles and direct contrasts |

For example, set a runner's `--data-dir` to the extracted
`source-data/snapshots/oncology_20260615` for the historical oncology input.
These saved collections retain their distinct acquisition dates. Membership
totals must not be added as though they were unique records across collections.
Only the October snapshot includes `acquisition_rank`; June inputs preserve
their saved row order. Lossless Parquet recompression preserves all values,
column types, Arrow schema metadata and row order. The saved-draw replay does
not regenerate the original concept weights or recompute weighted AP from the
full predictions. The latter can be analysed with the supplied journal code.

## Raw works and scientific artifacts

Each raw input is named `works_YEAR.parquet`. The journal MeSH path uses these fields.

| Field | Meaning |
| --- | --- |
| `id` | Work identifier |
| `year` | Publication year |
| `mesh` | List of concept descriptors assigned to the record |
| `venue_id` | Venue identifier for uptake breadth |
| `subfield_ids` | List of subfield identifiers for uptake breadth |
| `acquisition_rank` | Nonnegative within-year acquisition order for reference views |

The historical generic text path requires `abstract` and `concepts`.
A MeSH-only example does not validate that extraction environment.

Annual `year_YEAR.pkl` graphs contain `vocab`, `uidx`, `unit_df`, `paper_units`,
`edges`, and `n_papers`. Edges refer to vocabulary indices. The function named
`weighted_degree` returns distinct-partner degree. It does not sum edge weights.
Embedding NPZ files contain `units` and `mat`. Decomposition pickles contain typed
LE/NS dictionaries, degrees, and a `RecombinationResult` with RC, proximity, linear
coefficients, intercept, and defined-proximity count.

Outcome tables retain `t`, `kind`, `key`, `breadth`, `thr`, and binary `label`.
Pair keys are canonicalized as `u|v`. Missing or conflicting labels fail validation.
A missing label is never replaced by a negative. Long-form prediction tables retain
the same IDs, condition, cohort, analysis, SGNS seed, model group, predicted score,
label, and Boolean `prox_defined` flag.

## Shared scientific settings

The journal comparisons evaluate 2016 after training on 2011–2015 cohorts. Historical
observation starts in 2010. The uptake horizon is three years. The original ten SGNS
seeds are 0–9. The forecaster and training/calibration partition remain fixed. The
2016 outcomes do not select features or parameters. These are retrospective analyses,
because some training-label windows mature after the evaluation year.

MeSH graphs remove check tags and use at most 40 units per paper. The annual graph DF
threshold is 10, 20, or 30. SGNS `min_count` stays at 20 across these thresholds, so
prior-vector availability remains a separate criterion. Reference configuration files
retain the scientific values and relative artifact locations. Historical CLI defaults
do not replace the explicit journal train/test years.

Raw history is constructed before annual DF filtering and per-paper truncation.
Raw first/last observation, recurrence gap, and graph-entry cause are separate fields.
Adjacent, graph-first, and raw-first membership preserve the relevant frozen labels.
Matched comparisons intersect both IDs and requested membership across every
training and test year. Full candidate composition and matched subsets have different
denominators and can have different prevalence.

For the current reference, the full DF20 adjacent mixed-front cohort sets one annual
top-decile breadth cutoff. All views use that full-reference outcome definition.
Sampling and SGNS seeds are separate sources of variation. Recombination is fitted on
the original adjacent cohort before filtering. Native missing values are retained,
and defined-proximity analyses use their common eligible subset.

S is the exact original B2+B7 feature union. P is prior cosine proximity. Z is
`log(deg_t(u) * deg_t(v) + 1)`. R is the saved RC residual. R is deterministic given
P, Z, and its cohort/seed fit, so adding R after P+Z tests the learner representation.
NS is a term-displacement measurement and is not a forecast feature here.

Average precision uses `sklearn.metrics.average_precision_score`. The primary effect
averages paired seed-level differences rather than averaging probabilities first.
Conditional dependence sensitivity draws independent Exp(rate=1) concept multipliers,
uses their endpoint product on pair rows, and shares each draw across models, seeds,
cohorts, and views. The registered settings are 2,000 draws, analysis RNG seed 20261017,
and 2.5–97.5 percentile bounds. Those bounds are conditional on fixed predictions.

The native temporal check fixes `PYTHONHASHSEED=0` before process launch, uses one SGNS
worker, compares graphs/vectors/profiles directly, and exercises a known future-pooling
defect. This check addresses future-dated records within a snapshot. It does not
reconstruct historical annotation availability or prove unique component identification.
