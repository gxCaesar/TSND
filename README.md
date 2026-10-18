# Temporal Semantic Novelty Decomposition

Research code for **Temporal semantic novelty profiles and observation rules in
biomedical research**, a manuscript prepared for *Scientometrics*.
Authors, in article order: Tongyang Zhang,
Song Yang, Hongbo Dong, Qiujie Lv, and Guanxing Chen. Correspondence: Guanxing Chen,
Department of Computer Science, City University of Hong Kong (Dongguan),
guanxing.chen@cityu-dg.edu.cn. Software version: 0.1.0.

Repository: [gxCaesar/TSND](https://github.com/gxCaesar/TSND). The same code is supplied
with the manuscript as Online Resource 2.

TSND constructs a typed profile of lexical emergence, aligned same-term displacement,
and degree-conditioned pair recombination in scientific concept histories. It preserves
component applicability and missing prior-vector coverage. The journal analyses compare
observation histories, document-frequency thresholds, and document views within a dated
OpenAlex reference. Uptake ranking is a retrospective diagnostic.

This tree contains the implementation, journal analysis entry points, unit tests, an
offline example, and the numerical inputs and drawing code for the article figures.
The supplied figure values support reconstruction of the reported plots. They do not
replace the dated raw corpora, saved embeddings, original labels, or per-front predictions
needed to reproduce the experiments.

## Environment

Clone the repository and create a Python 3.11 environment:

```bash
git clone https://github.com/gxCaesar/TSND.git
cd TSND
```

The offline checks and figure scripts were exercised with Python 3.11.15. The tested
journal dependencies are pinned in `requirements.txt`, including NumPy 1.26.4,
SciPy 1.13.1, Gensim 4.3.3, LightGBM 4.5.0, and Matplotlib 3.7.5.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The dependency file describes the tested analysis packages. It is not a complete lock
of platform-specific transitive dependencies. The tests use one-worker SGNS on tiny
synthetic corpora and small CPU LightGBM fits. No model download or GPU is needed.
Optional noun-chunk, pretrained-encoder, and survival paths have separate extras in
`pyproject.toml`. Their models and environment are not included or exercised by the
offline checks. Reproduction of the historical generic corpus additionally needs its
original language model and extraction environment.

## Offline checks and example

Run these commands from the repository root.

```bash
PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src python examples/offline_profile.py
PYTHONPATH=src python -m tsnd.cli --help
PYTHONPATH=src python scripts/journal_rc_ablation.py --help
```

The example uses synthetic records to show why an entering graph edge can already be
present in raw history. It prints `graph_entry_cause` as `df_threshold`, preserves the
pair lacking prior proximity as undefined, and checks the saved residual identity.
The tests cover annual graph construction, cohort histories, fixed outcome cutoffs,
matched populations, pair-feature groups, weighted average precision, native temporal
invariance, and a deliberately defective future-pooling control. They do not rerun the
reported full-corpus experiments.

Expected example output is JSON containing `input_type: synthetic_example`,
`graph_entry_cause: df_threshold`, two pairs with defined prior proximity, and
`b|c` as the pair without prior proximity. It reports finite lexical/displacement
components and a residual-identity error below `1e-12`. The example is small and
runs on a CPU without downloading data or models.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `src/tsnd/` | Data preparation, temporal measurements, forecasting and conditional sensitivity |
| `configs/` | Original scientific settings, including ten-seed oncology configurations |
| `scripts/` | Runnable journal analyses, native invariance and component-response entry points |
| `examples/` | Synthetic offline profile and observation-history example |
| `tests/` | Deterministic unit and scientific boundary checks |
| `figures/source/` | Editable schematic, plotting scripts and the reported display values |
| `docs/DATA_AND_METHODS.md` | Dated-input schemas, data-file mapping and analysis settings |

## Figure reconstruction

The four data figures read only the supplied CSV and JSON files in `figures/source/`.
They do not read the manuscript source or retrieve data. `reported_tables.json` retains
all reported table values, including tables represented graphically in the article.
`domain_cases.json` contains selected scientific examples and compact graph/history
exports. Network layout coordinates are drawing positions, not semantic distances.

```bash
python figures/source/make_historical_overview.py
python figures/source/plot_observation.py
python figures/source/plot_recombination.py
python figures/source/plot_reference.py
```

These commands write `figures/Fig2.pdf` through `figures/Fig5.pdf` at their declared
170 mm width. The original editable schematic is retained as three TikZ sources.
With `pdflatex`, TikZ, `standalone`, and `newtxtext` available, build it with

```bash
python figures/source/compose_overview.py
```

This writes `figures/Fig1.pdf`. The script also writes the two component PDFs beside
their TikZ sources. The figure inputs contain reported values and selected cases,
not all experimental rows or the complete source corpus.

## Experimental inputs

Full reproduction requires the original dated inputs. The June historical corpus and
the 17 October 2026 reference are separate acquisitions. Repeating an API query later
creates a new snapshot and cannot reconstruct either original input exactly. The
October reference is not an uncapped reconstruction of the June corpus.

The four dated input collections and 590 saved prediction tables are preserved in
the separate manuscript data attachment, Online Resource 1 (`ESM_1.zip`, 610.679 MB).
The archive accompanies the manuscript and is not hosted in this code repository.
For access before the article has a public supplementary-data link, contact the
corresponding author at guanxing.chen@cityu-dg.edu.cn. Within its
`source-data/` directory, `datasets.json` lists the files, row counts, columns and
sizes. `snapshots/` contains oncology (15 June and 17 October), generic (13 June)
and cardiology (16 June) inputs. `predictions/june/` and `predictions/october/`
contain the complete saved test predictions for the journal comparison studies.
`resampling/` preserves the original draw gains and paired seed AP summaries;
its standard-library `replay.py` replays those saved quantities without fitting
or drawing new concept weights. The dated inputs and prediction values were
losslessly recompressed as Parquet without rounding or changing row order.

This code tree does not duplicate the data attachment. Fitted model weights,
trained embeddings, processed graphs, software environments and the original
concept-weight vectors are not included. Their absence does not prevent display
reconstruction or saved-draw replay, but neither operation establishes a fresh
end-to-end reproduction. Regenerating the intermediate artifacts requires the
documented processing, extraction and model settings. No dataset DOI is assigned.
OpenAlex source metadata are CC0; the MIT licence here
covers the software and does not assign rights to author-generated datasets.

See `docs/DATA_AND_METHODS.md` for file schemas and scientific settings. The saved
scientific inputs needed by the journal runners are

| Input | Required content |
| --- | --- |
| Raw works | `works_YEAR.parquet`, including MeSH and outcome venue/subfield fields |
| Annual graphs | `year_YEAR.pkl` with the processed graph and paper-unit records |
| Yearly embeddings | `emb_YEAR_sSEED.npz` and the corresponding alignment artifacts |
| Decompositions | `decomp_YEAR_sSEED.pkl` with LE, NS, RC, proximity, and degrees |
| Frozen outcomes | `labels_YEAR.parquet` with unchanged IDs, breadth, cutoff, and labels |
| Classified cohorts | `classified_CONDITION_YEAR.parquet` with raw and graph histories |
| Common IDs | `year_YEAR.parquet` carrying the shared `t`, `kind`, and `key` values |

Pickle inputs are executable Python serialization and must come from a trusted source.
Use fresh output directories. The analysis entry points reject existing scientific outputs.

## Journal entry points

The following commands assume the original inputs have been placed in the relative
directories shown. They describe the complete analysis settings, not an executed
full-corpus rerun. All commands run from this repository root.

Rebuild a frequency-threshold input chain from the frozen retained works. Repeat with
DF 30, changing both the condition name and output directory. DF 20 is the saved reference.

```bash
PYTHONPATH=src python scripts/journal_prepare_inputs.py \
  --config configs/mesh_onco_10seed.yaml --data-dir data/frozen_oncology \
  --output-dir results/df10 --condition df10 --min-df 10 --year-end 2016 \
  --seeds 0 1 2 3 4 5 6 7 8 9

PYTHONPATH=src python scripts/journal_cohort_sensitivity.py \
  --data-dir data/frozen_oncology --terms-dir artifacts/df20/terms \
  --condition df10=results/df10/terms --condition df30=results/df30/terms \
  --labels-dir artifacts/frozen_labels --output-dir results/classified \
  --years 2011 2012 2013 2014 2015 2016
```

The classifier writes per-year cohort Parquets and `summary.csv`/`summary.json`, with
both raw-history and graph-entry classifications. It leaves original labels unchanged.
Use `tsnd.journal.fixed_outcomes.common_membership` to construct masks after selecting
the requested history class. Intersect both IDs and membership flags across conditions.

Run the explicit pair-only feature comparisons. The example below uses the adjacent
population. Supply the classified cohort directory for graph-first or raw-first
membership. A common-ID mask is necessary for matched threshold or corpus-view contrasts.

```bash
PYTHONPATH=src python scripts/journal_rc_ablation.py \
  --config configs/mesh_onco_10seed.yaml --data-dir data/frozen_oncology \
  --terms-dir artifacts/df20/terms --embed-dir artifacts/df20/embed \
  --decomp-dir artifacts/df20/decompose --labels-dir artifacts/frozen_labels \
  --condition df20 --cohort adjacent --cohort-condition df20 \
  --cohort-dir results/classified --common-ids-dir artifacts/common_adjacent \
  --train-years 2011..2015 --test-year 2016 --seeds 0..9 \
  --output-dir results/df20_adjacent
```

This writes `predictions_sSEED.parquet` and `summary.json`. The seven groups are S,
S+P, S+Z, S+R, S+P+Z, S+P+R, and S+P+Z+R. S is the original B2+B7 structural
feature union. P is prior cosine proximity. Z is the current log degree product.
R is the saved residual fitted before cohort filtering. `--legacy-mixed` additionally
reproduces the original mixed term/pair comparison with its original feature order.

After every requested fit has completed, summarize fixed-prediction dependence jointly.
Include all conditions, cohorts, views, and seeds being compared in this one invocation.

```bash
PYTHONPATH=src python scripts/journal_rc_ablation.py \
  --prediction-files results/df20_adjacent/predictions_s*.parquet \
  --output-dir results/joint_sensitivity --resamples 2000 --uncertainty-seed 20261017
```

This writes `dependence_sensitivity.json`. The primary effect is the mean of seed-paired
average-precision differences. Seed SD and conditional dependence-sensitivity intervals
remain separate quantities. The latter do not establish population-coverage guarantees.

Run the native dated-input check with a fixed launch-time Python hash seed.

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python scripts/journal_native_a9.py \
  --config configs/mesh_onco_10seed.yaml --data-dir data/frozen_oncology \
  --output-dir results/native_a9 --year 2016 --seeds 0 1 2 3 4 5 6 7 8 9
```

This rebuilds the isolated prior-only and future-exposed inputs using one-worker SGNS,
checks direct numerical agreement, and tests a deliberately defective future-pooling
control. Invariance within a dated snapshot does not establish historical metadata access.

## Component-response criteria

The existing `scripts/a10_strict.py` preserves the original perturbations and RNG
order while computing response-only, strict, and dominance criteria. Both relabeling
and drift criteria change. The novel-pair criterion stays unchanged because its
partner swap leaves the other term components fixed.

With the original saved artifacts available, run

```bash
PYTHONPATH=src python scripts/a10_strict.py artifacts/df20 2016 0,1,2
```

It prints the aggregate and per-seed JSON to standard output. Its legacy field names
include `identifiability`, but the scores measure component response and attribution.
They do not establish unique independent factors. The script needs the original
`terms/`, `embed/`, `align/`, and `decompose/` directories. Running it on the supplied
plot-value files cannot reproduce the experiment.

## Same-snapshot reference views

`scripts/journal_collect_reference.py` retains the acquisition implementation. It requires
the caller's authorized API access and retrieves a newly dated reference when run.
It is not part of the offline checks and cannot recreate an earlier API snapshot.

For an original complete reference already on disk, call
`tsnd.journal.corpus_views.materialize_corpus_views` with years 2010–2019, cap 15,000,
sampling seeds 0–9, and `selection_year_end=2016`. This constructs full, API-order,
and ten random feature/history views. Build each view's inputs with the preparation
entry point above. Use `tsnd.journal.fixed_outcomes.build_fixed_reference_outcomes`
with the full DF20 graphs and full raw records to assign outcome breadth and the single
annual reference cutoff. Outcome graph construction must cover every forward window
through 2019, including 2012–2018 for training cohorts. It must not use sampled future
records. The function writes shared outcome labels and view-specific candidate tables.
Classify each view, intersect the requested memberships with `common_membership`, then
run the same feature comparisons and joint sensitivity analysis.

## Scope and reuse

TSND measures corpus-relative profiles. Graph entry does not establish first conceptual
invention. Component-response diagnostics do not prove unique factor identification.
The uptake analysis uses retrospective labels and is not an as-of-2016 deployment test.
Conditional seed-paired comparisons do not establish universal predictive superiority.

The existing software licence is reproduced in `LICENSE`. Data sources and third-party
models retain their applicable terms. The software licence does not establish permission
to redistribute every corpus or model.

## Citation

The accompanying manuscript can be cited provisionally as

```bibtex
@unpublished{zhang2026tsnd,
  title = {Temporal semantic novelty profiles and observation rules in biomedical research},
  author = {Zhang, Tongyang and Yang, Song and Dong, Hongbo and Lv, Qiujie and Chen, Guanxing},
  year = {2026},
  note = {Manuscript. Research code version 0.1.0},
  url = {https://github.com/gxCaesar/TSND}
}
```
