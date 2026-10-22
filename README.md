# Second-life battery screening: leakage-aware, pack-grouped evaluation

Analysis code and derived inputs for the article

> Zhang, J.; Wang, S. *Battery Passport and Early Diagnostics for Second-Life
> Lithium-Ion Battery Screening: A Data Leakage-Aware, Pack-Grouped Evaluation.*
> Batteries 2026.

The repository reproduces the pack-grouped screening results, the stopping
replay, the resource–risk frontier, the manuscript figures and the three
supplementary tables from the derived cell-level inputs in `data/`.
Raw public datasets are not redistributed; see [docs/DATA.md](docs/DATA.md) for
their provenance.

## Layout

```
data/                 derived inputs (cell-level tables from the public sources)
src/                  analysis, figure and supplementary-table code
results/              outputs of the current run
results_reference/    outputs of the published run (reference for verification)
figures/              regenerated Figures 1–5 (PDF + PNG)
supplementary/        regenerated Tables S1–S3 (CSV)
docs/                 data provenance and external-validation notes
```

## Quick start

Python 3.12 was used for the published run.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python src/run_decision_consequence_analysis.py   # models, replay, frontier
python src/make_figures.py                        # Figures 1–5
python src/make_supplementary.py                  # Tables S1–S3
python src/verify_results.py                      # checks against the article
```

`verify_results.py` exits with status 0 when every headline number matches the
published value (balanced accuracy 0.686 / 0.761 / 0.862 / 0.970, false
rejections 17 → 6 of 36 eligible cells, independent cohort 0.660 / 0.743 /
0.627) and the regenerated metric table agrees with `results_reference/`.

## What each script produces

| Script | Outputs |
|---|---|
| `src/run_decision_consequence_analysis.py` | `results/salinas_grouped_metrics.csv`, `results/salinas_grouped_predictions.csv`, capacity and regression metrics, `results/salinas_resource_frontier_*.csv`, `results/salinas_pack_yield_audit.csv`, Figure 2 source data |
| `src/make_figures.py` | `figures/fig1_screening_information_timeline.pdf`, `fig2_delayed_capacity_trajectories`, `fig3_screening_performance_progression`, `fig4_screening_resource_risk_frontier`, `fig5_independent_cohort_validation` |
| `src/make_supplementary.py` | `supplementary/table_s1_variable_map.csv`, `table_s2_threshold_sweep.csv`, `table_s3_full_metrics.csv` |
| `src/run_wang_external_validation.py` | independent-cohort metrics (requires the two prepared Wang matrices; see [docs/WANG_VALIDATION.md](docs/WANG_VALIDATION.md)) |
| `src/verify_results.py` | pass/fail report against the published numbers |

## Method summary

Five cumulative information blocks are compared under leave-one-pack-out
validation on 1,034 retired notebook cells from 152 packs: battery history and
passport-compatible fields, physical observations after recovery, early
electrical diagnostics, full electrical characterization, and a delayed storage
measurement. Models are L2 logistic regressions with median imputation and
standardization fitted inside each training fold; the pack yield summary is
excluded because it is outcome-derived. Confidence intervals resample the 16
source packs. The stopping replay withholds a cell from the reference sequence
when its predicted probability falls below a threshold, and the threshold sweep
is reported in full in Table S2.

## Notes and limitations

- The inputs in `data/` are derived tables; the raw workbooks and raw cycling
  files remain at their public repositories (see [docs/DATA.md](docs/DATA.md)).
- The independent-cohort feature matrices are not redistributed. The published
  outputs are included in `results_reference/` and are what Figure 5 and
  Table S3 use; the model script is included for transparency.
- Group-index lookup in the pack bootstrap is precomputed. This is a
  performance-only change: the resampling sequence, the estimates and the
  intervals are unchanged.
- The analysis is a bounded evaluation on public records; it does not observe
  commercial route allocation, processing cost or incoming batches.

## License and citation

Code: MIT (see `LICENSE`). Derived tables in `data/` and `results_reference/`:
CC BY 4.0. Cite the article and this repository; metadata are in
`CITATION.cff`.
