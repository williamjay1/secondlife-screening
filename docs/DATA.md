# Data provenance

The analysis reads the derived tables in `data/`. They were assembled from
three public sources; the raw workbooks and raw cycling files are **not**
redistributed here.

## 1. Salinas 2019 notebook-cell collection

- Salinas, F. *Data for: A Second Life for Li-Ion Cells Rescued from Notebook
  Batteries.* Mendeley Data, version 4.
  DOI: [10.17632/kg75m9ydfz.4](https://doi.org/10.17632/kg75m9ydfz.4)
- Files used: the cell workbook (1,034 cells in 152 packs), the
  characterization workbook (100 characterized cells, variables x1–x29) and the
  pack-level battery-management fields.
- Derived files in `data/`: `salinas_characterization.csv`,
  `salinas_performance.csv`, `salinas_1034_linked_audit.csv`,
  `salinas_linked.csv`, `salinas_pack_predecision.csv`.

## 2. Salinas 2021 delayed-performance release

- Salinas, F. *Second Life Assessment of Lithium-Ion Cells Obtained from
  Portable Electronics.* PhD thesis, Technische Universität Berlin, 2021.
  DOI: [10.14279/depositonce-11887](https://doi.org/10.14279/depositonce-11887)
- Characterization data: DOI
  [10.14279/depositonce-11469](https://doi.org/10.14279/depositonce-11469)
- Provides the 100-cycle delayed capacity trajectories used for the
  source-defined eligibility label and the chronology sensitivity.

## 3. Independent cohort (external validation)

- Wang, S.; Gao, F.; Tian, H. *Data for: Deep sorting of reused batteries for
  enabling long-term consistency grouping with unknown prior conditions.*
  Zenodo, 2025. DOI: [10.5281/zenodo.14859405](https://doi.org/10.5281/zenodo.14859405)
  (CC BY 4.0)
- 86 commercial NCM(523)/graphite cells (Lishen LR1865SZ, 2.4 Ah), 16
  first-life protocols, one standardized second-life protocol.

## Derived tables produced by the code

- `salinas_2021_decision_frame.csv` — cell-level analysis frame (written by
  `src/run_decision_consequence_analysis.py`).
- `salinas_2021_trajectory_summary.csv` — delayed capacity trajectory summary.

## Licensing

The code in this repository is released under the MIT license (see `LICENSE`).
The derived tables in `data/` and `results_reference/` are released under
CC BY 4.0. The source datasets remain under their own licenses; consult the
records above before redistributing raw files.
