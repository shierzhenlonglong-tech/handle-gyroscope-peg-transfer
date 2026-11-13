# handle-gyroscope-peg-transfer

Analysis code and derived trial-level tables for:

> Liang Wang. *Beyond completion time: responsiveness, reliability and external consistency of tool-handle gyroscope metrics in laparoscopic peg-transfer training.* Manuscript in preparation.

The study examines whether angular-velocity features from tool-handle gyroscopes in laparoscopic peg-transfer practice change with training (responsiveness), remain informative after controlling for completion time, show usable reliability, and are externally consistent with an independent instrument-tracking cohort.

This repository does **not** contain raw motion recordings, official public-dataset archives, ethics paperwork, or consent forms.

## Layout (flat)

All analysis scripts and derived tables sit in the **repository root**. Run the scripts from this directory (the folder that contains `h1_mixed_models.py` and `trial_features.csv`). Output is written to `results/` and `figures/`, which are created as needed.

| Item | Role |
| --- | --- |
| `extract_features.py`, `build_trial_table.py` | Build trial-level features from a local SPD-FLS1 tree (optional if you use the derived tables below). |
| `h1_mixed_models.py`, `h1_robustness.py` | Primary mixed models and pre-specified robustness analyses. |
| `h2_prognostic.py` | Exploratory prognostic models. |
| `extra_reliability_retention.py` | Post-hoc reliability, between-session change and second-half slopes. |
| `h3_lask.py` | External-consistency analysis (requires a local LASK extract). |
| `make_figures_v2.py`, `make_figures.py` | Manuscript figures (v2) and earlier figures used as supplements. |
| `feasibility_check_20261110.py` | Pre-plan exploratory check (known limitations are in its docstring). |
| `trial_index.csv`, `trial_features.csv` and `_deg` / `_n0165` / `_n05` variants | Derived trial-level tables used in the models. |
| `features_part{1,2,3}.csv` and tagged variants | Per-subset extracts concatenated by `build_trial_table.py`. |
| `analysis_plan_frozen_20261110.md` | Frozen analysis plan (Chinese original, dated 10 November 2026). |
| `analysis_plan_frozen_20261110_EN.md` | English translation; the Chinese original is authoritative if the two texts differ. |
| `extract_windowed.py`, `a1_windowed_smoothness.py`, `a1_lask_windowed.py` | Supplementary analysis A1 (post hoc): windowed SPARC / LDLJ in SPD-FLS1 and LASK. |
| `extract_wrist.py`, `a2_wrist_vs_handle.py` | Supplementary analysis A2 (post hoc): wrist-IMU replication of the handle metrics and handle–wrist comparison. |
| `extract_wrist_bimanual.py`, `b1_bimanual.py` | Supplementary analysis B1 (post hoc): bimanual coordination from the two synchronized wrist sensors. |
| `b2_coupling_check.py`, `b2_followup.py` | Supplementary analysis B2 (post hoc): handle/wrist speed ratio as a sensor-coupling check, and follow-up checks. |
| `make_figures_v3.py` | Manuscript Figure 7 (wrist sensors: coupling check, handle vs wrist practice effects, bimanual movement). |
| `trial_features_windowed.csv`, `wrist_features.csv`, `wrist_bimanual.csv` | Derived tables for the supplementary analyses: hand-trial table with windowed smoothness columns; wrist sensor-trial features (passing quality control and 0.9–1.2 duration coverage, with the sensor-to-hand assignment); one row per trial with both wrists available (bimanual metrics). |
| `a1_*.csv`, `a2_*.csv`, `b1_*.csv`, `b2_*.csv` | Result tables of the supplementary analyses as reported in the manuscript (reference copies; rerunning the scripts writes fresh copies to `results/`). Files ending in `_all` are the results **before** the wrist 5 °/s exclusion rule (see below). |
| `fig7_wrist.pdf` | Figure 7 as used in the manuscript (reference copy; `make_figures_v3.py` writes `figures/fig7_wrist.pdf`). |
| `analysis_plan_supplement_A_20261111_ZH.md`, `analysis_plan_supplement_B_20261111_ZH.md` | Supplementary analysis plans A (windowed smoothness and wrist IMU) and B (bimanual coordination and sensor consistency), Chinese originals dated 11 November 2026, each written before the related computations, with the results records appended after running. |
| `analysis_plan_supplement_A_windowed_smoothness_wrist_IMU_20261111_EN.md`, `analysis_plan_supplement_B_bimanual_sensor_consistency_20261111_EN.md` | English translations of supplements A and B; the Chinese originals are authoritative. |
| `requirements.txt` | Third-party Python packages inferred from script imports. |

## Scripts from a related study

`mp3_gum.py` and `oh2026_qc_sensitivity.py` belong to a **separate study in preparation** by the same author on the same public SPD-FLS1 data (measurement uncertainty of an instrumented handle). They are kept in this repository but are **not required to reproduce the results of the present paper**. `extract_features.py` imports quality-control helpers from `oh2026_qc_sensitivity.py`. Both scripts are written to run from this repository root:

```bash
python mp3_gum.py selftest
python oh2026_qc_sensitivity.py selftest
```

Those self-tests use synthetic fixtures only. To run the quality-control pipeline on the public dataset:

```bash
python oh2026_qc_sensitivity.py run --root /path/to/SPD-FLS1 --out ./oh2026_out
```

## Requirements

Python 3 with:

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `statsmodels`, `scikit-learn`.

## Data sources (not included)

Download these official releases yourself; do not substitute unofficial copies.

1. **SPD-FLS1 v2** (primary cohort; handle IMU and trial logs, plus the Shimmer wrist IMU files for the supplementary wrist analyses). Oh, Deng & Lau (2026), Virginia Tech Data Repository, DOI [10.7294/29421842](https://doi.org/10.7294/29421842) (CC0). Needed only to rebuild features with `extract_features.py`, `extract_windowed.py`, `extract_wrist.py` or `extract_wrist_bimanual.py`, or for example traces in `make_figures_v2.py`.
2. **LASK v1.0** (external-consistency cohort; 37 public trials). Official release: [https://doi.org/10.5281/zenodo.20752651](https://doi.org/10.5281/zenodo.20752651) (CC BY 4.0). Needed only to rerun `h3_lask.py` and `a1_lask_windowed.py`. Pass the extracted directory (containing `participants.csv` and kinematics) as the script argument.

The wrist-sensor analyses (A2, B1, B2) use the Shimmer wrist IMU files distributed with SPD-FLS1. These raw files are **not** included here (about 7.3 GB); they are needed only to rerun `extract_wrist.py` and `extract_wrist_bimanual.py`.

Participant codes such as `P01` are the public identifiers published with SPD-FLS1, not real names.

## How to run

From the repository root:

```bash
mkdir -p results figures
# Feature extraction (optional if you use the provided derived tables):
# python extract_features.py /path/to/spd-fls1 features_part1.csv --only part1
# python build_trial_table.py            # also: _deg, _n0165, _n05

python h1_mixed_models.py
python h1_robustness.py
python h2_prognostic.py
python extra_reliability_retention.py
python h3_lask.py /path/to/LASK/extracted
python make_figures_v2.py [/path/to/spd-fls1]
```

### Supplementary analyses (A1, A2, B1, B2; post hoc)

These run from the derived tables in this repository. Run `h1_mixed_models.py` first (it writes `results/h1_primary.csv`, which `make_figures_v3.py` reads), and run `a2_wrist_vs_handle.py` before `make_figures_v3.py`.

```bash
python a1_windowed_smoothness.py        # -> results/a1_windowed.csv
python a2_wrist_vs_handle.py            # -> results/a2_device_mapping.csv, a2_wrist_models.csv, a2_handle_vs_wrist.csv
A2_KEEP_ALL=1 python a2_wrist_vs_handle.py   # -> results/a2_wrist_models_all.csv, a2_handle_vs_wrist_all.csv
python b1_bimanual.py                   # -> results/b1_bimanual.csv
B1_KEEP_ALL=1 python b1_bimanual.py     # -> results/b1_bimanual_all.csv
python b2_coupling_check.py             # -> results/b2_rule_summary.csv, b2_flagged.csv, b2_rho_by_participant.csv
python b2_followup.py                   # -> results/b2_wrist_models_excl_lowwrist.csv, b2_bimanual_excl_lowwrist.csv
python make_figures_v3.py               # -> figures/fig7_wrist.pdf, figures/fig7_wrist.png
python a1_lask_windowed.py /path/to/LASK/extracted   # needs LASK; run h3_lask.py first -> results/a1_lask_windowed.csv
```

On Windows (cmd), set the variable first, for example `set A2_KEEP_ALL=1` and then `python a2_wrist_vs_handle.py`; clear it afterwards with `set A2_KEEP_ALL=`.

**The `_all` suffix.** After the sensor-coupling check (B2), wrist records with mean angular speed below 5 °/s were excluded by the same rule as deviation D1 for the handle (21 wrist records: P16 session 1, both wrists, and one right-wrist record of P21). The reported A2 and B1 results use this exclusion. Setting `A2_KEEP_ALL=1` or `B1_KEEP_ALL=1` reproduces the results before the exclusion, written to files ending in `_all`. `wrist_features.csv` keeps all wrist records so that B2 can use them. The released `b1_bimanual_all.csv` was regenerated with the current `b1_bimanual.py` and therefore includes the exploratory `bi_excess` row.

**Which inputs need raw data.**

- No raw data: `a1_windowed_smoothness.py`, `a2_wrist_vs_handle.py`, `b1_bimanual.py`, `b2_coupling_check.py`, `b2_followup.py`, `make_figures_v3.py` (they read `trial_features.csv`, `trial_index.csv`, `trial_features_windowed.csv`, `wrist_features.csv` and `wrist_bimanual.csv`).
- SPD-FLS1 handle IMU and logs: `extract_windowed.py` (writes `features_w_part{1,2,3}.csv`).
- SPD-FLS1 Shimmer wrist files: `extract_wrist.py` (writes `wrist_part{0..11}.csv`) and `extract_wrist_bimanual.py` (writes `wristbi_part{0..11}.csv`). Their usage lines are in the script docstrings.
- LASK: `a1_lask_windowed.py` (and `h3_lask.py`).

If the per-chunk extraction outputs (`features_w_part*.csv`, `wrist_part*.csv`, `wristbi_part*.csv`) are present in the repository root, `a1_windowed_smoothness.py`, `a2_wrist_vs_handle.py` and `b1_bimanual.py` rebuild `trial_features_windowed.csv`, `wrist_features.csv` and `wrist_bimanual.csv` from them; otherwise they read the merged tables shipped here. The per-chunk files are not included because the merged tables contain their content.

Random seeds used in the scripts are fixed (for example `20261110` in `h2_prognostic.py` and `20261111` in `a1_lask_windowed.py`).

Versioned snapshots of this repository are archived on Zenodo. All versions: [https://doi.org/10.5281/zenodo.24123988](https://doi.org/10.5281/zenodo.24123988) (always resolves to the latest release). Please cite the specific version used; see `ZENODO_GITHUB_ARCHIVE.md`.

## Citation

Wang, L. *Beyond completion time: responsiveness, reliability and external consistency of tool-handle gyroscope metrics in laparoscopic peg-transfer training.* Manuscript in preparation.

Please also cite the source datasets above.

This code is released under the MIT License (see `LICENSE` in the GitHub repository).

## Changelog

- **v1.1.0** (13 November 2026). Adds the post hoc supplementary analyses reported in the manuscript: windowed SPARC / LDLJ smoothness (A1, including LASK); replication of the handle metrics on the SPD-FLS1 wrist IMUs and handle–wrist comparison (A2); bimanual coordination from the two synchronized wrist sensors (B1); the handle/wrist speed ratio as a sensor-coupling check, with follow-up checks (B2); and Figure 7 (`make_figures_v3.py`). Includes the derived tables `trial_features_windowed.csv`, `wrist_features.csv` and `wrist_bimanual.csv`, the result tables `a1_*`, `a2_*`, `b1_*`, `b2_*` (with `_all` versions before the wrist 5 °/s exclusion) and `fig7_wrist.pdf`. Supplementary analysis plans A and B are included in the Chinese originals with English translations. Existing scripts, tables and results are unchanged.
- **v1.0.1** (11 November 2026). `h1_mixed_models.py`, `h1_robustness.py`: a mixed-model fit is now retained only if it is admissible (optimiser converged, finite standard errors, positive-definite random-effects covariance); previously the fit with the highest restricted likelihood was taken without this check, which could select a degenerate boundary solution for some secondary metrics and crash when random effects were predicted. All primary and robustness estimates are unchanged; three rows of the secondary-metric table change in the second decimal. All analysis and figure scripts now create `results/` and `figures/` if they do not exist.
- **v1.0.0** (11 November 2026). First archived release.
