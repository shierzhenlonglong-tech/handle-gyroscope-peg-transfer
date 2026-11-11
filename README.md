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

1. **SPD-FLS1 v2** (primary cohort; handle IMU and trial logs only). Oh, Deng & Lau (2026), Virginia Tech Data Repository, DOI [10.7294/29421842](https://doi.org/10.7294/29421842) (CC0). Needed only to rebuild features with `extract_features.py` or example traces in `make_figures_v2.py`.
2. **LASK v1.0** (external-consistency cohort; 37 public trials). Official release: [https://doi.org/10.5281/zenodo.20752651](https://doi.org/10.5281/zenodo.20752651) (CC BY 4.0). Needed only to rerun `h3_lask.py`. Pass the extracted directory (containing `participants.csv` and kinematics) as the script argument.

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

Random seeds used in the scripts are fixed (for example `20261110` in `h2_prognostic.py`).

Versioned snapshots of this repository are archived on Zenodo. All versions: [https://doi.org/10.5281/zenodo.24123988](https://doi.org/10.5281/zenodo.24123988) (always resolves to the latest release). Please cite the specific version used; see `ZENODO_GITHUB_ARCHIVE.md`.

## Citation

Wang, L. *Beyond completion time: responsiveness, reliability and external consistency of tool-handle gyroscope metrics in laparoscopic peg-transfer training.* Manuscript in preparation.

Please also cite the source datasets above.

This code is released under the MIT License (see `LICENSE` in the GitHub repository).

## Changelog

- **v1.0.1** (11 November 2026). `h1_mixed_models.py`, `h1_robustness.py`: a mixed-model fit is now retained only if it is admissible (optimiser converged, finite standard errors, positive-definite random-effects covariance); previously the fit with the highest restricted likelihood was taken without this check, which could select a degenerate boundary solution for some secondary metrics and crash when random effects were predicted. All primary and robustness estimates are unchanged; three rows of the secondary-metric table change in the second decimal. All analysis and figure scripts now create `results/` and `figures/` if they do not exist.
- **v1.0.0** (11 November 2026). First archived release.
