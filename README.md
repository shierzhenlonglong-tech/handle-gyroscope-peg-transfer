# handle-gyroscope-peg-transfer

Analysis code and derived trial-level tables for:

> Liang Wang. *Beyond completion time: responsiveness, reliability and external consistency of tool-handle gyroscope metrics in laparoscopic peg-transfer training.* Manuscript in preparation.

The study examines whether angular-velocity features from tool-handle gyroscopes in laparoscopic peg-transfer practice change with training (responsiveness), remain informative after controlling for completion time, show usable reliability, and are externally consistent with an independent instrument-tracking cohort.

This repository does **not** contain raw motion recordings, official public-dataset archives, ethics paperwork, or consent forms.

## Repository layout

| Path | Contents |
| --- | --- |
| `scripts/` | Analysis scripts (from the working-tree folder `03_scripts/`). |
| `data/derived/` | Derived and summary tables built from the public source datasets (from `02_数据/`). |
| `docs/analysis_plan_frozen_20261110.md` | Frozen analysis plan dated 10 November 2026, including recorded deviations. |
| `requirements.txt` | Third-party Python packages inferred from script imports. |

The scripts still use the original relative paths (`02_数据/…`, `../02_数据/`, `../07_results/`, `../04_figures/`). They have not been rewritten. To run them from this repository without editing the code, create a layout that matches those paths (see below).

## Requirements

Python 3 with:

```
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `statsmodels`, `scikit-learn`. Standard-library modules only otherwise.

## Data sources (not included)

Only datasets actually used in the analysis are listed. Download them from the official sites; do not substitute unofficial copies.

1. **SPD-FLS1 v2** (primary cohort; handle IMU and trial logs only).  
   Oh, Deng & Lau (2026), Virginia Tech Data Repository, DOI [10.7294/29421842](https://doi.org/10.7294/29421842) (CC0).  
   Needed to rebuild features from raw IMU files with `extract_features.py` and for example traces in `make_figures_v2.py`. The derived tables in `data/derived/` already contain the trial-level features used in the models.

2. **LASK v1.0** (external-consistency cohort; 37 public trials).  
   Official release: [https://doi.org/10.5281/zenodo.20752651](https://doi.org/10.5281/zenodo.20752651) (CC BY 4.0).  
   Needed only to rerun `h3_lask.py`. Pass the extracted directory (containing `participants.csv` and kinematics) as the script argument. Official zip archives and `participants.csv` are **not** redistributed here.

Participant codes such as `P01` are the public identifiers published with SPD-FLS1, not real names.

## Derived tables (`data/derived/`)

| File | Role |
| --- | --- |
| `trial_index.csv` | One row per trial, with exclusion flags and valid-trial markers. |
| `trial_features.csv` | One row per hand-trial, merged with the index (baseline features). |
| `trial_features_deg.csv` | Features recomputed under the alternative deg/s unit reading. |
| `trial_features_n0165.csv`, `trial_features_n05.csv` | Features after injecting 0.165 °/s or 0.5 °/s per-axis white noise. |
| `features_part{1,2,3}.csv` and the `_deg`, `_n0165`, `_n05` part files | Per-subset feature extracts that `build_trial_table.py` concatenates. |

No file in this release exceeds 50 MB. Analysis outputs written to `07_results/` and figures written to `04_figures/` can be regenerated and are not included.

## How to run

From a POSIX shell, in the repository root (Windows users can use junctions or copies with the same names):

```bash
# Scripts expect these sibling folder names
ln -s "$(pwd)/data/derived" 02_数据
mkdir -p 07_results 04_figures/v2
cd scripts
```

Feature extraction from a local SPD-FLS1 tree (optional if you use the provided derived tables):

```bash
# Run from the original project root, or create 02_数据 first.
# python extract_features.py <SPD-FLS1 root, zip or extracted> 02_数据/features_part1.csv --only part1
python extract_features.py /path/to/spd-fls1 ../02_数据/features_part1.csv --only part1
python build_trial_table.py            # also: _deg, _n0165, _n05
```

Primary analyses (working directory = `scripts/`):

```bash
python h1_mixed_models.py
python h1_robustness.py
python h2_prognostic.py
python extra_reliability_retention.py
python h3_lask.py /path/to/LASK/extracted
python make_figures_v2.py [/path/to/spd-fls1]   # figures; optional raw path for example traces
```

`feasibility_check_20261110.py` is a pre-plan exploratory check (known limitations are in its docstring). `oh2026_qc_sensitivity.py` is the quality-control module used by feature extraction. `mp3_gum.py` is a related uncertainty toolkit (`python mp3_gum.py selftest` uses synthetic data only).

Random seeds used in the scripts are fixed (for example `20261110` in `h2_prognostic.py`).

## Citation

Please cite the manuscript (status: *manuscript in preparation*) and the source datasets above:

Wang, L. *Beyond completion time: responsiveness, reliability and external consistency of tool-handle gyroscope metrics in laparoscopic peg-transfer training.* Manuscript in preparation.

This code is released under the MIT License (see `LICENSE`).
