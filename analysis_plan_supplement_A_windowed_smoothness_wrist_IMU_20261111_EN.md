# Supplementary analysis plan A (written 11 November 2026, 19:50, before any related computation)

**Language note.** English translation of the Chinese original (`analysis_plan_supplement_A_20261111_ZH.md`), dated 11 November 2026; the Chinese file is authoritative. The Chinese original was named `分析方案_补充A_窗口化平滑度与腕部IMU_20261111.md` in the author’s local project. Paths such as `03_scripts/`, `02_数据/` (data) and `07_results/` refer to that local layout; in this repository the scripts and derived tables are in the repository root, and scripts write their result tables to `results/`.

Background: the primary analysis had been completed and its results were known. The two analyses below were added after the primary results were known and are labelled post hoc throughout the paper. This file was written down before the related code was run; results are reported whether or not they match expectations.

## A1 Windowed smoothness metrics

Motivation: whole-trial SPARC and LDLJ depend strongly on trial duration (in M2, the standardized coefficients of log T are about −1.3 and −2.4, respectively). Hypothesis: computing them within fixed-length windows and then averaging over windows removes this dependence.

Definitions (for each hand-trial, within each continuous sampled segment):
- Cut the angular-speed magnitude s(t) (31.25 Hz) into non-overlapping 128-sample windows (4.096 s); a tail shorter than one window is discarded. A record must have at least 3 complete windows, otherwise the metric is set to missing.
- Windowed SPARC (`sparc_w4`): compute SPARC in each window by the original method (normalized spectral amplitude threshold 0.05, cut-off upper bound 10 Hz) and take the arithmetic mean over windows.
- Windowed LDLJ (`ldlj_w4`): compute the log dimensionless jerk in each window (duration taken as the window length) and average over windows.
- Sensitivity: window lengths of 2.048 s (64 samples) and 8.192 s (256 samples), columns `_w2` and `_w8`.

Tests (left and right hand separately, with the same M1/M2 models and the same fitting rule as the primary analysis):
1. Insensitivity to duration: the standardized coefficient β3* of log T in M2. Criterion: |β3*| ≤ 0.30 (among the primary metrics other than SPARC, the smallest is 0.31).
2. Response to practice: β1* has the same sign in M1 and M2, and p < 0.05 in M2 (two hands × two metrics = 4 tests, BH-FDR).
3. Descriptive: SRM, single-trial ICC (the same last-20-trial detrended algorithm as in the main text), and Spearman correlation with experience in LASK (LASK is 14 Hz, so the window is 56 samples = 4 s and the SPARC upper bound is 5 Hz).

Three possible outcomes and how each would be written up:
- Criteria 1 and 2 both met: the windowed metrics are a workable correction for “whole-trial smoothness is affected by duration” and are written into the main text as a methodological contribution.
- Only criterion 1 met: windowing removes the duration dependence, but smoothness itself does not change with practice; this is reported as found, and the conclusion becomes “at this sampling specification, smoothness is not a valid learning metric”.
- Criterion 1 not met: windowing does not solve the problem; this is reported as found.

## A2 Comparison of wrist IMU and handle IMU

Data: the Shimmer files of SPD-FLS1 (two wrist sensors, about 51 Hz, gyroscope already calibrated to deg/s, time stamps from the system time of the same computer).

Processing: the angular-speed magnitude of each wrist sensor is resampled to 31.25 Hz, and the same six primary metrics as for the handle are computed (same thresholds). Quality control reuses the four handle checks (sample count, sampling stability, flat/saturated signal, duration coverage), with the nominal interval set to the measured median. Correspondence between wrist sensors and left/right hands: first check the data documentation; if there is none, decide at participant level by “which handle its 1 s smoothed angular speed correlates with more strongly”, and report the agreement rate of the assignment.

Tests:
1. Fit M1 and M2 for each wrist metric and report β1* beside that of the handle.
2. Within-person correlation (repeated-measures correlation) between wrist and handle metrics on the same trial and the same hand.
3. Whether a handle metric is still associated with practice after control for the wrist metric of the same name (M2 with the wrist metric added as a covariate), and vice versa.
4. Descriptive: the magnitude of wrist angular speed, as circumstantial evidence for the interpretation of the handle gyroscope unit (a plausibility argument only, not a proof).

No direction is assumed. Whether the wrist results are better, worse or the same, they are reported.

---

# A1 results record (evening of 11 November 2026, appended after running)

Scripts: `03_scripts/extract_windowed.py`, `a1_windowed_smoothness.py`, `a1_lask_windowed.py`; results: `07_results/a1_windowed.csv`, `a1_lask_windowed.csv`.

| Metric | Hand | M1 β1* | M2 β1* (p) | log T coefficient β3* | SRM | Single-trial ICC |
|---|---|---|---|---|---|---|
| Whole-trial SPARC | L / R | +0.21 / +0.18 | −0.17 / −0.25 | −1.25 / −1.35 | 0.87 / 0.51 | 0.39 / 0.38 |
| Windowed SPARC (4 s) | L / R | −0.39 / −0.42 | −0.16 (0.002) / −0.20 (<0.001) | +0.77 / +0.73 | −2.50 / −2.03 | 0.20 / 0.36 |
| Whole-trial LDLJ | L / R | +0.78 / +0.76 | +0.03 / +0.04 | −2.44 / −2.34 | 4.04 / 3.78 | 0.32 / 0.48 |
| Windowed LDLJ (4 s) | L / R | −0.13 / −0.06 | +0.03 (0.69) / +0.14 (0.09) | +0.54 / +0.62 | −0.31 / −0.09 | 0.84 / 0.63 |

Against the criteria written down in advance:
- **Criterion 1 (|β3*| ≤ 0.30) is not met.** Windowing moves the duration coefficient of SPARC from about −1.3 to about +0.75: its magnitude is halved and its sign reversed, but it is not eliminated. The coefficients for 2 s and 8 s windows are 0.44 / 0.43 and 0.61 / 0.30, which also do not meet the criterion.
- **Criterion 2 is met for windowed SPARC** (same sign in M1 and M2, FDR-adjusted p = 0.005 and 0.001) **and not met for windowed LDLJ**.

Interpretation (post hoc):
- The direction “becomes smoother with practice” of whole-trial SPARC in M1 is an artefact of shorter durations: after windowing, both M1 and M2 show that the speed spectrum within each 4 s window becomes more complex with practice (more negative SPARC), which agrees with the direction of whole-trial SPARC in M2 and with the rise in speed-peak rate.
- Windowed SPARC remains positively correlated with duration. A reasonable explanation is that slow trials contain more low-speed windows, which is an association of content rather than a computational coupling; but this is only an explanation and has not been tested.
- Windowed SPARC is highly responsive (SRM −2.5 / −2.0) but has low single-trial reliability (ICC 0.20 / 0.36) and is unrelated to experience in LASK (ρ = −0.07). Windowed LDLJ does not change with practice in SPD-FLS1 and correlates with experience in LASK at ρ = 0.34 (p = 0.038); the two datasets disagree.

Conclusion: windowing is not a clean “correction method” and cannot be written up as a methodological contribution. The claim it can support is narrower: the direction of the practice effect on whole-trial SPARC is a duration artefact; within a fixed window, this quantity changes with practice in the direction opposite to “smoother”, and behaves more like a measure of movement density.

---

# A2 results record (evening of 11 November 2026, appended after running)

Scripts: `03_scripts/extract_wrist.py`, `a2_wrist_vs_handle.py`; results: `07_results/a2_device_mapping.csv`, `a2_wrist_models.csv`, `a2_handle_vs_wrist.csv`; feature table: `02_数据/wrist_features.csv`. Raw files: `02_数据/shimmer_raw` (2004 files, 7.3 GB; part of the SPD-FLS1 raw data, not included in the code repository).

One issue during implementation: Shimmer files from later sessions have an extra column `VideoFrameN`. In the first run these files were missed; after the parsing was corrected, everything was rerun.

Data volume: among valid trials, 3368 wrist sensor-trials have a file; 3358 pass quality control and cover 0.9–1.2 times the task duration (left 1677, right 1681).

Correspondence of sensors to left and right hands: no data documentation was found, so the assignment was made from the rank correlation of trial-level mean angular speed with the handles. All 19 participants were assigned Shimmer_DA3F = left wrist; on a per-session recheck, 54 of 58 sessions agreed.

| Wrist metric | M1 β1* (L/R) | M2 β1* (L/R) | M2 p (L/R) | SRM (L/R) |
|---|---|---|---|---|
| Mean angular speed | 0.47 / 0.51 | 0.28 / 0.22 | <0.001 / 0.006 | 1.48 / 1.39 |
| P95 angular speed | 0.38 / 0.38 | 0.25 / 0.16 | 0.001 / 0.050 | 1.14 / 1.09 |
| Low-speed fraction | −0.55 / −0.58 | −0.29 / −0.29 | <0.001 / <0.001 | −1.16 / −1.20 |
| Pause-time fraction | −0.49 / −0.52 | −0.23 / −0.26 | 0.017 / 0.002 | −0.81 / −0.92 |
| SPARC | 0.21 / 0.28 | −0.17 / −0.21 | 0.051 / 0.018 | 0.44 / 0.57 |
| Speed-peak rate | 0.45 / 0.47 | 0.28 / 0.27 | <0.001 / <0.001 | 1.29 / 1.15 |

- The main handle findings are largely reproduced on the wrist sensors: all directions agree, effect sizes are similar, and SPARC likewise reverses sign after control for time.
- Within-person correlation between handle and wrist metrics on the same trial and the same hand: mean angular speed 0.84 / 0.88, P95 0.73 / 0.83, low-speed fraction 0.79 / 0.79, speed-peak rate 0.73 / 0.70, pause-time fraction 0.61 / 0.57, SPARC 0.58 / 0.48.
- After mutual adjustment, most no longer have an independent practice effect (handle after control for wrist: 4 of 12 with p < 0.05; wrist after control for handle: 4 of 12); that is, the sensors at the two locations provide largely overlapping information.
- Circumstantial evidence on units: handle mean angular speed is 1.27 times that of the wrist on the same side (median; 1.31–1.34 times for P95). The wrist gyroscope is in manufacturer-calibrated deg/s. If the handle readings were read as deg/s, handle speed would be only about 1/45 of wrist speed, which is physically implausible. This is evidence independent of the conversion itself, but it is still a plausibility argument, not a direct confirmation of the acquisition code.
