# Analysis plan (frozen edition, 10 November 2026)

**Language note.** This is an English translation of `analysis_plan_frozen_20261110.md`. The Chinese original is authoritative if the two texts differ. The plan was written and frozen on 10 November 2026, before the formal mixed models were fitted. It was not registered on an independent platform.

This plan was written down before the formal models were run. Later changes are recorded in the “Deviations” section at the end and are disclosed in the manuscript.

## 0 Results already seen (disclosed)

Before this plan was written, a coarse check had been run (`feasibility_check_20261110.py`) on only four metrics (mean angular speed, P95 angular speed, low-speed fraction at 5 °/s, and time density at the threshold) and their within-person correlation with trial order. The trial order used in that check was incorrect (P01’s calibration record was treated as a trial; invalid records were not removed). The relationship of the remaining features (SPARC, pauses, peak rate, acceleration, bimanual coordination) to any outcome had not been examined.

## 1 Research questions and hypotheses

- **H1a (responsiveness).** Handle angular-motion metrics change monotonically with practice (within person).
- **H1b (independent information).** After control for the completion time of the same trial, the H1a effects remain.
- **H2 (prognostic, exploratory).** Adding IMU features to the completion times of previous trials improves prediction of whether the next trial meets the proficiency criterion and of the trial at which proficiency is first reached.
- **H3 (external consistency, exploratory).** The same metrics, in the LASK novice-to-expert range, have the same direction as H1a.

## 2 Data

SPD-FLS1 v2 (DOI 10.7294/29421842, CC0), using only handle IMU records and trial logs. Gyroscope readings are interpreted as rad/s and converted to °/s (the basis is recorded in the quality-control notes of a related project; the manuscript states this as an explicit assumption). The sampling rate is treated as 31.25 Hz.

## 3 Exclusion rules (implemented in `build_trial_table.py`)

At the trial level, 2007 records:

| Rule | Count |
|---|---|
| Calibration/baseline records (block 0; and P01 trial 20230911124510) | 39 |
| No log | 3 |
| Completion time < 20 s (interrupted or accidental stop) | 32 |
| Experimenter comments indicating equipment failure or a void record | 15 |
| **Valid trials retained** | **1918** |

A further 107 trials have duplicated numbers within a session (often a re-recording after an interruption). They are retained but flagged, and a sensitivity analysis removes them.

At the hand-trial level, IMU quality control from a related project is reused (row counts, sampling stability, flat/saturated axes, coverage of logged duration). Valid and quality-control pass: left 1843, right 1502, both hands 1492.

Trial order is the rank of each participant’s valid trials after sorting by time stamp.

## 4 Features

Computed per hand-trial, all from the magnitude of angular velocity:

| Family | Feature | Column |
|---|---|---|
| Primary (six) | Mean angular speed | `w_mean` |
| | P95 angular speed | `w_p95` |
| | Low-speed fraction (< 5 °/s) | `qlow_5` |
| | Pause-time fraction (< 5 °/s for ≥ 0.5 s) | `pause_frac` |
| | Spectral arc length SPARC (cut-off 10 Hz) | `sparc` |
| | Speed-peak rate (after 5 Hz low-pass, prominence ≥ 5 °/s) | `peaks_per_s` |
| Secondary | Total angular displacement, coefficient of variation, pause rate, low-speed fraction at 2 and 10 °/s, log dimensionless jerk, SD of acceleration magnitude | `ang_disp` `w_cv` `pause_rate` `qlow_2` `qlow_10` `ldlj` `acc_mag_sd` |
| Secondary (bimanual) | 1 s smoothed speed correlation, fraction of time both hands move, left-hand share of angular displacement | `bi_corr` `bi_both_move` `bi_left_share` |

Notes:

- `ldlj` is sensitive to the 0.57 °/s quantisation and is secondary only.
- The two handle IMUs use separate device clocks. Bimanual features are aligned at each stream’s first sample and may have tens of milliseconds of offset, so only 1 s smoothed quantities are used, and this is stated in the manuscript.
- Total angular displacement is approximately mean angular speed times duration and is not entered in models that already control for time.

## 5 Primary analysis (H1)

For each primary feature and each hand, linear mixed models are fitted:

- M1: feature ~ log(order) + first trial of session + (1 + log(order) | participant)
- M2: M1 + log(completion time) + whether an object was dropped

The test is the fixed effect of log(order), two-sided α = 0.05. Six features × two hands = 12 tests form one family; M1 and M2 are each adjusted by BH-FDR. H1b is taken to hold if the M2 effect is significant after adjustment and the slope sign is consistent in at least 15 of 19 participants.

Also reported:

- Standardized response mean (SRM): each person’s mean of the last 10 valid trials minus the mean of the first 10, divided by the SD of that difference; placed beside the SRM of completion time.
- The distribution of individual slopes (forest plot).

Pre-specified robustness analyses:

1. Replace log(completion time) with a spline, and split completion time into a person mean and a within-person deviation.
2. Left hand only; only trials in which both hands pass quality control.
3. Drop trials with duplicated numbers; drop trials with a dropped object.
4. Recompute features after injecting 0.165 °/s and 0.5 °/s per-axis white noise.
5. Low-speed thresholds of 2 and 10 °/s; SPARC cut-off 5 Hz.
6. Leave one participant out at a time (with attention to P10).

## 6 Prognostic analysis (H2, exploratory)

The logs cannot reproduce the original stopping rule: only 7 of 19 people accumulated 10 trials at ≤ 48 s with no drop, and most finished before six sessions. The endpoint is therefore not “reached the stopping criterion”, but:

- Outcome A: whether trial k is proficient (≤ 48 s and no drop), on all valid trials.
- Outcome B: the trial index of the first proficient trial (18 of 19 people have an event; P21 is censored).

Predictors use only information from before trial k: rolling means of completion time and of the primary features over the previous five trials, the slope so far, and trial order.

- Models: L2-regularised logistic regression (outcome A); discrete-time hazard model (outcome B). The control model contains only completion time and order.
- Validation: leave-one-subject-out cross-validation; AUC, Brier score and C-index; confidence intervals for differences by participant-clustered bootstrap (2000 replicates).
- No participant-level (n = 19) significance claims are made.

## 7 External consistency (H3, exploratory)

The public LASK subset (Zenodo 20752651, 37 trials at the time of writing). Angular velocity is obtained from quaternion pose, resampled to 31.25 Hz and quantised to 0.573 °/s, the same primary features are computed, and a trend test by experience grade is planned (Jonckheere–Terpstra). Only direction and effect size are reported. If trial-level grade labels are missing, this section is cancelled.

## 8 Software and reproducibility

Python (numpy, scipy, pandas, statsmodels). All scripts for this paper are in the repository root, random seeds are fixed, and the code is released at submission. (The public repository uses a flat layout; scripts are intended to be run from the repository root.)

## 9 Deviations

The following deviations occurred while the formal models were run on 10 November 2026 and are disclosed in Section 3.6 of the manuscript.

- **D1 Sensor-failure rule (added after seeing the data).** Learning curves showed that 30 right-hand records in P26’s second session had mean angular speed below 2 °/s and acceleration-magnitude SD below 0.06 m/s², while the task was completed normally; the sensor was judged not to have moved with the handle. New rule: hand-trials with mean angular speed < 5 °/s are excluded (the next lowest value in the dataset is 6.0 °/s). Right-hand records fall from 1502 to 1472; trials with both hands valid fall from 1492 to 1462. Results without this rule are reported as a robustness analysis.
- **D2 Optimiser choice.** The default optimiser returned a non-optimal solution for two models. The fit with the highest restricted likelihood among five optimisers was retained. This changed the left-hand pause-fraction M2 effect from p = 0.058 to 0.002, the only one of the 12 M2 effects whose significance changed.
- **D3 Algorithm for the count of consistent signs.** The plan did not specify which individual slopes to use. Unshrunk per-person least-squares slopes were used (the stricter choice); the shrunk mixed-model estimates give higher counts (14–19 of 19). By the strict algorithm, mean speed, P95 speed and speed-peak rate reach ≥ 15 of 19 in both hands; the low-speed fraction does so only for the left hand; pause fraction and SPARC do not.
- **D4 External consistency (H3) not done (as of 10 November 2026).** Neither the cloud environment nor the local command-line environment could reach Zenodo, so LASK had not been downloaded.
- **H2 result is negative.** Adding IMU features made prediction worse (outcome A, AUC 0.827 vs 0.863). The model was not adjusted in order to recover a positive result.

Added on 11 November 2026:

- **D4 update: H3 has been completed, with a change in procedure.** LASK was downloaded locally by the author (the local path is omitted from this public copy). The LASK sampling rate is about 14.7 Hz (cohorts A and B) and about 27 Hz (cohort C), so resampling to 31.25 Hz as planned was not used; all series were resampled to 14 Hz. Experience is a procedure count rather than a grade, so the trend test was changed to Spearman rank correlation, with additional within-cohort rank correlations and a partial correlation controlling for duration; the SPARC cut-off upper bound was 5 Hz. Result: P95 angular speed vs experience ρ = 0.41 (p = 0.012), duration ρ = −0.13; the other metrics were not significantly associated.
- **D5 Post-hoc analyses outside the plan.** Single-trial reliability and minimal detectable change, between-session change, and second-half slopes were added after the primary results were known and are labelled post hoc in the manuscript.
- **Alternative unit reading.** All features were recomputed as if the logged values were already in deg/s: standardised results for mean speed, P95 and SPARC were unchanged; the low-speed fraction, pause fraction and speed-peak rate degraded at the nominal thresholds.
