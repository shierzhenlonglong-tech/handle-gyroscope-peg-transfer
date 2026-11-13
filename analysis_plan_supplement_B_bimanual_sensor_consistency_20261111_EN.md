# Supplementary analysis plan B (written 11 November 2026, 21:00, before the related computations)

**Language note.** English translation of the Chinese original (`analysis_plan_supplement_B_20261111_ZH.md`), dated 11 November 2026; the Chinese file is authoritative. The Chinese original was named `分析方案_补充B_双手协调与传感器一致性_20261111.md` in the author’s local project. In this repository the scripts and derived tables are in the repository root, and scripts write their result tables to `results/`.

Both analyses were added after the primary results were known and are labelled post hoc in the paper. Related information already known when this file was written: the within-person correlation between wrist and handle mean angular speed is about 0.84–0.88; the median handle/wrist speed ratio is about 1.27; the 30 handle records of P26’s right hand in session 2 have mean angular speed of about 1.5 °/s (deviation D1). The relationship of bimanual coordination metrics to practice and to time had not yet been examined.

## B1 Bimanual coordination from the synchronized wrist sensors

Rationale: the time stamps of the two wrist sensors come from the system clock of the same computer and can be placed on one time axis; this is not possible for the two handle sensors, which each use their own clock.

Processing: in each trial in which both wrist sensors pass quality control, take the overlapping interval of their time stamps, linearly interpolate the angular-speed magnitude onto one common 31.25 Hz time axis, and apply a 0.5 s moving average (16 samples) to each. The overlapping interval must cover 0.9–1.2 times the logged task duration. The movement threshold is the same as in the main text (5 °/s).

Primary metrics (three, one value per trial):
- `bi_both`: fraction of time both hands move simultaneously (both ≥ 5 °/s).
- `bi_single`: fraction of time one hand moves (exactly one ≥ 5 °/s).
- `bi_corr`: zero-lag Pearson correlation of the two hands’ smoothed angular speeds.

Secondary metrics: `bi_idle` (fraction of time both hands are < 5 °/s); `bi_lag` (time lag, in seconds, at which the absolute cross-correlation within ±2 s is largest); `bi_xcmax` (the cross-correlation value at that lag).

Tests: the same M1 and M2 mixed models and fitting rule as the primary analysis (trial level, one row per trial). BH-FDR is applied to the three primary metrics within M1 and within M2 separately. SRM and the single-trial ICC over the last 20 trials are also reported. Criterion: an M2 adjusted p < 0.05 and unshrunk per-person slopes with the same sign in at least 15 of 19 participants are taken to mean “still changes with practice at equal completion time”.

No direction is assumed. The three fractions sum to 1 (including `bi_idle`), so they are not independent and should be interpreted accordingly.

## B2 Handle–wrist agreement as a sensor-coupling check

Purpose: to turn the judgement “the sensor did not move with the instrument” from manual inspection of plots into a rule that can be applied automatically, and to test it with the independent wrist data.

Definition: for the same trial and the same hand, the ratio ρ = handle mean angular speed / wrist mean angular speed (for the handle, all records passing conventional quality control before D1 is applied are used).
- Rule (written down before the distribution of the ratio was examined): ρ < 0.25 is judged as decoupling of the handle sensor; ρ > 5 is judged as decoupling of the wrist sensor.

Reported:
1. The distribution of ρ (median, 1% and 99% quantiles), and the number of records flagged by the rule, with the participants and sessions involved.
2. Agreement with D1 (mean angular speed < 5 °/s): how many of the 30 records excluded by D1 that have wrist data are flagged by the ρ rule; which additional records the ρ rule flags.
3. How conventional quality control (row count, sampling stability, flat/saturated signal, duration coverage) judges these records, to show whether the conventional checks can detect them.
4. The ρ values of P10’s later left-hand records (low acceleration SD), descriptive only.
5. Sensitivity: change in the number flagged with thresholds of 0.15 and 0.35.

Limitations (stated in advance): the thresholds were set with knowledge that the P26 fault exists, and the sample may contain only this one fault event, so the analysis can only show that the rule is workable and cannot give reliable estimates of sensitivity and specificity. If the primary results change after removing the records additionally flagged by the ρ rule, this is reported as found.

---

## Results record (evening of 11 November 2026, carried out according to the plan written above)

### B2 first, because it changes the inputs to B1 and A2

- 3345 handle records (before D1), of which 2873 have a wrist pair on the same hand. Median ρ 1.27; 5%–99% quantiles 1.07–1.71; per-participant medians 1.08–1.55.
- **ρ < 0.25: 30 records flagged, exactly the 30 excluded by D1 (P26 right hand, session 2), no more and no fewer.** These 30 have ρ of 0.09–0.16; the lowest among the remaining records is 0.49, with nothing in between. A threshold of 0.35 gives the same result; 0.15 misses 2. In the same P26 session, the left hand, and the right hand in the sessions before and after, all have ρ of 1.06–1.34.
- **ρ > 5: 20 records flagged, all in P16’s session 1 (10 left, 10 right).** This was not known beforehand: in that session the two wrist sensors had mean angular speeds of only 2.2 and 1.5 °/s while the handles had 13.5 and 14.0 °/s, indicating that the wrist sensors were not worn on the hands or did not move with them at the time. A threshold of 4 gives the same result; a threshold of 6 flags 4 fewer.
- Conventional quality control (row count, sampling stability, flat/saturated signal, duration coverage) passed all 50 of these records.
- P10 left hand: at trial 56, ρ suddenly rises to 2.79 and the handle acceleration SD to 1.49 (the sensor was knocked); over the following 25 trials ρ drops to 0.59–0.79 (previously 1.15–1.45), and the handle acceleration SD falls from about 0.5 to about 0.09, while the wrist is unchanged. **The pre-specified absolute thresholds did not flag these 25 records**: this is partial decoupling, and the rule missed it.
- After removing the 20 handle records with ρ > 5, the primary results are essentially unchanged (β1* changes ≤ 0.02, all still p < 0.01). These 20 handle records are not faulty themselves; the problem lies with the wrist, so they are not excluded from the primary analysis.

### Two changes prompted by B2 (made only after seeing the results; stated in the paper)

1. The D1 rule is also applied to wrist records (exclude mean angular speed < 5 °/s): 21 are removed (20 from P16 session 1 + 1 right-wrist record of P21 at 4.7 °/s). A2 and B1 are both based on the results after exclusion; the results before exclusion are kept in `a2_*_all.csv` and `b1_bimanual_all.csv`.
   - A2 wrist M2 β1* (left/right): mean angular speed 0.24/0.19 (p .001/.021), P95 0.21/0.14 (p .010/.093), low-speed fraction −0.24/−0.25, pause-time fraction −0.17/−0.21, SPARC −0.12/−0.18 (p .148/.023), peak rate 0.25/0.24. 10 of 12 have p < 0.05 (11 before exclusion, left-hand SPARC p = .051).
   - Handle–wrist within-person correlations (left/right): 0.85/0.92, 0.75/0.88, 0.90/0.92, 0.87/0.83, 0.61/0.52, 0.77/0.78. Speed ratio 1.27; P95 ratio 1.31–1.34.
2. Exploratory: the ratio relative to the person’s own median, ρ/ρ̃ < 0.7, can flag those 25 records of P10’s left hand (plus 1 each from P04 and P07); after removing these 27, the left-hand primary results are slightly stronger (mean angular speed 0.27 → 0.31; other directions unchanged; all p < 0.01). Note that the relative rule fails when faults make up half of a person’s records (the own median of P26’s right hand is pulled down). This rule was not pre-specified; it is offered only as a direction for improvement in the discussion, and the primary analysis is not changed.

### B1 Bimanual coordination (1464 trials, 19 participants; 1475 before exclusion)

The median start-time offset between the two wrist sensors is 0.045 s; the 99% quantile is 0.17 s.

| Metric | Mean | M1 β1* | M2 β1* [95% CI] | M2 adjusted p | logT coefficient | Same-sign persons M1/M2 | SRM | ICC |
|---|---|---|---|---|---|---|---|---|
| bi_both | 0.65 | 0.65 | 0.25 [0.14, 0.36] | < .001 | −1.34 | 19/16 | 2.72 | 0.73 |
| bi_single | 0.32 | −0.64 | −0.23 [−0.35, −0.11] | < .001 | 1.37 | 19/15 | −2.51 | 0.73 |
| bi_corr | 0.33 | 0.33 | 0.08 [−0.07, 0.23] | .27 | −0.84 | 17/11 | 0.89 | 0.51 |

- Criterion (M2 adjusted p < 0.05 and ≥ 15/19 same sign): **bi_both and bi_single pass; bi_corr does not.** The conclusion is the same before exclusion (0.28, 17/19; −0.19, 15/19; bi_corr p = .13).
- From the first 10 to the last 10 trials, the fraction of simultaneous movement of both hands increases by 0.28 on average and the fraction of one-hand movement decreases by 0.23. The absolute SRMs (2.7, 2.5) are comparable to that of completion time (2.3–2.4) and higher than for any single-hand metric.
- Secondary metrics: bi_idle M2 −0.23 (p = .043); bi_lag and bi_xcmax are unrelated to practice (p > .5); median lag 0.06 s.
- Points that need to be stated plainly: bi_both is highly correlated with each hand’s own movement fraction (within-person r 0.86, 0.90), so a large part of it reflects “each hand pauses less”. The exploratory metric bi_excess (fraction of simultaneous movement minus its expected value if the two hands were independent; not pre-specified) has M2 β1* 0.24 (p = .007, 15/19 same sign), indicating that, beyond each hand pausing less, the movements of the two hands also overlap more in time; but this metric was added post hoc.
- bi_both depends strongly on completion time (logT coefficient −1.34), as do the single-hand metrics, so it cannot be said to be independent of time.

### Conclusions

- B1 holds: the synchronized wrist sensors give bimanual metrics that the handle sensors cannot, and these still change with practice at equal completion time. The quantity is “simultaneous movement”, not “speed correlation”.
- B2 holds in part: the pre-specified rule had no misses and no false flags for complete decoupling (two events, one handle and one wrist; the latter was not known beforehand), but it missed one partial decoupling. With only 3 events, sensitivity and specificity cannot be given.
