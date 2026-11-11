#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oh2026_qc_sensitivity.py - QC funnel and filter / noise-floor sensitivity of the angular metrics on the public
peg-transfer dataset of Oh, Deng & Lau (2026), Virginia Tech Data Repository, DOI 10.7294/29421842.

What it does
  1. Discovers trials: participant folders holding session archives 'PXX_FLS Task A_YY.zip' (or their extracted
     contents) with files '[YYYYMMDDHHMMSS]_Trial[Block]-[Trial]_IMU.txt', '..._log.txt', ...
  2. Writes a QC funnel next to the numbers reported in the data paper (1,971 task trials; 1,902 non-empty
     tool-handle IMU files; sampling-rate estimates from 1,893 left / 1,550 right trials):
        S0 trial keys -> S1 task trials (Block 0 / Trial 0 baselines removed) -> S2 IMU file present and non-empty
        -> S3 parsable rows (per hand) -> S4 stable sampling (per hand) -> S5 gyro sanity (per hand)
        -> S6 IMU span covers the logged task time (per hand) -> S7 both hands pass.
     Criteria are explicit (QCConfig) and printed with the funnel; S4 is our rule, so it need not reproduce the
     paper's 1,893 / 1,550 exactly - the difference itself is reported.
  3. On QC-passed hands: splits at gaps (never bridged), resamples each segment to a uniform 31.25 Hz grid, and
     scans low-pass cut-off fc (4th-order Butterworth, zero-phase filtfilt) x injected isotropic white noise
     sigma_add (deg/s per axis, added before filtering).  Common random numbers: one standard-normal draw per
     hand-trial is reused for every (fc, sigma) cell, so the surfaces are smooth and differences are paired.
  4. Estimates the native gyro noise floor from the Block-0 / Trial-0 baseline records (about 30 s of gaze
     fixation) when |omega - bias| follows the Maxwell (chi-3) law, i.e. when the tools were at rest.
  5. Outputs the systematic change of M04 (mean angular speed), M05 (cumulative angular activity A; its relative
     change equals that of M04 because T_valid is unchanged) and Q_low (low-angular-speed fraction) at several
     thresholds, and compares each with the analytic prediction of Lemma 2 / Prop. 2-3 (mp3_gum.py), using
     sigma_eff = sigma_add * sqrt(G(fc)) with G the white-noise power gain of the zero-phase filter.

Scope: agreement validates the propagation model on independent, realistic peg-transfer motion (speed
distributions, quiet fractions, reversals) and gives realistic sensitivity coefficients.  It says nothing about
MP3 hardware accuracy: different IMU (LSM6DSOX at ~31 Hz), no force channel, no hardware synchronisation.

Gyroscope fields: 'Gyro_X/Y/Z', 'Time' in ms since device start.  The data paper (Table 1) lists the gyroscope in
deg/s, but the logged values of SPD-FLS1 v2 behave as rad/s, the unit returned by the Adafruit LSM6DS library named
in the dataset README: at rest the per-axis SD is 0.003-0.01 at a print resolution of 0.01, and during the task the
median 95th percentile of the speed is about 0.74 (42 deg/s if rad/s, 0.74 deg/s if deg/s).  The default is
therefore --gyro-unit rad (values are multiplied by 180/pi); qc_funnel.md repeats this evidence for every run.
The 0.01 print resolution (0.573 deg/s per step in rad/s) is a quantisation floor that is reported as well.
Magnetometer fields are invalid and ignored.  Licence of SPD-FLS1 v2: CC0 1.0; cite the data paper and the DOI.
Folder layouts handled: session archives, or extracted folders named 'PXX_FLS Task A_YY' (sessions '5' and '05'
are the same session); baseline files are named '..._Trial0_IMU.txt' or '..._Trial0-0_IMU.txt'.

Usage
    python oh2026_qc_sensitivity.py run --root "F:/数据/SPD-FLS1_v2/解压文件" --out "D:/.../oh2026_out" [--jobs 4]
    python oh2026_qc_sensitivity.py selftest          # SYNTHETIC fixture in the documented format
Options: --fc 2 3 5 8 12 15  --sigma 0 0.1 0.25 0.5 1 2  --thresholds 2 5 10 20  --max-trials N  --no-plots
         --gyro-unit rad|deg (default rad for SPD-FLS1 v2)
Requires numpy, scipy, pandas, matplotlib; imports mp3_gum.py from the same folder.
"""
from __future__ import annotations

import argparse
import functools
import math
import os
import re
import sys
import tempfile
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mp3_gum as G  # noqa: E402

PAPER = {"S1_task_trials": 1971, "S2_imu_nonempty": 1902, "S4_stable_sampling_L": 1893, "S4_stable_sampling_R": 1550}
PAPER_NOTE = {"S4_stable_sampling_L": "paper: trials in the left-hand sampling-rate estimate",
              "S4_stable_sampling_R": "paper: trials in the right-hand sampling-rate estimate"}
TRIAL_RE = re.compile(r"(?P<ts>\d{14})[_ ]Trial(?P<block>\d+)(?:-(?P<trial>\d+))?(?P<suffix>(_[A-Za-z]+)?\.[A-Za-z0-9]+)$")
SES_DIR_RE = re.compile(r"(?P<pid>P\d+)_FLS Task A_(?P<session>\d+)$", re.I)
ZIP_RE = re.compile(r"(?P<pid>P\d+)_FLS Task A_(?P<session>\d+)\.zip$", re.I)
PID_RE = re.compile(r"^P\d+$", re.I)
SUFFIX = {"_imu.txt": "imu", "_log.txt": "log", "_eye.txt": "eye", "_shimmer.txt": "shimmer", ".mp4": "video", ".avi": "video"}
STAGES = ["S2", "S3", "S4", "S5", "S6"]  # stage reached by a hand: S6 = passed everything


@dataclass(frozen=True)
class QCConfig:
    min_rows: int = 50                 # S3: rows per hand
    nominal_dt: float = 0.032          # s (31.25 Hz per hand)
    tol_median_dt: float = 0.004       # S4: |median dt - nominal| <= 4 ms  (27.8-35.7 Hz)
    irregular_rel: float = 0.25        # S4: an interval is irregular if |dt - median| > 25 % of median ...
    max_irregular_frac: float = 0.05   # ... and at most 5 % of intervals may be irregular
    min_axis_std: float = 1e-3         # S5: flatline if any axis std < this (deg/s)
    max_plateau_frac: float = 0.01     # S5: saturation heuristic, fraction of samples at the per-axis |max|
    min_cover_of_log: float = 0.9      # S6: IMU span >= 90 % of the logged task time
    gap_factor: float = 3.0            # a gap > 3 x median dt splits segments (no bridging)
    min_segment_s: float = 2.0         # shorter segments are dropped from the metrics (>> filtfilt padding)
    gyro_scale: float = 1.0            # multiplier to deg/s (180/pi for SPD-FLS1 v2, see the module docstring)


@dataclass
class TrialRef:
    participant: str
    session: str
    key: str
    block: int
    trial: int
    members: dict = field(default_factory=dict)  # kind -> (container zip or None, member name or path, size)


def _seed(*parts) -> int:
    """Deterministic seed (Python's hash() is salted per process)."""
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) & 0xFFFFFFFF


# ----------------------------------------------------------------------------------------------
# Discovery and parsing
# ----------------------------------------------------------------------------------------------
def discover(root: Path):
    trials: dict[tuple, TrialRef] = {}
    info = {"duplicate_members": 0, "nested_zip_members": 0}

    def add(pid, ses, name, container, member, size):
        m = TRIAL_RE.search(name)
        if not m:
            return
        kind = SUFFIX.get(m.group("suffix").lower())
        if kind is None:
            return
        trial_no = int(m.group("trial")) if m.group("trial") is not None else 0
        key = f"{m.group('ts')}_Trial{m.group('block')}-{trial_no}"
        tr = trials.setdefault((pid, ses, key), TrialRef(pid, ses, key, int(m.group("block")), trial_no))
        if kind in tr.members:
            info["duplicate_members"] += 1
            if size <= tr.members[kind][2]:
                return
        tr.members[kind] = (container, member, size)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".zip":
            zm = ZIP_RE.search(path.name)
            pid = zm.group("pid").upper() if zm else next((p.upper() for p in path.parts if PID_RE.match(p)), "P?")
            ses = f"{int(zm.group('session')):02d}" if zm else path.stem
            with zipfile.ZipFile(path) as z:
                for zi in z.infolist():
                    if zi.is_dir():
                        continue
                    if zi.filename.lower().endswith(".zip"):
                        info["nested_zip_members"] += 1  # extract outer archives first
                        continue
                    add(pid, ses, os.path.basename(zi.filename), path, zi.filename, zi.file_size)
        else:
            sm = next((SES_DIR_RE.search(p) for p in reversed(path.parts[:-1]) if SES_DIR_RE.search(p)), None)
            if sm:
                pid, ses = sm.group("pid").upper(), f"{int(sm.group('session')):02d}"
            else:
                pid, ses = next((p.upper() for p in path.parts if PID_RE.match(p)), "P?"), path.parent.name
            add(pid, ses, path.name, None, str(path), path.stat().st_size)
    return sorted(trials.values(), key=lambda r: (r.participant, r.session, r.key)), info


def _read(ref) -> bytes:
    container, member, _ = ref
    if container is None:
        return Path(member).read_bytes()
    with zipfile.ZipFile(container) as z:
        return z.read(member)


def parse_imu(text: str, gyro_scale: float = 1.0):
    """Returns ({'LH': array(N,4) [t_s, gx, gy, gz] with gyro in deg/s after gyro_scale, 'RH': ...}, n_bad_rows).

    Accepts 'LH,Time:...' (P01-P07) and 'frame,LH,Time:...' (P08+) rows and the Unicode minus sign."""
    out = {"LH": [], "RH": []}
    bad = 0
    for line in text.splitlines():
        line = line.strip().replace("\u2212", "-")
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if parts[0] in out:
            label, kv = parts[0], parts[1:]
        elif len(parts) > 1 and parts[1] in out:
            label, kv = parts[1], parts[2:]
        else:
            bad += 1
            continue
        d = {}
        for item in kv:
            if ":" in item:
                k, v = item.split(":", 1)
                d[k.strip()] = v.strip()
        try:
            out[label].append((float(d["Time"]) / 1000.0, gyro_scale * float(d["Gyro_X"]),
                               gyro_scale * float(d["Gyro_Y"]), gyro_scale * float(d["Gyro_Z"])))
        except (KeyError, ValueError):
            bad += 1
    return {h: np.array(v, float).reshape(-1, 4) for h, v in out.items()}, bad


def parse_log(text: str):
    d = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    T = None
    for k, v in d.items():
        if k.lower().startswith("time elapsed"):
            m = re.match(r"\s*([-+]?\d+(?:\.\d+)?)", v)      # SPD-FLS1 v2 writes e.g. '31.5492571s'
            if m:
                T = float(m.group(1))
    try:
        drops = int(float(d.get("# dropped", "") or 0))
    except ValueError:
        drops = None
    return {"T_log": T, "dropped": drops}


# ----------------------------------------------------------------------------------------------
# QC per hand
# ----------------------------------------------------------------------------------------------
def qc_hand(arr: np.ndarray, T_log, cfg: QCConfig):
    """Returns (stage_reached, reason, info, (t, omega) if passed)."""
    info = {"n_rows": int(arr.shape[0])}
    if arr.shape[0] < cfg.min_rows:
        return "S2", "too few rows", info, None
    info["non_monotonic_rows"] = int(np.sum(np.diff(arr[:, 0]) < 0))
    a = arr[np.argsort(arr[:, 0], kind="stable")]
    keep = np.concatenate([[True], np.diff(a[:, 0]) > 0])
    a = a[keep]
    info["duplicate_times_removed"] = int(np.sum(~keep))
    dt = np.diff(a[:, 0])
    med = float(np.median(dt))
    irr = float(np.mean(np.abs(dt - med) > cfg.irregular_rel * med))
    info.update({"median_dt_s": med, "fs_est_Hz": 1.0 / med if med > 0 else float("nan"), "irregular_frac": irr,
                 "longest_gap_s": float(dt.max())})
    if not (abs(med - cfg.nominal_dt) <= cfg.tol_median_dt and irr <= cfg.max_irregular_frac):
        return "S3", "unstable sampling", info, None
    g = a[:, 1:4]
    steps = np.diff(np.unique(np.round(g[:, 0], 9)))
    info["print_step_deg_s"] = float(steps[steps > 1e-9].min()) if np.any(steps > 1e-9) else float("nan")
    sp_all = np.linalg.norm(g, axis=1)
    info["speed_p50"], info["speed_p95"] = float(np.median(sp_all)), float(np.quantile(sp_all, 0.95))
    sd = g.std(axis=0)
    plateau = max(float(np.mean(np.isclose(np.abs(g[:, i]), np.abs(g[:, i]).max()))) for i in range(3))
    info.update({"axis_std_min": float(sd.min()), "plateau_frac": plateau})
    if sd.min() < cfg.min_axis_std or plateau > cfg.max_plateau_frac:
        return "S4", "flatline or saturation", info, None
    span = float(a[-1, 0] - a[0, 0])
    info["span_s"] = span
    if T_log is not None and T_log > 0 and span < cfg.min_cover_of_log * T_log:
        return "S5", "IMU span does not cover the logged task time", info, None
    return "S6", "pass", info, (a[:, 0], g)


def segments_uniform(t, omega, fs, gap_factor, min_seg_s):
    """Split at gaps, then resample each segment to a uniform grid by linear interpolation."""
    dt = np.diff(t)
    med = np.median(dt)
    cuts = np.where(dt > gap_factor * med)[0]
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts + 1, [t.size]])
    segs = []
    for s, e in zip(starts, ends):
        if t[e - 1] - t[s] < min_seg_s:
            continue
        tg = np.arange(t[s], t[e - 1], 1.0 / fs)
        segs.append(np.column_stack([np.interp(tg, t[s:e], omega[s:e, i]) for i in range(3)]))
    return segs


# ----------------------------------------------------------------------------------------------
# Sensitivity scan for one hand-trial
# ----------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def butter_ba(fc, fs, order=4):
    return signal.butter(order, fc / (fs / 2.0))


@functools.lru_cache(maxsize=64)
def noise_power_gain(fc, fs, order=4, zero_phase=True, n=1 << 16):
    """Fraction of white-noise variance passed by the filter (|H|^4 for filtfilt)."""
    b, a = butter_ba(fc, fs, order)
    _, h = signal.freqz(b, a, worN=n)
    mag2 = np.abs(h) ** 2
    return float(np.mean(mag2 ** 2 if zero_phase else mag2))


def _speeds(segs_y):
    return np.concatenate([np.linalg.norm(y, axis=1) for y in segs_y])


def _metrics(sp, fs, thresholds):
    w = np.full(sp.size, 1.0 / fs)
    T = float(w.sum())
    out = {"M04": float(np.sum(w * sp) / T), "M05": float(np.sum(w * sp)), "T_valid": T}
    for th in thresholds:
        out[f"Qlow_{th:g}"] = float(np.sum(w * (sp < th)) / T)
    return out, w


def empirical_cth(sp, th, h=None):
    """Time density of the measured speed at the threshold (per deg/s): the c_th of Prop. 3."""
    h = max(0.5, 0.1 * th) if h is None else h
    return float(np.mean(np.abs(sp - th) < h) / (2.0 * h))


def scan_hand(segs, fs, fcs, sigmas, thresholds, seed):
    """Baseline metrics + one row per (fc, sigma).  Linear filtering: filt(x + s z) = filt(x) + s filt(z)."""
    rng = np.random.default_rng(seed)
    zs = [rng.standard_normal(y.shape) for y in segs]          # common random numbers for the whole grid
    sp0 = _speeds(segs)
    base, _ = _metrics(sp0, fs, thresholds)
    for th in thresholds:
        base[f"cth_{th:g}"] = empirical_cth(sp0, th)
    rows = []
    for fc in [None] + list(fcs):
        if fc is None:
            ys, ns, gain = segs, zs, 1.0
        else:
            b, a = butter_ba(fc, fs)
            ys = [signal.filtfilt(b, a, y, axis=0) for y in segs]
            ns = [signal.filtfilt(b, a, z, axis=0) for z in zs]
            gain = noise_power_gain(fc, fs)
        sp_ref = _speeds(ys)
        ref, w_ref = _metrics(sp_ref, fs, thresholds)
        for sg in sigmas:
            sp = sp_ref if sg == 0 else _speeds([y + sg * n for y, n in zip(ys, ns)])
            m, _ = _metrics(sp, fs, thresholds)
            s_eff = sg * math.sqrt(gain)
            pred = G.predict_m04_noise(sp_ref, w_ref, s_eff) if sg > 0 else ref["M04"]
            row = {"fc_Hz": np.nan if fc is None else fc, "sigma_add": sg, "sigma_eff": s_eff, "noise_gain": gain,
                   "M04": m["M04"], "M05": m["M05"], "M04_pred": pred,
                   "dM04_rel_%": 100 * (m["M04"] / base["M04"] - 1), "dM04_pred_rel_%": 100 * (pred / base["M04"] - 1),
                   "dM05_deg": m["M05"] - base["M05"], "pred_err_rel_%": 100 * (m["M04"] / pred - 1)}
            for th in thresholds:
                q = m[f"Qlow_{th:g}"]
                qp = G.predict_qlow_noise(sp_ref, w_ref, s_eff, th, nbins=300) if sg > 0 else ref[f"Qlow_{th:g}"]
                row[f"Qlow_{th:g}"] = q
                row[f"dQlow_{th:g}_pp"] = 100 * (q - base[f"Qlow_{th:g}"])
                row[f"dQlow_{th:g}_pred_pp"] = 100 * (qp - base[f"Qlow_{th:g}"])
                row[f"Qlow_{th:g}_pred_err_pp"] = 100 * (q - qp)
            rows.append(row)
    return base, rows


def process_trial(tr: TrialRef, cfg: QCConfig, fcs, sigmas, thresholds):
    """QC + scan of one task trial.  Returns (qc_rows, baseline_rows, sens_rows)."""
    ident = {"participant": tr.participant, "session": tr.session, "trial": tr.key, "block": tr.block, "trial_no": tr.trial}
    imu = tr.members.get("imu")
    if imu is None or imu[2] == 0:
        reason = "IMU file missing" if imu is None else "IMU file empty"
        return [{**ident, "hand": h, "imu_nonempty": False, "stage_reached": "S1", "reason": reason} for h in ("L", "R")], [], []
    T_log = drops = None
    if "log" in tr.members and tr.members["log"][2] > 0:
        lg = parse_log(_read(tr.members["log"]).decode("utf-8", errors="replace"))
        T_log, drops = lg["T_log"], lg["dropped"]
    data, nbad = parse_imu(_read(imu).decode("utf-8", errors="replace"), cfg.gyro_scale)
    fs = 1.0 / cfg.nominal_dt
    qc_rows, base_rows, sens_rows = [], [], []
    for h, hh in (("LH", "L"), ("RH", "R")):
        stage, reason, info, clean = qc_hand(data[h], T_log, cfg)
        q = {**ident, "hand": hh, "imu_nonempty": True, "stage_reached": stage, "reason": reason, "T_log": T_log,
             "dropped": drops, "n_bad_rows_file": nbad, **info}
        if stage == "S6":
            segs = segments_uniform(clean[0], clean[1], fs, cfg.gap_factor, cfg.min_segment_s)
            if not segs:
                q.update(stage_reached="S5", reason="no continuous segment >= min_segment_s")
            else:
                base, rows = scan_hand(segs, fs, fcs, sigmas, thresholds, _seed(tr.participant, tr.session, tr.key, hh))
                base_rows.append({**ident, "hand": hh, "T_log": T_log, "n_segments": len(segs), **base,
                                  **{k: v for k, v in info.items()}})
                sens_rows.extend({**ident, "hand": hh, **r} for r in rows)
        qc_rows.append(q)
    return qc_rows, base_rows, sens_rows


def _process_star(args):
    return process_trial(*args)


MAXWELL_MEDIAN = 1.5381722   # median of chi with 3 dof = sqrt(chi2_3.ppf(0.5))
MAXWELL_P95_OVER_MEDIAN = 1.8174


def baseline_noise(tr: TrialRef, cfg: QCConfig):
    """Native gyro noise floor from a Block-0 / Trial-0 baseline record (about 30 s of gaze fixation).

    Valid only if the tool was at rest: after removing the per-axis mean (bias), |omega - b| must follow a
    sigma-scaled chi-3 (Maxwell) law.  sigma_hat = median|omega - b| / 1.5382; 'static_like' checks the
    P95/median ratio (1.817 for Maxwell) and the axis SD balance.  Returns one row per hand.
    """
    imu = tr.members.get("imu")
    if imu is None or imu[2] == 0:
        return []
    data, _ = parse_imu(_read(imu).decode("utf-8", errors="replace"), cfg.gyro_scale)
    rows = []
    for h, hh in (("LH", "L"), ("RH", "R")):
        a = data[h]
        if a.shape[0] < cfg.min_rows:
            continue
        g = a[np.argsort(a[:, 0], kind="stable"), 1:4]
        b = g.mean(axis=0)
        sp = np.linalg.norm(g - b, axis=1)
        med = float(np.median(sp))
        ratio = float(np.quantile(sp, 0.95) / med) if med > 0 else float("nan")
        sd = g.std(axis=0)
        static = bool(abs(ratio / MAXWELL_P95_OVER_MEDIAN - 1) < 0.15 and sd.max() / max(sd.min(), 1e-12) < 2.0)
        steps = np.diff(np.unique(np.round(g[:, 0], 9)))
        rows.append({"print_step_deg_s": float(steps[steps > 1e-9].min()) if np.any(steps > 1e-9) else float("nan"),
                     "n_unique_x": int(np.unique(g[:, 0]).size), "speed_p50": float(np.median(np.linalg.norm(g, axis=1))),"participant": tr.participant, "session": tr.session, "trial": tr.key, "hand": hh, "n": int(sp.size),
                     "bias_x": b[0], "bias_y": b[1], "bias_z": b[2], "sd_x": sd[0], "sd_y": sd[1], "sd_z": sd[2],
                     "sigma_hat_maxwell": med / MAXWELL_MEDIAN, "p95_over_median": ratio, "static_like": static})
    return rows


# ----------------------------------------------------------------------------------------------
# Pipeline and reporting
# ----------------------------------------------------------------------------------------------
def funnel_counts(qc: pd.DataFrame, n_keys: int, n_task: int):
    f = {"S0_trial_keys": n_keys, "S1_task_trials": n_task,
         "S2_imu_nonempty": int(qc.loc[qc["hand"] == "L", "imu_nonempty"].sum())}
    lab = {"S3_parsable": 1, "S4_stable_sampling": 2, "S5_gyro_sane": 3, "S6_covers_log": 4}
    for name, idx in lab.items():
        for hh in ("L", "R"):
            d = qc[(qc["hand"] == hh) & qc["imu_nonempty"]]
            f[f"{name}_{hh}"] = int(d["stage_reached"].map(lambda s: STAGES.index(s) >= idx if s in STAGES else False).sum())
    both = qc.pivot_table(index=["participant", "session", "trial"], columns="hand", values="stage_reached", aggfunc="first")
    f["S7_both_hands_pass"] = int(((both.get("L") == "S6") & (both.get("R") == "S6")).sum())
    return f


def run(root: Path, out: Path, fcs, sigmas, thresholds, cfg: QCConfig = QCConfig(), max_trials=None,
        make_plots=True, jobs=1, verbose=True):
    out.mkdir(parents=True, exist_ok=True)
    trials, dinfo = discover(root)
    task = [r for r in trials if r.block != 0 and r.trial != 0]
    if max_trials:
        task = task[:max_trials]
    args = [(tr, cfg, tuple(fcs), tuple(sigmas), tuple(thresholds)) for tr in task]
    results = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for i, res in enumerate(ex.map(_process_star, args, chunksize=8)):
                results.append(res)
                if verbose and (i + 1) % 200 == 0:
                    print(f"  processed {i + 1}/{len(args)} task trials", flush=True)
    else:
        for i, a in enumerate(args):
            results.append(process_trial(*a))
            if verbose and (i + 1) % 200 == 0:
                print(f"  processed {i + 1}/{len(args)} task trials", flush=True)
    baseline = [r for r in trials if r.block == 0 or r.trial == 0]
    bn = pd.DataFrame([row for tr in baseline for row in baseline_noise(tr, cfg)])
    bn.to_csv(out / "baseline_noise.csv", index=False)
    qc = pd.DataFrame([r for res in results for r in res[0]])
    base = pd.DataFrame([r for res in results for r in res[1]])
    sens = pd.DataFrame([r for res in results for r in res[2]])
    f = funnel_counts(qc, len(trials), len(task))
    f.update({"duplicate_members_seen": dinfo["duplicate_members"], "nested_zip_members_skipped": dinfo["nested_zip_members"]})
    fdf = pd.DataFrame([{"stage": k, "count": v, "paper_reported": PAPER.get(k, ""), "note": PAPER_NOTE.get(k, "")}
                        for k, v in f.items()])
    fdf.to_csv(out / "qc_funnel.csv", index=False)
    qc.to_csv(out / "qc_per_hand.csv", index=False)
    qc[qc["stage_reached"] != "S6"].to_csv(out / "qc_exclusions.csv", index=False)
    byp = qc.assign(pass_=qc["stage_reached"] == "S6").groupby(["participant", "hand"]).agg(
        task_trials=("trial", "nunique"), imu_nonempty=("imu_nonempty", "sum"), passed=("pass_", "sum")).reset_index()
    byp.to_csv(out / "qc_funnel_by_participant.csv", index=False)
    base.to_csv(out / "trial_metrics_baseline.csv", index=False)
    sens.to_csv(out / "sensitivity_long.csv", index=False)
    surf = summarise(sens, thresholds)
    surf.to_csv(out / "sensitivity_surfaces.csv", index=False)
    write_funnel_md(out / "qc_funnel.md", fdf, qc, cfg, base, thresholds, bn)
    if make_plots and len(sens):
        plot_all(surf, thresholds, out)
    if verbose:
        print(fdf[["stage", "count", "paper_reported"]].to_string(index=False))
    return fdf, base, sens, surf


def write_funnel_md(path, fdf, qc, cfg, base, thresholds, bn=None):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Oh 2026 peg-transfer dataset: QC funnel\n\n")
        fh.write("Criteria (QCConfig): " + ", ".join(f"{k}={v}" for k, v in asdict(cfg).items()) + "\n\n")
        fh.write("| stage | our count | paper reported | note |\n|---|---|---|---|\n")
        for _, r in fdf.iterrows():
            fh.write(f"| {r['stage']} | {r['count']} | {r['paper_reported']} | {r['note']} |\n")
        fh.write("\nS4 applies our explicit sampling-stability rule; the paper's 1,893 (L) / 1,550 (R) are the trials it "
                 "used to estimate sampling rates, so the two need not match.\n\n")
        ok = qc[qc["imu_nonempty"] & qc["fs_est_Hz"].notna()] if "fs_est_Hz" in qc.columns else qc.iloc[0:0]
        if len(ok):
            fh.write("## Estimated sampling rate per hand (parsable hands)\n\n| hand | n | median Hz | IQR Hz |\n|---|---|---|---|\n")
            for hh, d in ok.groupby("hand"):
                q1, q2, q3 = d["fs_est_Hz"].quantile([0.25, 0.5, 0.75])
                fh.write(f"| {hh} | {len(d)} | {q2:.2f} | {q1:.2f}-{q3:.2f} |\n")
        reasons = qc[qc["stage_reached"] != "S6"].groupby(["hand", "reason"]).size().reset_index(name="n")
        if len(reasons):
            fh.write("\n## Exclusion reasons\n\n| hand | reason | n |\n|---|---|---|\n")
            for _, r in reasons.iterrows():
                fh.write(f"| {r['hand']} | {r['reason']} | {r['n']} |\n")
        if "speed_p95" in qc.columns and qc["speed_p95"].notna().any():
            ok2 = qc[qc["speed_p95"].notna()]
            fh.write(f"\n## Unit and resolution check (gyro_scale = {cfg.gyro_scale:.6g} to deg/s)\n\n"
                     f"Task hand-trials: median of per-hand speed P50 = {ok2['speed_p50'].median():.3f} deg/s, "
                     f"median of P95 = {ok2['speed_p95'].median():.3f} deg/s; print step of the gyro values "
                     f"(median over hands) = {ok2['print_step_deg_s'].median():.4f} deg/s.\n")
        if bn is not None and len(bn):
            fh.write("\n## Native noise floor from baseline records (Block 0 / Trial 0)\n\n"
                     "| hand | records | static-like | median sigma_hat (deg/s per axis) over static-like |\n|---|---|---|---|\n")
            for hh, d in bn.groupby("hand"):
                st = d[d["static_like"]]
                med = f"{st['sigma_hat_maxwell'].median():.3f}" if len(st) else "n/a"
                fh.write(f"| {hh} | {len(d)} | {len(st)} | {med} |\n")
            fh.write("\nOnly records whose |omega - bias| follows the Maxwell (chi-3) shape are used: the tools must have "
                     "been at rest.  If none qualify, the native noise floor stays unknown and only injected-noise shifts "
                     "are interpreted.\n")
        if len(base):
            fh.write("\n## Threshold sensitivity c_th of Q_low on the native data (per deg/s), median [IQR]\n\n")
            fh.write("| hand | " + " | ".join(f"{th:g} deg/s" for th in thresholds) + " |\n|---|" + "---|" * len(thresholds) + "\n")
            for hh, d in base.groupby("hand"):
                cells = []
                for th in thresholds:
                    q1, q2, q3 = d[f"cth_{th:g}"].quantile([0.25, 0.5, 0.75])
                    cells.append(f"{q2:.4f} [{q1:.4f}, {q3:.4f}]")
                fh.write(f"| {hh} | " + " | ".join(cells) + " |\n")


def _q25(x):
    return x.quantile(0.25)


def _q75(x):
    return x.quantile(0.75)


def _med_abs(x):
    return float(np.median(np.abs(x)))


def summarise(sens: pd.DataFrame, thresholds):
    if sens.empty:
        return sens
    s = sens.copy()
    s["fc_label"] = s["fc_Hz"].apply(lambda v: "none" if pd.isna(v) else f"{v:g}")
    spec = {"n_trial_hands": ("M04", "size")}
    for col in ("dM04_rel_%", "dM04_pred_rel_%", "dM05_deg"):
        spec[f"{col}_median"] = (col, "median")
        spec[f"{col}_q25"] = (col, _q25)
        spec[f"{col}_q75"] = (col, _q75)
    spec["pred_err_rel_%_median"] = ("pred_err_rel_%", "median")
    spec["pred_err_rel_%_median_abs"] = ("pred_err_rel_%", _med_abs)
    for th in thresholds:
        for col in (f"dQlow_{th:g}_pp", f"dQlow_{th:g}_pred_pp"):
            spec[f"{col}_median"] = (col, "median")
            spec[f"{col}_q25"] = (col, _q25)
            spec[f"{col}_q75"] = (col, _q75)
        spec[f"Qlow_{th:g}_pred_err_pp_median_abs"] = (f"Qlow_{th:g}_pred_err_pp", _med_abs)
    return s.groupby(["hand", "fc_label", "sigma_add"]).agg(**spec).reset_index()


# ----------------------------------------------------------------------------------------------
# Figures (reference palette of the dataviz method: diverging blue-gray-red, categorical slots 1-3)
# ----------------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4aaa9", "#e34948", "#9e2a2a"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def plot_all(surf: pd.DataFrame, thresholds, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    cmap = LinearSegmentedColormap.from_list("div_blue_gray_red", DIVERGING)

    def lum(rgb):
        r, g, b = rgb[:3]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def heat(col, title, fname, unit):
        hands = [h for h in ("L", "R") if h in set(surf["hand"])]
        fig, axes = plt.subplots(1, len(hands), figsize=(5.2 * len(hands), 3.9), constrained_layout=True, squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        vals = surf[col].to_numpy(float)
        vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
        vmax = vmax if vmax > 0 else 1.0
        for ax, hand in zip(axes[0], hands):
            d = surf[surf["hand"] == hand]
            piv = d.pivot(index="fc_label", columns="sigma_add", values=col)
            order = ["none"] + sorted([i for i in piv.index if i != "none"], key=float, reverse=True)
            piv = piv.reindex([o for o in order if o in piv.index])
            Z = piv.to_numpy(float)
            im = ax.imshow(Z, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(piv.shape[1]), [f"{c:g}" for c in piv.columns])
            ax.set_yticks(range(piv.shape[0]), ["native" if i == "none" else i for i in piv.index])
            ax.set_xlabel("injected noise sigma_add (deg/s per axis)")
            ax.set_ylabel("low-pass cut-off fc (Hz)")
            ax.set_title(f"{hand} hand", fontsize=9)
            _style(ax)
            for (i, j), v in np.ndenumerate(Z):
                if np.isfinite(v):
                    c = cmap((v + vmax) / (2 * vmax))
                    ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=7, color="#ffffff" if lum(c) < 0.45 else INK)
        cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.9)
        cb.set_label(unit, color=INK2)
        cb.ax.tick_params(labelsize=8, colors=INK2)
        fig.suptitle(title, fontsize=10, color=INK)
        fig.savefig(out / fname, dpi=300, facecolor=SURFACE)
        plt.close(fig)

    heat("dM04_rel_%_median", "Median change of M04 vs native data (equals the relative change of M05)", "surface_M04_rel.png", "%")
    heat("dM05_deg_median", "Median change of M05 (cumulative angular activity)", "surface_M05_abs.png", "deg")
    for th in thresholds:
        heat(f"dQlow_{th:g}_pp_median", f"Median change of Q_low at {th:g} deg/s vs native data",
             f"surface_Qlow_{th:g}.png", "percentage points")

    # predicted vs empirical (identity line); one point per (hand, fc, sigma) cell with sigma > 0
    d = surf[surf["sigma_add"] > 0]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    for i, hand in enumerate(h for h in ("L", "R") if h in set(d["hand"])):
        dh = d[d["hand"] == hand]
        ax.scatter(dh["dM04_pred_rel_%_median"], dh["dM04_rel_%_median"], s=36, color=CAT[i], edgecolor=SURFACE,
                   linewidth=1.0, label=f"{hand} hand", zorder=3)
    lim = [np.nanmin(d[["dM04_pred_rel_%_median", "dM04_rel_%_median"]].to_numpy()),
           np.nanmax(d[["dM04_pred_rel_%_median", "dM04_rel_%_median"]].to_numpy())]
    ax.plot(lim, lim, color=MUTED, lw=1.0, ls="--", zorder=2)
    ax.set_xlabel("predicted change of M04, Lemma 2 (%)")
    ax.set_ylabel("observed change of M04 (%)")
    ax.set_title("M04: analytic prediction vs perturbation", fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    _style(ax)
    ax = axes[1]
    allv = []
    for i, th in enumerate(thresholds[:3]):
        x = d[f"dQlow_{th:g}_pred_pp_median"]
        y = d[f"dQlow_{th:g}_pp_median"]
        allv += [x.to_numpy(), y.to_numpy()]
        ax.scatter(x, y, s=36, color=CAT[i], edgecolor=SURFACE, linewidth=1.0, label=f"threshold {th:g} deg/s", zorder=3)
    allv = np.concatenate(allv)
    lim = [np.nanmin(allv), np.nanmax(allv)]
    ax.plot(lim, lim, color=MUTED, lw=1.0, ls="--", zorder=2)
    ax.set_xlabel("predicted change of Q_low, Prop. 3 (pp)")
    ax.set_ylabel("observed change of Q_low (pp)")
    ax.set_title("Q_low: analytic prediction vs perturbation", fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    _style(ax)
    fig.savefig(out / "validation_pred_vs_observed.png", dpi=300, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Synthetic fixture (self-test only)
# ----------------------------------------------------------------------------------------------
def _fmt_row(new_format, frame, label, t_ms, g):
    vals = (f"Time:{int(t_ms)},Temp(C):22.50,Acc_X:0.10,Acc_Y:9.70,Acc_Z:0.30,"
            f"Gyro_X:{g[0]:.2f},Gyro_Y:{g[1]:.2f},Gyro_Z:{g[2]:.2f},Mag_X:1.00,Mag_Y:1.00,Mag_Z:1.00")
    vals = vals.replace("-", "\u2212") if frame % 7 == 0 else vals   # exercise the Unicode minus
    return (f"{frame},{label}," if new_format else f"{label},") + vals


def build_fixture(root: Path, seed=5):
    """Writes a small dataset in the documented layout; returns the expected funnel counts."""
    rng = np.random.default_rng(seed)
    exp = {k: 0 for k in ("S1_task_trials", "S2_imu_nonempty", "S3_parsable_L", "S3_parsable_R", "S4_stable_sampling_L",
                          "S4_stable_sampling_R", "S5_gyro_sane_L", "S5_gyro_sane_R", "S6_covers_log_L", "S6_covers_log_R",
                          "S7_both_hands_pass")}
    plan = {("P01", 1, 2): "empty", ("P02", 2, 4): "missing", ("P08", 1, 3): "empty",
            ("P02", 1, 5): "R_irregular", ("P08", 2, 1): "R_irregular", ("P10", 1, 6): "R_irregular",
            ("P10", 2, 2): "L_flat", ("P01", 2, 6): "short"}
    for pid in ("P01", "P02", "P08", "P10"):
        new_fmt = int(pid[1:]) >= 8
        pdir = root / pid
        pdir.mkdir(parents=True, exist_ok=True)
        for ses in (1, 2):
            with zipfile.ZipFile(pdir / f"{pid}_FLS Task A_{ses:02d}.zip", "w", zipfile.ZIP_DEFLATED) as z:
                for trial in range(0, 7):  # trial 0 = baseline
                    key = f"202605{ses:02d}{10 + trial:02d}00{trial:02d}00_Trial1-{trial}"
                    fault = plan.get((pid, ses, trial), "") if trial > 0 else ""
                    T_task = float(rng.uniform(40, 90))
                    z.writestr(f"{key}_log.txt", f"ParticipantID:{pid}\nTask:FLS Task A\nSessionID:{ses}\nTrialID:{trial}\n"
                                                 f"Time elapsed:{T_task:.4f}s\n# dropped:0\nComments:\n")
                    if trial > 0:
                        exp["S1_task_trials"] += 1
                    if fault == "missing":
                        continue
                    if fault == "empty":
                        z.writestr(f"{key}_IMU.txt", "")
                        continue
                    T_rec = T_task * (0.5 if fault == "short" else 1.05) + 2.0
                    lines, t0 = [], float(rng.uniform(1e5, 4e5))
                    for hand in ("LH", "RH"):
                        if trial == 0:   # baseline: tool at rest, bias + white noise (sigma 0.3 deg/s)
                            n = int(30 * 31.25)
                            om = np.tile(rng.normal(0, 0.5, 3), (n, 1))
                        else:
                            _, om = G.synthetic_motion(T=T_rec, fs=31.25, seed=int(rng.integers(1e9)))
                            n = om.shape[0]
                        dt = rng.uniform(0.010, 0.080, n) if (hand == "RH" and fault == "R_irregular") else 0.032 + rng.normal(0, 0.0005, n)
                        tt = t0 + np.cumsum(dt) * 1000.0
                        g = np.zeros_like(om) if (hand == "LH" and fault == "L_flat") else om + rng.normal(0, 0.3, om.shape)
                        lines += [(tt[k], _fmt_row(new_fmt, k, hand, tt[k], g[k])) for k in range(n)]
                    lines.sort(key=lambda x: x[0])
                    z.writestr(f"{key}_IMU.txt", "\n".join(l for _, l in lines) + "\n")
                    if trial == 0:
                        continue
                    exp["S2_imu_nonempty"] += 1
                    okL, okR = fault not in ("L_flat", "short"), fault not in ("R_irregular", "short")
                    exp["S3_parsable_L"] += 1
                    exp["S3_parsable_R"] += 1
                    exp["S4_stable_sampling_L"] += 1
                    exp["S4_stable_sampling_R"] += fault != "R_irregular"
                    exp["S5_gyro_sane_L"] += fault != "L_flat"
                    exp["S5_gyro_sane_R"] += fault != "R_irregular"
                    exp["S6_covers_log_L"] += okL
                    exp["S6_covers_log_R"] += okR
                    exp["S7_both_hands_pass"] += okL and okR
    return exp


def selftest():
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    thresholds = [5.0, 10.0]
    with tempfile.TemporaryDirectory() as td:
        root, out1, out2 = Path(td) / "data", Path(td) / "out1", Path(td) / "out2"
        exp = build_fixture(root)
        fdf, base, sens, surf = run(root, out1, fcs=[3, 8, 15], sigmas=[0, 0.5, 2.0], thresholds=thresholds, jobs=1)
        got = dict(zip(fdf["stage"], fdf["count"]))
        for k, v in exp.items():
            check(f"funnel {k}", got.get(k) == v, f"expected {v}, got {got.get(k)}")
        nz = sens[sens["sigma_add"] > 0]
        err = float(np.median(np.abs(nz["pred_err_rel_%"])))
        check("Lemma 2 prediction vs observed M04 under injected noise", err < 0.5, f"median |err| {err:.3f} %")
        qerr = float(np.median(np.abs(nz["Qlow_5_pred_err_pp"])))
        check("Prop. 3 prediction vs observed Q_low (5 deg/s)", qerr < 1.0, f"median |err| {qerr:.3f} pp")
        rel_eq = np.allclose((sens["M05"] - sens["dM05_deg"]) * (1 + sens["dM04_rel_%"] / 100), sens["M05"], rtol=1e-9)
        check("relative change of M05 equals that of M04 (T_valid unchanged)", rel_eq)
        gain_ok = all(abs(noise_power_gain(fc, 31.25) - np.var(signal.filtfilt(*butter_ba(fc, 31.25),
                      np.random.default_rng(3).standard_normal(400000))) ) < 0.01 for fc in (3.0, 8.0))
        check("filter noise-power gain |H|^4 equals simulated filtfilt variance", gain_ok)
        fdf2, _, sens2, _ = run(root, out2, fcs=[3, 8, 15], sigmas=[0, 0.5, 2.0], thresholds=thresholds, jobs=2, verbose=False)
        key = ["participant", "session", "trial", "hand", "fc_Hz", "sigma_add"]
        a = sens.sort_values(key).reset_index(drop=True)
        b = sens2.sort_values(key).reset_index(drop=True)
        check("parallel run reproduces the sequential run exactly (deterministic seeds)",
              fdf["count"].tolist() == fdf2["count"].tolist() and np.allclose(a["M04"], b["M04"], rtol=0, atol=0))
        pngs = sorted(p.name for p in out1.glob("*.png"))
        check("figures written", len(pngs) >= 4, str(pngs))
        bn = pd.read_csv(out1 / "baseline_noise.csv")
        sig_ok = len(bn) > 0 and bn["static_like"].all() and abs(bn["sigma_hat_maxwell"].median() / 0.3 - 1) < 0.05
        check("baseline records: Maxwell estimate of the native noise floor", sig_ok,
              f"sigma_hat median {bn['sigma_hat_maxwell'].median():.3f} (true 0.300), {len(bn)} hand-records")
        # extracted-folder layout of SPD-FLS1 v2: 'PXX_FLS Task A_YY' folders (also nested), session '1' == '01',
        # baselines named '..._Trial0_IMU.txt'; the funnel must not change and no participant may be 'P?'
        ext = Path(td) / "extracted"
        for zp in sorted(root.rglob("*.zip")):
            zm = ZIP_RE.search(zp.name)
            ses = int(zm.group("session"))
            folder = ext / (f"{zm.group('pid')}_FLS Task A_{ses}" if ses == 1 else f"{zm.group('pid')}_FLS Task A_{ses:02d}")
            dest = folder / folder.name if zm.group("pid") == "P01" else folder      # nested copy for P01
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp) as z:
                for n in z.namelist():
                    name = re.sub(r"_Trial1-0_", "_Trial0_", n)                       # baseline without the dash
                    (dest / name).write_bytes(z.read(n))
        tr_zip, _ = discover(root)
        tr_ext, _ = discover(ext)
        task_keys = lambda trs: sorted((t.participant, t.session, t.block, t.trial) for t in trs if t.block != 0 and t.trial != 0)
        same = task_keys(tr_zip) == task_keys(tr_ext) and len(task_keys(tr_zip)) == 48
        check("extracted-folder layout and 'Trial0' baselines discovered like the archives",
              same and all(t.participant != "P?" for t in tr_ext) and sum(t.block == 0 for t in tr_ext) == 8,
              f"{len(tr_zip)} archive keys, {len(tr_ext)} extracted keys")
        txt = "LH,Time:1000,Gyro_X:0.01,Gyro_Y:-0.02,Gyro_Z:1.00\nRH,Time:1032,Gyro_X:0.00,Gyro_Y:0.00,Gyro_Z:0.00\n"
        d_rad, _ = parse_imu(txt, 180.0 / math.pi)
        check("gyro unit conversion rad/s -> deg/s", np.isclose(d_rad["LH"][0, 3], 57.29578, atol=1e-4))
        mdtxt = (out1 / "qc_funnel.md").read_text(encoding="utf-8")
        check("funnel markdown lists the paper's numbers", all(str(v) in mdtxt for v in PAPER.values()))
    n_ok = sum(results)
    print(f"\nselftest: {n_ok}/{len(results)} checks passed (SYNTHETIC fixture; software check only)")
    return n_ok == len(results)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--fc", type=float, nargs="+", default=[2, 3, 5, 8, 12, 15])
    r.add_argument("--sigma", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1.0, 2.0])
    r.add_argument("--thresholds", type=float, nargs="+", default=[2, 5, 10, 20])
    r.add_argument("--max-trials", type=int, default=None)
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--no-plots", action="store_true")
    r.add_argument("--gyro-unit", choices=["rad", "deg"], default="rad",
                   help="unit of the logged Gyro_X/Y/Z values (SPD-FLS1 v2: rad/s, see the module docstring)")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return 0 if selftest() else 1
    fs = 1.0 / QCConfig().nominal_dt
    bad = [f for f in a.fc if not 0 < f < fs / 2]
    if bad:
        raise SystemExit(f"cut-offs must lie in (0, {fs / 2:.3f}) Hz (Nyquist of 31.25 Hz): {bad}")
    sig = sorted(set([0.0] + list(a.sigma)))
    cfg = QCConfig(gyro_scale=180.0 / math.pi if a.gyro_unit == "rad" else 1.0)
    run(Path(a.root), Path(a.out), a.fc, sig, a.thresholds, cfg=cfg, max_trials=a.max_trials, make_plots=not a.no_plots,
        jobs=a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
