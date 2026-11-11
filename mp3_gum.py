#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mp3_gum.py - metric-level GUM uncertainty tools for the MP3 sensor pods.

Implements the derivations in 论文2/规划与方法/GUM推导与统计分析方案.md:
  * Lemma 2      mean / CDF / PDF of the 3-D noncentral chi distribution (noise floor of |omega|)
  * Prop. 2, 3   noise and residual-bias propagation to M04, M05 and Q_low (low-angular-speed fraction)
  * Prop. 1      mount-induced bound on M04 from the gyro error matrix E (estimated from T02), T04 check
  * Plate        indexing-plate reference-angle uncertainty (caliper chords, pin-hole play, shaft play)
  * Prop. 4      non-ratiometric FSR divider: force-domain sensitivity, GUM budget for T05
  * GUM          combination (u_c, Welch-Satterthwaite nu_eff, U(k=2), U95) and Monte Carlo propagation

Test IDs follow thesis Table 3.1 (7 Nov 2026): T01 static noise, T02 indexing plate, T04 remounting,
T05 installed force calibration (incl. sustained-load creep and five remounts).

No measured data exist yet.  `python mp3_gum.py selftest` uses SYNTHETIC data only: it checks the
software and the mathematics, not the hardware.

Usage (examples):
    python mp3_gum.py selftest
    python mp3_gum.py plate --R 62 --u-chord 0.03 --d-pin 3.00 --D-base 3.20 --D-rotor 3.20 \
                            --D-shaft-hole 12.5 --D-shaft 12.0
    python mp3_gum.py t02 --csv T02_runs.csv [--tilt-bound-deg 1.0]
    python mp3_gum.py t04 --csv T04_runs.csv --delta-E 0.012 [--alpha 0.05]
    python mp3_gum.py force --F 5 --R-fsr 6000 --beta 1.0 --u-ref 0.02 --u-val 0.25 --nu-val 20 \
                            --u-eps 0.002 --remount-offset-sd 0.10 --remount-gain-sd 0.03
    python mp3_gum.py allan --csv T01_static.csv --fs 100

Requires numpy, scipy, pandas (matplotlib only for optional plots).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erf

SQRT_8_OVER_PI = math.sqrt(8.0 / math.pi)  # 1.5958 - mean of the 3-D noise magnitude at rest (in sigma)


# ----------------------------------------------------------------------------------------------
# Lemma 2: 3-D noncentral chi distribution of |a + n|, n ~ N(0, sigma^2 I3)
# ----------------------------------------------------------------------------------------------
def nc_chi3_mean(a, sigma):
    """E|a + n| for |a| = a >= 0 and isotropic Gaussian noise of per-axis std sigma.

    E = sigma * [ sqrt(2/pi) exp(-l^2/2) + (l + 1/l) erf(l/sqrt(2)) ],  l = a/sigma.
    Limits: l -> 0 gives sqrt(8/pi) sigma (1 + l^2/6);  l -> inf gives a + sigma^2/a.
    """
    a = np.asarray(a, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    a, sigma = np.broadcast_arrays(a, sigma)
    out = np.array(a, dtype=float, copy=True)
    pos = sigma > 0
    if np.any(pos):
        s = sigma[pos]
        lam = a[pos] / s
        small = lam < 1e-4
        res = np.empty_like(lam)
        res[small] = SQRT_8_OVER_PI * (1.0 + lam[small] ** 2 / 6.0)
        lb = lam[~small]
        res[~small] = math.sqrt(2.0 / math.pi) * np.exp(-0.5 * lb ** 2) + (lb + 1.0 / lb) * erf(lb / math.sqrt(2.0))
        out[pos] = s * res
    return out


def _nc_chi3_core(x, lam):
    """Returns (Phi(x-l) - Phi(-x-l), [phi(x-l) - phi(x+l)]/l) for x >= 0, l >= 0 (Lemma 2), numerically stable.

    Uses phi(x -/+ l) = phi(x) exp(+/- l x) exp(-l^2/2): for small l the bracket is 2 phi(x) e^(-l^2/2) sinh(l x)/l
    (no cancellation, Maxwell limit 2 x phi(x)); for l >= 1e-3 the direct difference is exact enough.
    """
    from scipy.special import ndtr
    phi = lambda z: np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)  # noqa: E731
    P = ndtr(x - lam) - ndtr(-x - lam)
    D = np.empty_like(x)
    small = lam < 1e-3
    if np.any(small):
        xs, ls = x[small], lam[small]
        sh = np.where(ls > 0, np.sinh(ls * xs) / np.where(ls > 0, ls, 1.0), xs)
        D[small] = 2.0 * phi(xs) * np.exp(-0.5 * ls * ls) * sh
    if np.any(~small):
        xb, lb = x[~small], lam[~small]
        D[~small] = (phi(xb - lb) - phi(xb + lb)) / lb
    return P, D


def nc_chi3_cdf(r, a, sigma):
    """P(|a + n| <= r) for n ~ N(0, sigma^2 I_3), closed form of Lemma 2(2).

    F(x; l) = Phi(x - l) - Phi(-x - l) - [phi(x - l) - phi(x + l)]/l with x = r/sigma, l = |a|/sigma.  Only the
    normal cdf/pdf are needed, so it is robust across SciPy versions (ncx2.cdf overflows for some arguments in
    older releases, e.g. SciPy 1.15).
    """
    r = np.asarray(r, dtype=float)
    a = np.asarray(a, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    r, a, sigma = np.broadcast_arrays(r, a, sigma)
    out = np.where(np.abs(a) <= r, 1.0, 0.0).astype(float)  # sigma == 0 limit
    pos = sigma > 0
    if np.any(pos):
        x = np.maximum(r[pos], 0.0) / sigma[pos]
        lam = np.abs(a[pos]) / sigma[pos]
        P, D = _nc_chi3_core(x, lam)
        out[pos] = np.clip(P - D, 0.0, 1.0)
    return out


def nc_chi3_pdf(r, a, sigma):
    """Density of R = |a + n| at r (per unit of r): (1/sigma) (x/l) [phi(x - l) - phi(x + l)], Lemma 2(1)."""
    r = np.asarray(r, dtype=float)
    a = np.asarray(a, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    r, a, sigma = np.broadcast_arrays(r, a, sigma)
    out = np.zeros_like(r, dtype=float)
    pos = (sigma > 0) & (r > 0)
    if np.any(pos):
        x = r[pos] / sigma[pos]
        lam = np.abs(a[pos]) / sigma[pos]
        _, D = _nc_chi3_core(x, lam)
        out[pos] = x * D / sigma[pos]
    return out


# ----------------------------------------------------------------------------------------------
# Angular metrics (thesis 3.9): M04, M05, Q_low on valid continuous intervals
# ----------------------------------------------------------------------------------------------
def interval_weights(t, max_gap):
    """Left-rectangle weights: sample k represents [t_k, t_{k+1}) if 0 < dt <= max_gap, else 0.

    Integration therefore never bridges a gap longer than max_gap (thesis: no integration across gaps).
    """
    t = np.asarray(t, dtype=float)
    w = np.zeros_like(t)
    if t.size < 2:
        return w
    dt = np.diff(t)
    ok = (dt > 0) & (dt <= max_gap)
    w[:-1] = np.where(ok, dt, 0.0)
    return w


@dataclass
class AngularMetrics:
    M04: float          # mean angular-speed magnitude, deg/s
    M05: float          # cumulative angular activity, deg
    T_valid: float      # s
    Q_low: float | None  # fraction of valid time with |omega| < threshold
    mbar: float         # |time-weighted mean unit rotation direction| in [0, 1]


def angular_metrics(t, omega, threshold=None, max_gap=0.1) -> AngularMetrics:
    """omega: (N, 3) bias-corrected angular velocity in deg/s (any orthogonal frame)."""
    omega = np.asarray(omega, dtype=float)
    w = interval_weights(t, max_gap)
    speed = np.linalg.norm(omega, axis=1)
    T = float(np.sum(w))
    if T <= 0:
        return AngularMetrics(float("nan"), float("nan"), 0.0, None, float("nan"))
    M05 = float(np.sum(w * speed))
    M04 = M05 / T
    Q = None
    if threshold is not None:
        Q = float(np.sum(w * (speed < threshold)) / T)
    nz = speed > 0
    unit = np.zeros_like(omega)
    unit[nz] = omega[nz] / speed[nz, None]
    mbar = float(np.linalg.norm(np.sum(w[:, None] * unit, axis=0)) / T)
    return AngularMetrics(M04, M05, T, Q, mbar)


def _binned(speed, w, nbins):
    if nbins is None:
        return np.asarray(speed, float), np.asarray(w, float)
    smax = float(np.max(speed)) if len(speed) else 1.0
    edges = np.linspace(0.0, smax * 1.0000001 + 1e-12, nbins + 1)
    hist, _ = np.histogram(speed, bins=edges, weights=w)
    centres = 0.5 * (edges[:-1] + edges[1:])
    keep = hist > 0
    return centres[keep], hist[keep]


def predict_m04_noise(speed, w, sigma, nbins=None):
    """E[M04] when isotropic noise of per-axis std sigma is added to a trace with magnitudes `speed`."""
    s, ww = _binned(speed, w, nbins)
    return float(np.sum(ww * nc_chi3_mean(s, sigma)) / np.sum(ww))


def predict_qlow_noise(speed, w, sigma, threshold, nbins=None):
    """E[Q_low] when noise sigma is added: time-average of P(|a + n| < threshold)."""
    s, ww = _binned(speed, w, nbins)
    return float(np.sum(ww * nc_chi3_cdf(threshold, s, sigma)) / np.sum(ww))


def qlow_sensitivities(speed, w, sigma, threshold, nbins=200, rel_step=1e-3):
    """Sensitivity coefficients of E[Q_low]:
       c_th    = dE[Q]/d(threshold)  = time-density of the measured speed at the threshold  (per deg/s)
       c_sigma = dE[Q]/d(sigma)      (central difference)                                     (per deg/s)
    """
    s, ww = _binned(speed, w, nbins)
    c_th = float(np.sum(ww * nc_chi3_pdf(threshold, s, sigma)) / np.sum(ww))
    h = max(sigma * rel_step, 1e-9)
    qp = np.sum(ww * nc_chi3_cdf(threshold, s, sigma + h)) / np.sum(ww)
    qm = np.sum(ww * nc_chi3_cdf(threshold, s, max(sigma - h, 0.0))) / np.sum(ww)
    c_sigma = float((qp - qm) / (2 * h))
    return c_th, c_sigma


def m04_noise_sensitivity(speed, w, sigma, nbins=None, rel_step=1e-3):
    """c_sigma = dE[M04]/d(sigma): sqrt(8/pi) for quiescent samples, about 2 sigma/|a| for moving ones."""
    h = max(sigma * rel_step, 1e-9)
    return (predict_m04_noise(speed, w, sigma + h, nbins) - predict_m04_noise(speed, w, max(sigma - h, 0.0), nbins)) / (2 * h)


def misclassified_rest_fraction(k):
    """Fraction of truly quiescent samples with |n| >= k sigma (i.e. counted as 'moving'): 1 - chi2_3.cdf(k^2)."""
    return float(1.0 - stats.chi2.cdf(k * k, 3))


# ----------------------------------------------------------------------------------------------
# T01: Allan deviation, residual bias
# ----------------------------------------------------------------------------------------------
def allan_deviation(rate, fs, n_tau=60):
    """Overlapping Allan deviation of a rate signal (deg/s) sampled at fs (Hz).

    Returns (tau [s], adev [deg/s]).  White rate noise of std sigma gives adev ~ sigma / sqrt(tau fs).
    """
    x = np.asarray(rate, dtype=float)
    N = x.size
    theta = np.concatenate([[0.0], np.cumsum(x)]) / fs  # integrated angle, length N+1
    m_max = max(1, (N - 1) // 9)
    ms = np.unique(np.logspace(0, math.log10(m_max), n_tau).astype(int))
    taus, adev = [], []
    for m in ms:
        if 2 * m >= theta.size:
            break
        d = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
        tau = m / fs
        avar = np.sum(d * d) / (2.0 * tau ** 2 * d.size)
        taus.append(tau)
        adev.append(math.sqrt(avar))
    return np.array(taus), np.array(adev)


def allan_summary(taus, adev):
    """ARW coefficient N (deg/sqrt(s)) read at tau = 1 s, and the Allan floor (deg/s).

    Bias instability is conventionally floor / 0.664 (flicker-noise model); both are reported.
    """
    taus = np.asarray(taus)
    adev = np.asarray(adev)
    N_arw = float(np.exp(np.interp(0.0, np.log(taus), np.log(adev)))) if taus.min() <= 1.0 <= taus.max() else float("nan")
    i = int(np.argmin(adev))
    return {"ARW_N_deg_per_sqrt_s": N_arw, "allan_floor_deg_s": float(adev[i]),
            "tau_at_floor_s": float(taus[i]), "bias_instability_deg_s": float(adev[i] / 0.664)}


def residual_bias_u(adev_at, T_static, T_trial):
    """Conservative per-axis standard uncertainty of the residual bias during a trial (deg/s).

    u(b_r) ~ sqrt( sigma_A(T_static)^2 + sigma_A(T_trial)^2 ):
      first term  - uncertainty of the static-hold bias estimate,
      second term - bias wander over the trial horizon.
    `adev_at` is a callable tau -> Allan deviation.
    """
    return float(math.sqrt(adev_at(T_static) ** 2 + adev_at(T_trial) ** 2))


def residual_bias_effect(omega, w, u_b, rest_threshold=None):
    """Effect of a constant residual bias (per-axis standard uncertainty u_b, deg/s) on M04 (Lemma 3).

    conservative : |dM04| <= |b_r|, so u <= sqrt(3) u_b (RMS of |b_r| for an isotropic residual)
    first_order  : dM04 ~ ubar . b_r with ubar the time-weighted mean unit direction of the moving samples,
                   so u ~ |ubar| u_b (moving fraction only; quiet samples respond at second order).
    """
    om = np.asarray(omega, float)
    sp = np.linalg.norm(om, axis=1)
    mov = sp > (rest_threshold if rest_threshold is not None else 0.0)
    ww = np.asarray(w, float)
    ubar = np.sum((ww[mov, None] * om[mov]) / sp[mov, None], axis=0) / np.sum(ww)
    return {"u_M04_conservative": math.sqrt(3.0) * u_b, "u_M04_first_order": float(np.linalg.norm(ubar)) * u_b,
            "mean_unit_direction_norm": float(np.linalg.norm(ubar))}


def integrated_angle_u(u_bias, T_turn, N_arw, bias_inst=0.0):
    """Type-A standard uncertainty (deg) of one T02 integrated angle: bias-estimate error x duration,
    angle random walk during the turn, and a bias-instability drift term (linear interpolation residual)."""
    return float(math.sqrt((u_bias * T_turn) ** 2 + (N_arw ** 2) * T_turn + (bias_inst * T_turn) ** 2))


# ----------------------------------------------------------------------------------------------
# Proposition 1: physical-remount bound
# ----------------------------------------------------------------------------------------------
def mount_bound_exact(E):
    """sigma_min, sigma_max of (I + E) and Delta_E = sigma_max - sigma_min (exact bound, Prop. 1(2))."""
    s = np.linalg.svd(np.eye(3) + np.asarray(E, float), compute_uv=False)
    return float(s.min()), float(s.max()), float(s.max() - s.min())


def mount_bound_first_order(E):
    """First-order version: eigenvalues of the symmetric part E_s = (E + E^T)/2 (Prop. 1(3))."""
    E = np.asarray(E, float)
    lam = np.linalg.eigvalsh(0.5 * (E + E.T))
    return float(lam.min()), float(lam.max()), float(lam.max() - lam.min())


def axis_weights(omega_sensor, w):
    """w_i = int(omega_i^2/|omega|) / int(|omega|): convex weights of Prop. 1(4)."""
    om = np.asarray(omega_sensor, float)
    sp = np.linalg.norm(om, axis=1)
    nz = sp > 0
    num = np.sum((w[nz, None] * om[nz] ** 2) / sp[nz, None], axis=0)
    return num / np.sum(w[nz] * sp[nz])


def estimate_E_from_T02(df: pd.DataFrame, tilt_bound_deg: float | None = None, n_mc: int = 20000, seed: int = 1):
    """Estimate the gyro error matrix terms from T02 runs and the Prop. 1 bound Delta_E.

    Required columns: pod, axis (x|y|z), direction (+1|-1), theta_ref_deg (>0),
                      Theta_x_deg, Theta_y_deg, Theta_z_deg (bias-corrected, signed integrated vector).
    Optional column : placement_deg (0 or 180): reversal about the rotation axis to separate
                      bracket tilt from cross-axis sensitivity.
    Returns a dict per pod with the diagonal terms E_jj^(+/-), off-diagonal estimates or bounds,
    Delta_E (Gershgorin, central) and Delta_E_95 (Monte Carlo over run-to-run scatter).
    """
    ax_idx = {"x": 0, "y": 1, "z": 2}
    rng = np.random.default_rng(seed)
    out = {}
    for pod, g in df.groupby("pod"):
        diag = {}       # (j, sign) -> (mean, se, n)
        leak = {}       # (i, j, placement) -> list of values
        for _, r in g.iterrows():
            j = ax_idx[str(r["axis"]).lower()]
            sgn = 1 if float(r["direction"]) > 0 else -1
            th = float(r["theta_ref_deg"])
            vec = np.array([r["Theta_x_deg"], r["Theta_y_deg"], r["Theta_z_deg"]], float)
            s_norm = np.linalg.norm(vec) / th - 1.0           # tilt-insensitive to first order
            diag.setdefault((j, sgn), []).append(s_norm)
            v = vec / (th * sgn)
            pl = float(r["placement_deg"]) if "placement_deg" in g.columns and not pd.isna(r.get("placement_deg")) else 0.0
            for i in range(3):
                if i != j:
                    leak.setdefault((i, j, pl), []).append(v[i])
        dstat = {k: (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"), len(v))
                 for k, v in diag.items()}
        # off-diagonal E_ij (rotation about j seen on axis i)
        Eij, Eij_se, method = np.zeros((3, 3)), np.zeros((3, 3)), {}
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                vals0 = leak.get((i, j, 0.0), [])
                vals180 = leak.get((i, j, 180.0), [])
                if vals0 and vals180:      # reversal: tilt cancels in the mean of the two placements
                    m0, m1 = np.mean(vals0), np.mean(vals180)
                    Eij[i, j] = 0.5 * (m0 + m1)
                    se0 = np.std(vals0, ddof=1) / math.sqrt(len(vals0)) if len(vals0) > 1 else 0.0
                    se1 = np.std(vals180, ddof=1) / math.sqrt(len(vals180)) if len(vals180) > 1 else 0.0
                    Eij_se[i, j] = 0.5 * math.hypot(se0, se1)
                    method[(i, j)] = "reversal"
                elif vals0:
                    tb = math.radians(tilt_bound_deg) if tilt_bound_deg is not None else float("nan")
                    Eij[i, j] = abs(np.mean(vals0)) + (tb if not math.isnan(tb) else 0.0)  # conservative magnitude
                    Eij_se[i, j] = np.std(vals0, ddof=1) / math.sqrt(len(vals0)) if len(vals0) > 1 else 0.0
                    method[(i, j)] = "bound(no reversal)" + ("" if tilt_bound_deg is not None else ", tilt bound missing")
        # Gershgorin radii of E_s.  'abs': 0.5(|E_ij|+|E_ji|) (conservative, keeps any rigid chip rotation);
        # 'signed': |E_ij+E_ji|/2 where both directions were reversal-estimated - a common rigid rotation of the
        # chip relative to the housing is antisymmetric and cancels (Prop. 1(3)); other pairs fall back to 'abs'.
        both_rev = np.zeros((3, 3), bool)
        for i in range(3):
            for j in range(3):
                both_rev[i, j] = i != j and method.get((i, j)) == "reversal" and method.get((j, i)) == "reversal"

        def radii(Emat, signed):
            A = 0.5 * (np.abs(Emat) + np.abs(Emat.T))
            if signed:
                S = 0.5 * np.abs(Emat + Emat.T)
                A = np.where(both_rev, S, A)
            np.fill_diagonal(A, 0.0)
            return A.sum(axis=0)

        R = radii(Eij, False)
        R_signed = radii(Eij, True)

        def gersh(dvals_by_axis, Rvec):
            hi = max(d + Rvec[j] for (j, _), d in dvals_by_axis.items())
            lo = min(d - Rvec[j] for (j, _), d in dvals_by_axis.items())
            return hi - lo

        dmean = {k: v[0] for k, v in dstat.items()}
        delta_c = gersh(dmean, R)
        delta_c_signed = gersh(dmean, R_signed)
        # Monte Carlo upper 95% value of Delta_E (run-to-run scatter of every estimated term).  Reference errors
        # common to all axes (plate geometry, time base) shift every delta equally and cancel in Delta_E.
        sims = np.empty(n_mc)
        sims_s = np.empty(n_mc)
        keys = list(dstat.keys())
        for n in range(n_mc):
            dd = {k: dstat[k][0] + (0.0 if math.isnan(dstat[k][1]) else rng.normal(0, dstat[k][1])) for k in keys}
            Ep = Eij + rng.normal(0, 1, (3, 3)) * Eij_se
            sims[n] = gersh(dd, radii(Ep, False))
            sims_s[n] = gersh(dd, radii(Ep, True))
        out[pod] = {
            "diag_E_jj": {f"{'xyz'[j]}{'+' if s > 0 else '-'}": {"mean": m, "se": se, "n": n} for (j, s), (m, se, n) in dstat.items()},
            "offdiag_E_ij": {f"E_{'xyz'[i]}{'xyz'[j]}": {"value": float(Eij[i, j]), "se": float(Eij_se[i, j]), "method": method.get((i, j), "")}
                             for i in range(3) for j in range(3) if i != j and (i, j) in method},
            "Delta_E_central": float(delta_c),
            "Delta_E_95": float(np.quantile(sims, 0.95)),
            "Delta_E_signed_central": float(delta_c_signed),
            "Delta_E_signed_95": float(np.quantile(sims_s, 0.95)),
            "note": "use Delta_E_95 (conservative) unless every off-diagonal pair has reversal runs; "
                    "Delta_E_signed_95 then removes a rigid chip-to-housing rotation, which cannot change |omega|",
        }
    return out


def oneway_components(values: pd.Series, groups: pd.Series):
    """One-way random-effects variance components (groups = mountings, reps within)."""
    d = pd.DataFrame({"y": values.astype(float), "g": groups})
    k = d["g"].nunique()
    N = len(d)
    gm = d.groupby("g")["y"]
    n_i = gm.size().values
    means = gm.mean().values
    grand = d["y"].mean()
    ssb = float(np.sum(n_i * (means - grand) ** 2))
    ssw = float(np.sum((d["y"] - gm.transform("mean")) ** 2))
    msb = ssb / (k - 1) if k > 1 else float("nan")
    msw = ssw / (N - k) if N > k else float("nan")
    n0 = (N - np.sum(n_i ** 2) / N) / (k - 1) if k > 1 else float("nan")
    s2b = max((msb - msw) / n0, 0.0) if k > 1 else 0.0
    s2w = msw
    sem = math.sqrt(s2b + s2w)
    # Satterthwaite dof of s2b + s2w = msb/n0 + msw (1 - 1/n0)
    c1, c2 = 1.0 / n0, 1.0 - 1.0 / n0
    nu = (c1 * msb + c2 * msw) ** 2 / ((c1 * msb) ** 2 / (k - 1) + (c2 * msw) ** 2 / (N - k))
    return {"k_groups": int(k), "N": int(N), "n0": float(n0), "MS_between": msb, "MS_within": msw,
            "s2_between": s2b, "s2_within": s2w, "SEM_new_mount": sem, "MDC95": 1.96 * math.sqrt(2.0) * sem,
            "nu_satterthwaite": float(nu)}


def t04_verify(df: pd.DataFrame, delta_E_upper: float, u_ref_rel: float = 0.0, alpha: float = 0.05):
    """T04 check of Prop. 1 (five removals and reattachments, repeated reference motions per installation).

    Columns: pod, mount, rep, theta_ref_deg, Theta_x_deg, Theta_y_deg, Theta_z_deg; optional shaft_deg
    (integrated rate about the documented shaft axis after the software transform R) and optional motion
    (e.g. shaft | transverse): each motion is checked separately, because a shaft-roll reference is insensitive
    to the re-clip angle (Corollary 1) and the scale anisotropy only shows up under a transverse rotation.
    delta_E_upper may be a float or a dict {pod: Delta_E_95}.  Result keys are "pod" or "pod|motion".
    r = |Theta| / theta_ref per run.  Criterion on the range of the n_m mount means of r:
        range <= Delta_E_upper * mean(r) + q_{1-alpha}(n_m, nu_w) * sqrt(s_w^2 / n_rep + u_ref_rel^2)
    q is the studentized-range quantile (nu_w = N - n_m).  u_ref_rel must be a reference component that is
    independent between mountings (normally 0: the pin/shaft play of each run is already inside s_w, and the
    hole-position and time-base errors are common to all mountings and cancel in the range).
    A larger range indicates pod slip / looseness or drift, not gyro anisotropy.
    """
    res = {}
    keys = ["pod", "motion"] if "motion" in df.columns else ["pod"]
    for key, g in df.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        pod = key[0]
        name = "|".join(map(str, key))
        dE = float(delta_E_upper.get(pod, max(delta_E_upper.values()))) if isinstance(delta_E_upper, dict) \
            else float(delta_E_upper)
        vec = g[["Theta_x_deg", "Theta_y_deg", "Theta_z_deg"]].to_numpy(float)
        r = np.linalg.norm(vec, axis=1) / g["theta_ref_deg"].to_numpy(float)
        gg = g.assign(r=r)
        comp = oneway_components(gg["r"], gg["mount"])
        n_m = gg["mount"].nunique()
        n_rep = gg.groupby("mount").size().mean()
        nu_w = comp["N"] - n_m
        mm = gg.groupby("mount")["r"].mean()
        spread = float(mm.max() - mm.min())
        q = float(stats.studentized_range.ppf(1 - alpha, n_m, nu_w)) if nu_w > 0 else float("nan")
        allowed = dE * float(mm.mean()) + q * math.sqrt(comp["s2_within"] / n_rep + u_ref_rel ** 2)
        entry = {"mount_means_r": mm.to_dict(), "spread_r": spread, "allowed_spread": allowed, "q_studentized_range": q,
                 "verdict": "PASS (within Prop. 1 bound)" if spread <= allowed else "FLAG (exceeds bound: check slip/looseness)",
                 "SEM_new_mount_rel": comp["SEM_new_mount"] / float(mm.mean()),
                 "MDC95_rel": comp["MDC95"] / float(mm.mean()), "norm_ratio_components": comp}
        if "shaft_deg" in g.columns:
            p = gg["shaft_deg"].to_numpy(float) / gg["theta_ref_deg"].to_numpy(float)
            gp = gg.assign(p=p)
            pm = gp.groupby("mount")["p"].mean()
            entry["shaft_projection_spread"] = float(pm.max() - pm.min())
            entry["shaft_projection_components"] = oneway_components(gp["p"], gp["mount"])
        entry.update({"delta_E_upper": dE, "nu_within": float(nu_w), "n_rep": float(n_rep)})
        res[name] = entry
    return res


# ----------------------------------------------------------------------------------------------
# Indexing-plate reference-angle uncertainty
# ----------------------------------------------------------------------------------------------
@dataclass
class PlateGeometry:
    R: float = 62.0            # index-hole circle radius, mm (make_plate.py INDEX_R)
    u_chord: float = 0.03      # standard uncertainty of a caliper centre-to-centre chord, mm (Type A, measure!)
    d_pin: float = 3.00        # pin diameter, mm
    D_base: float = 3.20       # base-hole diameter, mm (make_plate.py HOLE_D; measure after drilling)
    D_rotor: float = 3.20      # rotor-hole diameter, mm
    D_shaft_hole: float = 12.5 # rotor centre bore, mm (SHAFT_D + SHAFT_CLEAR)
    D_shaft: float = 12.0      # shaft diameter, mm
    preload_factor: float = 1.0  # <1 if the operator always preloads against the pin in the turn direction


def plate_reference_u(theta_nom_deg: float, geo: PlateGeometry = PlateGeometry()):
    """Standard uncertainty (deg) of the reference angle between the start and end pinned positions.

    Components (rectangular distributions, start and end stops independent):
      hole position : from caliper chords; zero for whole turns (same hole)
      pin-hole play : half-width a_pin = ((D_base-d_pin)/2 + (D_rotor-d_pin)/2) / R
      shaft play    : half-width a_shaft = ((D_shaft_hole - D_shaft)/2) / R  (rotor centre can move)
    """
    R = geo.R
    a_pin = ((geo.D_base - geo.d_pin) / 2 + (geo.D_rotor - geo.d_pin) / 2) / R * geo.preload_factor
    a_sh = ((geo.D_shaft_hole - geo.D_shaft) / 2) / R
    u_pin_stop = a_pin / math.sqrt(3)
    u_sh_stop = a_sh / math.sqrt(3)
    u_play = math.sqrt(2.0 * (u_pin_stop ** 2 + u_sh_stop ** 2))  # rad
    th = abs(theta_nom_deg) % 360.0
    if math.isclose(th, 0.0, abs_tol=1e-9):
        u_hole = 0.0
        note = "whole turn: hole-position term cancels; play terms remain"
    else:
        # angle of each <=90 deg segment from its chord: theta = 2 asin(c / 2R); R from the diametral chord
        n90 = int(math.ceil(th / 90.0 - 1e-9))
        seg = th / n90
        t = math.radians(seg)
        uc_seg = geo.u_chord / (R * math.cos(t / 2))          # independent chord readings
        uR_seg = 2.0 * math.tan(t / 2) / R * geo.u_chord / 2   # common radius estimate (u_R = u_chord/2)
        u_hole = math.sqrt(n90 * uc_seg ** 2 + (n90 * uR_seg) ** 2)
        note = f"{n90} chord segment(s) of {seg:.1f} deg; chord terms independent, radius term common"
    u_tot = math.sqrt(u_play ** 2 + u_hole ** 2)
    return {"theta_nom_deg": theta_nom_deg, "u_hole_deg": math.degrees(u_hole), "u_pin_deg": math.degrees(math.sqrt(2) * u_pin_stop),
            "u_shaft_deg": math.degrees(math.sqrt(2) * u_sh_stop), "u_ref_deg": math.degrees(u_tot),
            "u_ref_rel": math.degrees(u_tot) / abs(theta_nom_deg), "U_k2_rel": 2 * math.degrees(u_tot) / abs(theta_nom_deg), "note": note}


def plate_mc(theta_nom_deg: float, geo: PlateGeometry = PlateGeometry(), n: int = 200000, seed: int = 2):
    """Monte Carlo of the rotor orientation change under the same rectangular model (checks plate_reference_u)."""
    rng = np.random.default_rng(seed)
    R = geo.R
    a_pin = ((geo.D_base - geo.d_pin) / 2 + (geo.D_rotor - geo.d_pin) / 2) * geo.preload_factor
    a_sh = (geo.D_shaft_hole - geo.D_shaft) / 2
    th = abs(theta_nom_deg) % 360.0
    if math.isclose(th, 0.0, abs_tol=1e-9):
        hole_err = np.zeros(n)
    else:
        u_hole = math.radians(plate_reference_u(theta_nom_deg, geo)["u_hole_deg"])
        hole_err = rng.normal(0, u_hole, n)
    # tangential displacements (mm) at start and end: pin play + rotor-centre shift, both rectangular
    t_start = rng.uniform(-a_pin, a_pin, n) + rng.uniform(-a_sh, a_sh, n)
    t_end = rng.uniform(-a_pin, a_pin, n) + rng.uniform(-a_sh, a_sh, n)
    err = hole_err + (t_end - t_start) / R
    return float(np.degrees(np.std(err)))


# ----------------------------------------------------------------------------------------------
# Proposition 4: non-ratiometric FSR divider (HW-B: 3V3_A - R23 - FSR - R24 - GND; HW-A: 22k / 15k)
# ----------------------------------------------------------------------------------------------
HW_B = {"R_s": 2200.0, "R_l": 15000.0, "label": "HW-B: R23 2.2 kOhm series current-limiting, R24 15 kOhm load"}
HW_A = {"R_s": 22000.0, "R_l": 15000.0, "label": "HW-A: 22 kOhm series current-limiting, 15 kOhm load"}
LSB_V = 4.096 / 32768.0  # 125 uV: ADS1115 single-ended at PGA +/-4.096 V (positive half only: 15-bit effective)


def divider_vin(R_fsr, V_exc=3.3, R_s=2200.0, R_l=15000.0):
    return V_exc * R_l / (R_s + np.asarray(R_fsr, float) + R_l)


def fsr_from_vin(V_in, V_exc=3.3, R_s=2200.0, R_l=15000.0):
    return V_exc * R_l / np.asarray(V_in, float) - R_s - R_l


def dlnF_dlnV(R_fsr, beta=1.0, R_s=2200.0, R_l=15000.0):
    """Prop. 4: d ln F / d ln V_in = beta (1 + (R_s + R_l)/R_fsr), beta = d ln F / d ln G (G = 1/R_fsr).

    The same factor multiplies a relative excitation drift eps = dV_exc/V_exc, because the ADS1115 measures
    against its internal reference while the divider is excited from 3.3 V (non-ratiometric).
    """
    return beta * (1.0 + (R_s + R_l) / np.asarray(R_fsr, float))


@dataclass
class Component:
    name: str
    u: float                 # standard uncertainty in output units (already multiplied by |c_i|)
    dof: float = math.inf
    kind: str = "B"
    in_validation: bool = False  # already contained in the held-out validation residual -> not added again
    note: str = ""


def combine(components: Sequence[Component], include_validation_flagged: bool = False):
    """GUM combination: u_c, nu_eff (Welch-Satterthwaite), U(k=2), k95 from t(nu_eff), U95, contributions."""
    use = [c for c in components if include_validation_flagged or not c.in_validation]
    u2 = sum(c.u ** 2 for c in use)
    uc = math.sqrt(u2)
    denom = sum((c.u ** 4) / c.dof for c in use if math.isfinite(c.dof) and c.dof > 0)
    nu_eff = (uc ** 4) / denom if denom > 0 else math.inf
    k95 = float(stats.t.ppf(0.975, nu_eff)) if math.isfinite(nu_eff) else 1.959964
    rows = [{"component": c.name, "type": c.kind, "u": c.u, "dof": c.dof,
             "share_%": (100 * c.u ** 2 / u2) if (u2 > 0 and c in use) else 0.0,
             "counted": c in use, "note": c.note} for c in components]
    return {"u_c": uc, "nu_eff": nu_eff, "U_k2": 2 * uc, "k95": k95, "U95": k95 * uc, "table": pd.DataFrame(rows)}


def creep_bias_rel(t_curve, c_curve, t_cal, t_hold):
    """Relative bias of a contact-period mean caused by creep.

    c(t): measured fractional output change since loading (T05 sustained load), c(0) = 0.
    The calibration is read at t_cal; a task hold lasts t_hold.  bias = mean_{[0,t_hold]} c - c(t_cal).
    """
    t_curve = np.asarray(t_curve, float)
    c_curve = np.asarray(c_curve, float)
    tt = np.linspace(0.0, t_hold, 501)
    mean_c = float(np.trapezoid(np.interp(tt, t_curve, c_curve), tt) / t_hold) if hasattr(np, "trapezoid") \
        else float(np.trapz(np.interp(tt, t_curve, c_curve), tt) / t_hold)
    return mean_c - float(np.interp(t_cal, t_curve, c_curve))


def force_budget(F, R_fsr, beta=1.0, hw=HW_B, V_exc=3.3, u_ref_N=0.0, u_val_N=0.0, nu_val=math.inf,
                 u_eps=0.0, u_adc_noise_V=0.0, n_avg=1, creep_rel=0.0, remount_offset_sd_N=0.0,
                 remount_gain_sd_rel=0.0, n_mount=5, u_temp_rel=0.0, hysteresis_in_validation=True):
    """GUM budget of the installed force reading at level F (N) - T05.

    The in-situ calibration absorbs nominal resistor values and the nominal V_exc; only *changes* after
    calibration (excitation drift, remount, creep, temperature) and the residual misfit remain.
    """
    R_s, R_l = hw["R_s"], hw["R_l"]
    V_in = float(divider_vin(R_fsr, V_exc, R_s, R_l))
    S = float(dlnF_dlnV(R_fsr, beta, R_s, R_l))
    cV = F * S / V_in  # dF/dV_in, N per V
    comps = [
        Component("reference load cell + HX711 (traceable to weights)", u_ref_N, math.inf, "B"),
        Component("calibration: held-out validation residual (misfit, hysteresis, noise)", u_val_N, nu_val, "A",
                  note="contains hysteresis if both branches were validated"),
        Component("ADC quantization 125 uV/sqrt(12)", abs(cV) * LSB_V / math.sqrt(12), math.inf, "B",
                  in_validation=True, note="already inside the validation residual; listed for transparency"),
        Component("ADC noise (averaged over n_avg samples)", abs(cV) * u_adc_noise_V / math.sqrt(max(n_avg, 1)), math.inf, "A",
                  in_validation=True),
        Component("excitation drift eps = dV_exc/V_exc (non-ratiometric)", abs(F * S * u_eps), math.inf, "B",
                  note=f"amplification d lnF/d lnV = {S:.2f} at R_fsr = {R_fsr:.0f} Ohm"),
        Component("creep over the task hold vs calibration read time", abs(F * creep_rel) / math.sqrt(3), math.inf, "B",
                  note="rectangular; correct the bias instead if the creep curve is well characterised"),
        Component("remount: offset", remount_offset_sd_N, n_mount - 1, "A"),
        Component("remount: gain", F * remount_gain_sd_rel, n_mount - 1, "A"),
        Component("temperature (if not tested: Type B, state basis)", F * u_temp_rel, math.inf, "B"),
    ]
    if not hysteresis_in_validation:
        comps.append(Component("hysteresis (not in validation: add half-width/sqrt(3))", 0.0, math.inf, "B"))
    res = combine(comps)
    res.update({"F_N": F, "R_fsr_Ohm": R_fsr, "V_in_V": V_in, "dlnF_dlnV": S, "dF_dV_N_per_V": cV,
                "force_resolution_N_per_LSB": abs(cV) * LSB_V, "hardware": hw["label"]})
    return res


# ----------------------------------------------------------------------------------------------
# Force metrics and Monte Carlo propagation
# ----------------------------------------------------------------------------------------------
def weighted_quantile(x, w, q):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = w > 0
    x, w = x[m], w[m]
    if x.size == 0:
        return float("nan")
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w) - 0.5 * w
    cw /= np.sum(w)
    return float(np.interp(q, cw, x))


def contact_mask(F, F_on, F_off):
    """Hysteretic contact detection: on when F >= F_on, off when F < F_off (F_off < F_on)."""
    F = np.asarray(F, float)
    m = np.zeros(F.size, bool)
    state = False
    for k, f in enumerate(F):
        if not state and f >= F_on:
            state = True
        elif state and f < F_off:
            state = False
        m[k] = state
    return m


def force_metrics(t, F, F_on, F_off, max_gap=0.1):
    """M08 (contact-period mean force), F95 (time-weighted 95th percentile over contact), FTI, T_contact."""
    w = interval_weights(t, max_gap)
    c = contact_mask(F, F_on, F_off)
    wc = w * c
    Tc = float(np.sum(wc))
    if Tc <= 0:
        return {"M08": float("nan"), "F95": float("nan"), "FTI": 0.0, "T_contact": 0.0}
    FTI = float(np.sum(wc * F))
    return {"M08": FTI / Tc, "F95": weighted_quantile(F, wc, 0.95), "FTI": FTI, "T_contact": Tc}


def mc_force(t, F_meas, F_on, F_off, R_fsr_of_F, beta=1.0, hw=HW_B, n_mc=2000, u_val_N=0.0, u_eps=0.0,
             remount_offset_sd_N=0.0, remount_gain_sd_rel=0.0, creep_curve=None, t_cal=None, seed=3,
             sources=None):
    """GUM Supplement-1 style propagation of systematic force errors to M08, F95, FTI.

    The measured trace is the plug-in estimate; each draw applies one realisation of every selected source
    (offset/gain remount, calibration misfit as offset, excitation drift amplified by Prop. 4, creep relative to
    the calibration read time) and recomputes contact segmentation.  `sources` restricts the active sources
    (for one-at-a-time ablation).  Returns dict of arrays and summary statistics.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float)
    F_meas = np.asarray(F_meas, float)
    all_src = {"remount", "calibration", "excitation", "creep"}
    act = all_src if sources is None else set(sources)
    S = dlnF_dlnV(np.maximum(R_fsr_of_F(np.maximum(F_meas, 1e-6)), 1.0), beta, hw["R_s"], hw["R_l"])
    # time since contact onset (for creep)
    on = contact_mask(F_meas, F_on, F_off)
    since = np.zeros_like(t)
    for k in range(1, t.size):
        since[k] = since[k - 1] + (t[k] - t[k - 1]) if (on[k] and on[k - 1]) else 0.0
    creep_rel = np.zeros_like(t)
    if creep_curve is not None and "creep" in act:
        tc, cc = creep_curve
        c_cal = float(np.interp(t_cal, tc, cc)) if t_cal is not None else 0.0
        creep_rel = np.interp(since, tc, cc) - c_cal
    out = {"M08": np.empty(n_mc), "F95": np.empty(n_mc), "FTI": np.empty(n_mc)}
    for n in range(n_mc):
        a = rng.normal(0, remount_offset_sd_N) if "remount" in act else 0.0
        g = 1.0 + (rng.normal(0, remount_gain_sd_rel) if "remount" in act else 0.0)
        e_cal = rng.normal(0, u_val_N) if "calibration" in act else 0.0
        eps = rng.normal(0, u_eps) if "excitation" in act else 0.0
        F = (a + e_cal) + g * F_meas * (1.0 + S * eps) / (1.0 + creep_rel)
        m = force_metrics(t, F, F_on, F_off)
        for key in out:
            out[key][n] = m[key]
    base = force_metrics(t, F_meas, F_on, F_off)
    summ = {k: {"base": base[k], "mean": float(np.nanmean(v)), "u": float(np.nanstd(v, ddof=1)),
                "ci95": (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975)))} for k, v in out.items()}
    return {"draws": out, "summary": summ}


def mc_angular(t, omega, n_mc=2000, scale_sd=0.0, cross_sd=0.0, bias_u=0.0, threshold=None, seed=4, max_gap=0.1):
    """Propagate systematic gyro errors (scale, cross-axis, residual bias) to M04, M05, Q_low.

    The noise floor is handled analytically (Lemma 2) and is not re-added here, because the measured trace
    already contains one realisation of the noise.
    """
    rng = np.random.default_rng(seed)
    omega = np.asarray(omega, float)
    keys = ["M04", "M05"] + (["Q_low"] if threshold is not None else [])
    out = {k: np.empty(n_mc) for k in keys}
    for n in range(n_mc):
        E = np.diag(rng.normal(0, scale_sd, 3)) if scale_sd > 0 else np.zeros((3, 3))
        if cross_sd > 0:
            off = rng.normal(0, cross_sd, (3, 3))
            np.fill_diagonal(off, 0.0)
            E = E + off
        b = rng.normal(0, bias_u, 3) if bias_u > 0 else np.zeros(3)
        om = omega @ (np.eye(3) + E).T + b
        m = angular_metrics(t, om, threshold, max_gap)
        out["M04"][n], out["M05"][n] = m.M04, m.M05
        if threshold is not None:
            out["Q_low"][n] = m.Q_low
    base = angular_metrics(t, omega, threshold, max_gap)
    bvals = {"M04": base.M04, "M05": base.M05, "Q_low": base.Q_low}
    summ = {k: {"base": bvals[k], "mean": float(np.mean(v)), "u": float(np.std(v, ddof=1)),
                "ci95": (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))} for k, v in out.items()}
    return {"draws": out, "summary": summ}


# ----------------------------------------------------------------------------------------------
# Synthetic generators (self-test only)
# ----------------------------------------------------------------------------------------------
def synthetic_motion(T=60.0, fs=100.0, seed=0, quiet_frac=0.35):
    """Synthetic peg-transfer-like angular velocity (deg/s): quiescent holds + bidirectional movements."""
    rng = np.random.default_rng(seed)
    n = int(T * fs)
    t = np.arange(n) / fs
    om = np.zeros((n, 3))
    k = 0
    while k < n:
        L = int(rng.uniform(0.4, 2.0) * fs)
        if rng.uniform() > quiet_frac:
            axis = rng.normal(0, 1, 3)
            axis /= np.linalg.norm(axis)
            amp = rng.uniform(20, 120) * rng.choice([-1, 1])
            prof = np.sin(np.linspace(0, np.pi, min(L, n - k))) * amp
            om[k:k + prof.size] = prof[:, None] * axis[None, :]
        k += L
    return t, om


def _rand_rot(rng):
    q = rng.normal(0, 1, 4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def selftest(verbose=True):
    rng = np.random.default_rng(42)
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    # Lemma 2 mean vs Monte Carlo
    errs = []
    for lam in [0.0, 0.3, 1.0, 2.0, 5.0, 20.0]:
        n = rng.normal(0, 1, (400000, 3))
        mc = np.mean(np.linalg.norm(n + np.array([lam, 0, 0]), axis=1))
        errs.append(abs(nc_chi3_mean(lam, 1.0) / mc - 1))
    check("Lemma 2: noncentral chi3 mean vs MC", max(errs) < 3e-3, f"max rel err {max(errs):.2e}")
    check("Lemma 2: rest value sqrt(8/pi)", abs(nc_chi3_mean(0.0, 1.0) - 1.5957691) < 1e-6, f"{float(nc_chi3_mean(0.0, 1.0)):.6f}")
    big = float(nc_chi3_mean(50.0, 1.0))
    check("Lemma 2: motion limit a + sigma^2/a", abs(big - (50 + 1 / 50)) < 1e-6, f"{big:.6f}")
    # CDF vs MC
    n = rng.normal(0, 1, (400000, 3))
    cdf_mc = np.mean(np.linalg.norm(n + np.array([1.5, 0, 0]), axis=1) < 2.0)
    check("Lemma 2: CDF vs MC", abs(float(nc_chi3_cdf(2.0, 1.5, 1.0)) - cdf_mc) < 3e-3, f"{float(nc_chi3_cdf(2.0,1.5,1.0)):.4f} vs {cdf_mc:.4f}")
    # pdf integrates to cdf
    rr = np.linspace(0, 2.0, 4001)
    pdf_int = np.trapezoid(nc_chi3_pdf(rr, 1.5, 1.0), rr) if hasattr(np, "trapezoid") else np.trapz(nc_chi3_pdf(rr, 1.5, 1.0), rr)
    check("Lemma 2: PDF integrates to CDF", abs(pdf_int - float(nc_chi3_cdf(2.0, 1.5, 1.0))) < 1e-4)

    # Prediction of M04 under injected noise on a synthetic trace
    t, om = synthetic_motion(seed=1)
    w = interval_weights(t, 0.1)
    sp = np.linalg.norm(om, axis=1)
    sig = 0.8
    emp = np.mean([angular_metrics(t, om + rng.normal(0, sig, om.shape)).M04 for _ in range(40)])
    pred = predict_m04_noise(sp, w, sig)
    check("Prop. 2: predicted vs empirical M04 under noise", abs(pred / emp - 1) < 2e-3, f"pred {pred:.4f} emp {emp:.4f}")
    th = 3.0
    empq = np.mean([angular_metrics(t, om + rng.normal(0, sig, om.shape), th).Q_low for _ in range(40)])
    predq = predict_qlow_noise(sp, w, sig, th)
    check("Prop. 3: predicted vs empirical Q_low under noise", abs(predq - empq) < 3e-3, f"pred {predq:.4f} emp {empq:.4f}")
    cth, csig = qlow_sensitivities(sp, w, sig, th, nbins=None)
    h = 0.01
    fd = (predict_qlow_noise(sp, w, sig, th + h) - predict_qlow_noise(sp, w, sig, th - h)) / (2 * h)
    check("Prop. 3: c_th equals finite difference", abs(cth - fd) < 1e-3 * max(1.0, abs(fd)), f"c_th {cth:.4f} fd {fd:.4f}")
    check("Prop. 3: misclassified rest fraction k=3", abs(misclassified_rest_fraction(3.0) - (1 - stats.chi2.cdf(9, 3))) < 1e-12,
          f"{misclassified_rest_fraction(3.0):.4f}")

    # Lemma 3: |dM04| <= |b| for any constant bias
    ok3 = True
    for _ in range(50):
        b = rng.normal(0, 0.5, 3)
        d = abs(angular_metrics(t, om + b).M04 - angular_metrics(t, om).M04)
        ok3 &= d <= np.linalg.norm(b) + 1e-12
    re = residual_bias_effect(om, w, 0.1)
    check("Lemma 3: |dM04| <= |b_r| (Lipschitz) and first-order coefficient <= conservative", ok3 and
          re["u_M04_first_order"] <= re["u_M04_conservative"], f"|ubar| = {re['mean_unit_direction_norm']:.3f}")

    # Proposition 1
    ok_bounds, ok_first = True, True
    for trial in range(200):
        E = np.diag(rng.normal(0, 0.03, 3)) + (rng.normal(0, 0.005, (3, 3)) * (1 - np.eye(3)))
        smin, smax, _ = mount_bound_exact(E)
        t2, omt = synthetic_motion(T=20, seed=trial + 10)
        w2 = interval_weights(t2, 0.1)
        true = angular_metrics(t2, omt).M04
        for _ in range(3):
            C = _rand_rot(rng)
            meas = angular_metrics(t2, (omt @ C.T) @ (np.eye(3) + E).T).M04
            ratio = meas / true
            if not (smin - 1e-12 <= ratio <= smax + 1e-12):
                ok_bounds = False
        Ed = np.diag(rng.normal(0, 0.03, 3))
        C = _rand_rot(rng)
        oms = omt @ C.T
        wts = axis_weights(oms, w2)
        pred_rel = float(np.sum(wts * np.diag(Ed)))
        meas_rel = angular_metrics(t2, oms @ (np.eye(3) + Ed).T).M04 / true - 1
        # exact remainder (Prop. 1(4)): 0 <= r <= (d_max - d_min)^2 / (8 (1 + d_min))
        rem = meas_rel - pred_rel
        dd = np.diag(Ed)
        if not (-1e-12 <= rem <= (dd.max() - dd.min()) ** 2 / (8 * (1 + dd.min())) + 1e-12):
            ok_first = False
    check("Prop. 1(1),(2): M04 ratio within [sigma_min, sigma_max] for random mounts", ok_bounds)
    check("Prop. 1(4): convex combination of axis scale errors + exact remainder bound", ok_first)
    E = np.array([[0.02, 0.004, 0.0], [0.001, -0.01, 0.0], [0.0, 0.0, 0.005]])
    Ea = E + np.array([[0, 0.003, 0], [-0.003, 0, 0], [0, 0, 0]])  # add antisymmetric part
    check("Prop. 1(3): antisymmetric part has no first-order effect",
          abs(mount_bound_first_order(E)[2] - mount_bound_first_order(Ea)[2]) < 1e-12 and
          abs(mount_bound_exact(Ea)[2] - mount_bound_first_order(Ea)[2]) < 2e-4)

    # T02 estimation with reversal
    E_true = np.array([[0.020, 0.004, -0.002], [0.003, -0.010, 0.001], [-0.001, 0.002, 0.008]])
    rows = []
    for axis, j in zip("xyz", range(3)):
        tilt = rng.normal(0, math.radians(1.0), 3)
        tilt[j] = 0.0
        for placement in (0.0, 180.0):
            tl = tilt if placement == 0.0 else -tilt
            for sgn in (1, -1):
                for th_ref in (90.0, 180.0, 360.0, 720.0):
                    for rep in range(3):
                        u = np.eye(3)[j] + tl
                        u /= np.linalg.norm(u)
                        vec = (np.eye(3) + E_true) @ u * th_ref * sgn + rng.normal(0, 0.02, 3)
                        rows.append({"pod": "L", "axis": axis, "direction": sgn, "placement_deg": placement,
                                     "theta_ref_deg": th_ref, "Theta_x_deg": vec[0], "Theta_y_deg": vec[1], "Theta_z_deg": vec[2]})
    est = estimate_E_from_T02(pd.DataFrame(rows), n_mc=2000)["L"]
    d_err = max(abs(est["diag_E_jj"][f"{a}+"]["mean"] - E_true[j, j]) for a, j in zip("xyz", range(3)))
    o_err = max(abs(est["offdiag_E_ij"][f"E_{'xyz'[i]}{'xyz'[j]}"]["value"] - E_true[i, j]) for i in range(3) for j in range(3) if i != j)
    check("T02: diagonal E_jj recovered from norm (tilt-insensitive)", d_err < 1e-3, f"max err {d_err:.1e}")
    check("T02: off-diagonal E_ij recovered by reversal", o_err < 1e-3, f"max err {o_err:.1e}")
    true_bound = mount_bound_first_order(E_true)[2]
    check("T02: Gershgorin Delta_E >= first-order spread", est["Delta_E_central"] >= true_bound - 1e-4,
          f"Gershgorin {est['Delta_E_central']:.4f} vs eig {true_bound:.4f}")

    # T04 verification
    rows = []
    for mount in range(5):
        C = _rand_rot(rng)
        for rep in range(3):
            u = C @ np.array([0, 0, 1.0])
            vec = (np.eye(3) + E_true) @ u * 360.0 + rng.normal(0, 0.05, 3)
            rows.append({"pod": "L", "mount": mount, "rep": rep, "theta_ref_deg": 360.0,
                         "Theta_x_deg": vec[0], "Theta_y_deg": vec[1], "Theta_z_deg": vec[2]})
    v = t04_verify(pd.DataFrame(rows), est["Delta_E_95"])["L"]
    check("T04: random remounts within Prop. 1 bound", v["verdict"].startswith("PASS"), f"spread {v['spread_r']:.4f} <= {v['allowed_spread']:.4f}")
    rows_slip = [dict(r) for r in rows]
    for r in rows_slip:
        if r["mount"] == 2:
            for key in ("Theta_x_deg", "Theta_y_deg", "Theta_z_deg"):
                r[key] *= 1.06  # 6 % spurious extra rotation: pod slip
    v2 = t04_verify(pd.DataFrame(rows_slip), est["Delta_E_95"])["L"]
    check("T04: simulated slip is flagged", v2["verdict"].startswith("FLAG"))

    # Plate analytic vs MC
    geo = PlateGeometry()
    ok = True
    for th in (90.0, 180.0, 360.0, 720.0):
        a = plate_reference_u(th, geo)["u_ref_deg"]
        m = plate_mc(th, geo, n=200000)
        ok &= abs(a / m - 1) < 0.02
    check("Plate: analytic u_ref equals MC (rectangular model)", ok)

    # Allan deviation on white noise
    sig, fs = 0.05, 100.0
    x = rng.normal(0, sig, 200000)
    taus, ad = allan_deviation(x, fs)
    sel = (taus >= 0.05) & (taus <= 20)
    ratio = ad[sel] / (sig / np.sqrt(taus[sel] * fs))
    check("T01: Allan deviation of white noise = sigma/sqrt(tau fs)", np.all(np.abs(ratio - 1) < 0.12), f"ratio {ratio.min():.3f}-{ratio.max():.3f}")

    # Proposition 4 finite difference
    Rf, Ve = 6000.0, 3.3
    V0 = float(divider_vin(Rf, Ve))
    G = lambda V: 1.0 / fsr_from_vin(V, Ve)
    h = 1e-6
    num = (math.log(G(V0 * (1 + h))) - math.log(G(V0 * (1 - h)))) / (2 * h)
    check("Prop. 4: d lnG/d lnV = 1 + (R_s+R_l)/R_fsr", abs(num - float(dlnF_dlnV(Rf))) < 1e-5, f"{num:.5f} vs {float(dlnF_dlnV(Rf)):.5f}")

    # GUM combine
    comps = [Component("a", 0.3, 4, "A"), Component("b", 0.4)]
    c = combine(comps)
    nu_ws = 0.5 ** 4 / (0.3 ** 4 / 4)
    check("GUM: u_c and Welch-Satterthwaite", abs(c["u_c"] - 0.5) < 1e-12 and abs(c["nu_eff"] - nu_ws) < 1e-9,
          f"nu_eff {c['nu_eff']:.1f}")

    # Force MC on a synthetic trace
    tt = np.arange(0, 20, 1 / 250.0)
    Ft = np.where((tt % 4) < 2.0, 4.0 + 0.5 * np.sin(2 * np.pi * tt), 0.0)
    Rof = lambda F: 60000.0 / F  # synthetic conductance-linear FSR: R = 60 kOhm*N / F
    mc = mc_force(tt, Ft, F_on=0.5, F_off=0.3, R_fsr_of_F=Rof, n_mc=300, remount_offset_sd_N=0.05,
                  remount_gain_sd_rel=0.02, u_eps=0.002)
    check("Force MC: runs and gives finite u(M08)", math.isfinite(mc["summary"]["M08"]["u"]) and mc["summary"]["M08"]["u"] > 0,
          f"u(M08) {mc['summary']['M08']['u']:.4f} N")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\nselftest: {len(results) - n_fail}/{len(results)} checks passed (SYNTHETIC data; software check only)")
    return n_fail == 0


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    pp = sub.add_parser("plate")
    for name, default in [("R", 62.0), ("u-chord", 0.03), ("d-pin", 3.00), ("D-base", 3.20), ("D-rotor", 3.20),
                          ("D-shaft-hole", 12.5), ("D-shaft", 12.0), ("preload-factor", 1.0)]:
        pp.add_argument(f"--{name}", type=float, default=default)
    p2 = sub.add_parser("t02")
    p2.add_argument("--csv", required=True)
    p2.add_argument("--tilt-bound-deg", type=float, default=None)
    p4 = sub.add_parser("t04")
    p4.add_argument("--csv", required=True)
    p4.add_argument("--delta-E", type=float, required=True, help="Delta_E_95 from the t02 command")
    p4.add_argument("--u-ref-rel", type=float, default=0.0, help="mount-independent reference part only (normally 0)")
    p4.add_argument("--alpha", type=float, default=0.05)
    pf = sub.add_parser("force")
    for name, default in [("F", 5.0), ("R-fsr", 6000.0), ("beta", 1.0), ("V-exc", 3.3), ("u-ref", 0.0), ("u-val", 0.0),
                          ("nu-val", math.inf), ("u-eps", 0.0), ("u-adc-noise-V", 0.0), ("n-avg", 1), ("creep-rel", 0.0),
                          ("remount-offset-sd", 0.0), ("remount-gain-sd", 0.0), ("n-mount", 5), ("u-temp-rel", 0.0)]:
        pf.add_argument(f"--{name}", type=float, default=default)
    pf.add_argument("--hw", choices=["A", "B"], default="B")
    pa = sub.add_parser("allan")
    pa.add_argument("--csv", required=True, help="columns gx, gy, gz in deg/s (static record)")
    pa.add_argument("--fs", type=float, required=True)
    a = p.parse_args(argv)

    if a.cmd == "selftest":
        return 0 if selftest() else 1
    if a.cmd == "plate":
        geo = PlateGeometry(R=a.R, u_chord=a.u_chord, d_pin=a.d_pin, D_base=a.D_base, D_rotor=a.D_rotor,
                            D_shaft_hole=a.D_shaft_hole, D_shaft=a.D_shaft, preload_factor=a.preload_factor)
        print(pd.DataFrame([plate_reference_u(th, geo) for th in (90.0, 180.0, 360.0, 720.0)]).to_string(index=False))
        return 0
    if a.cmd == "t02":
        print(json.dumps(estimate_E_from_T02(pd.read_csv(a.csv), a.tilt_bound_deg), indent=2, default=str))
        return 0
    if a.cmd == "t04":
        print(json.dumps(t04_verify(pd.read_csv(a.csv), a.delta_E, a.u_ref_rel, a.alpha), indent=2, default=str))
        return 0
    if a.cmd == "force":
        res = force_budget(a.F, a.R_fsr, a.beta, HW_B if a.hw == "B" else HW_A, a.V_exc, a.u_ref, a.u_val, a.nu_val,
                           a.u_eps, a.u_adc_noise_V, int(a.n_avg), a.creep_rel, a.remount_offset_sd, a.remount_gain_sd,
                           int(a.n_mount), a.u_temp_rel)
        print(res["table"].to_string(index=False))
        print({k: v for k, v in res.items() if k != "table"})
        return 0
    if a.cmd == "allan":
        d = pd.read_csv(a.csv)
        for ax in ("gx", "gy", "gz"):
            taus, ad = allan_deviation(d[ax].to_numpy(float), a.fs)
            print(ax, allan_summary(taus, ad))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(_main())
