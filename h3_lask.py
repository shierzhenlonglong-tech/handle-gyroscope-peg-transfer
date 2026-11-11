"""H3 外部一致性（探索性）：LASK 公开子集（37 个试次，横断面）。
由电磁跟踪四元数求器械角速度，计算与主分析相同的指标，与手术经验做秩相关。
用法：python h3_lask.py <LASK/extracted 目录>
注意：LASK 采样率约 14.7 Hz（A、B）和约 27 Hz（C），统一重采样到 14 Hz；与 SPD-FLS1 的 31.25 Hz 陀螺不可直接比数值，只比方向。"""
import sys, glob, os
import numpy as np, pandas as pd
from scipy import signal, stats
from extract_features import sparc, pauses

FS = 14.0
root = sys.argv[1]
part = pd.read_csv(glob.glob(os.path.join(root, "*", "participants.csv"))[0])


def ang_vel(t, q):
    """Body-frame angular velocity (deg/s) from unit quaternions (w, x, y, z) at times t."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    d = np.sum(q[1:] * q[:-1], axis=1)
    q1 = q[1:] * np.sign(d)[:, None]
    q0 = q[:-1]
    # relative rotation q0^-1 * q1
    w = q0[:, 0] * q1[:, 0] + np.sum(q0[:, 1:] * q1[:, 1:], axis=1)
    v = q0[:, 0:1] * q1[:, 1:] - q1[:, 0:1] * q0[:, 1:] - np.cross(q0[:, 1:], q1[:, 1:])
    ang = 2 * np.arctan2(np.linalg.norm(v, axis=1), np.abs(w))
    axis = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12) * np.sign(w)[:, None]
    dt = np.diff(t)
    return (t[1:] + t[:-1]) / 2, np.degrees(axis * (ang / dt)[:, None])


def feats(t, om):
    tg = np.arange(t[0], t[-1], 1 / FS)
    y = np.column_stack([np.interp(tg, t, om[:, i]) for i in range(3)])
    y = np.round(y / 0.5729578) * 0.5729578          # same print resolution as SPD-FLS1
    s = np.linalg.norm(y, axis=1)
    ba = signal.butter(4, 5.0 / (FS / 2))
    npk = signal.find_peaks(signal.filtfilt(*ba, s), prominence=5.0)[0].size
    n_p, t_p = pauses(s, FS)
    T = s.size / FS
    return {"w_mean": s.mean(), "w_p95": np.quantile(s, 0.95), "qlow_5": np.mean(s < 5), "pause_frac": t_p / T,
            "sparc": sparc(s, FS, fc=5.0), "peaks_per_s": npk / T}


rows = []
for f in sorted(glob.glob(os.path.join(root, "*", "*kinematics.csv"))):
    k = pd.read_csv(f)
    six = "time_ms" in k.columns
    tools = ("fenestrated", "curved") if six else ("tool1", "tool2")
    qn = ("q0", "q1", "q2", "q3") if six else ("qw", "qx", "qy", "qz")
    for tid, g in k.groupby("trial_id"):
        t = g["time_ms"].values / 1000.0 if six else g["time_s"].values
        per_tool = []
        for tool in tools:
            q = g[[f"{tool}_{c}" for c in qn]].values.astype(float)
            ok = np.isfinite(q).all(axis=1) & (np.linalg.norm(q, axis=1) > 0.5)
            tt, qq = t[ok], q[ok]
            # make time strictly increasing: a concatenated trial restarts its clock, so rebuild time from positive steps
            dtt = np.diff(tt)
            dtt[dtt <= 1e-4] = np.median(dtt[dtt > 1e-4])
            tt = tt[0] + np.concatenate([[0.0], np.cumsum(dtt)])
            tm, om = ang_vel(tt, qq)
            good = np.linalg.norm(om, axis=1) < 2000          # tracker glitches
            per_tool.append(feats(tm[good], om[good]))
        r = {k_: np.mean([p[k_] for p in per_tool]) for k_ in per_tool[0]}
        dt_all = np.diff(t)
        r.update({"trial_id": tid, "duration_s": float(np.sum(np.where(dt_all > 1e-4, dt_all, np.median(dt_all[dt_all > 1e-4])))),
                  "valid_frac": float(np.mean(ok))})
        rows.append(r)
d = pd.DataFrame(rows).merge(part, on="trial_id")
d["log_exp"] = np.log10(1 + d.procedures_lifetime)
d.to_csv("../07_results/h3_lask_trials.csv", index=False)
print(d[["dataset", "trial_id", "procedures_lifetime", "duration_s", "w_mean", "w_p95", "qlow_5", "pause_frac", "sparc", "peaks_per_s"]].round(2).to_string())
rng = np.random.default_rng(20261111)
out = []
for c in ["duration_s", "w_mean", "w_p95", "qlow_5", "pause_frac", "sparc", "peaks_per_s"]:
    rho, p = stats.spearmanr(d.log_exp, d[c])
    bs = [stats.spearmanr(*d.iloc[rng.integers(0, len(d), len(d))][["log_exp", c]].values.T)[0] for _ in range(2000)]
    # within-cohort: rank within dataset, then pooled correlation of ranks (controls cohort/tool differences)
    rk = d.groupby("dataset")[["log_exp", c]].rank(pct=True)
    rw, pw = stats.spearmanr(rk.log_exp, rk[c])
    # partial rank correlation controlling for duration
    if c != "duration_s":
        R = d[["log_exp", c, "duration_s"]].rank()
        res = lambda a: a - np.polyval(np.polyfit(R.duration_s, a, 1), R.duration_s)
        rp, pp = stats.pearsonr(res(R.log_exp), res(R[c]))
    else:
        rp = pp = np.nan
    out.append({"metric": c, "n": len(d), "rho": rho, "p": p, "ci_lo": np.quantile(bs, 0.025), "ci_hi": np.quantile(bs, 0.975),
                "rho_within_cohort": rw, "p_within": pw, "partial_rho_given_duration": rp, "p_partial": pp})
o = pd.DataFrame(out)
o.to_csv("../07_results/h3_lask_summary.csv", index=False)
print(o.round(3).to_string())
