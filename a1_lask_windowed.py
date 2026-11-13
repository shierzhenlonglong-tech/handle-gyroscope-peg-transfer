"""补充方案 A1（描述性）：LASK 上的窗口化 SPARC / LDLJ（14 Hz，56 点 = 4 s 窗口，SPARC 上限 5 Hz）与经验的 Spearman 相关。
用法：python a1_lask_windowed.py <LASK 解压目录>；需先运行 h3_lask.py（读取 results/h3_lask_trials.csv）。输出 results/a1_lask_windowed.csv。"""
import sys, glob, os
import numpy as np, pandas as pd
from scipy import stats
from extract_features import sparc, ldlj
import os as _os
_os.makedirs("results", exist_ok=True)
_os.makedirs("figures", exist_ok=True)
FS, N = 14.0, 56
root = sys.argv[1]
part = pd.read_csv(glob.glob(os.path.join(root, "*", "participants.csv"))[0])


def ang_speed(t, q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    d = np.sum(q[1:] * q[:-1], axis=1)
    q1, q0 = q[1:] * np.sign(d)[:, None], q[:-1]
    w = q0[:, 0] * q1[:, 0] + np.sum(q0[:, 1:] * q1[:, 1:], axis=1)
    v = q0[:, 0:1] * q1[:, 1:] - q1[:, 0:1] * q0[:, 1:] - np.cross(q0[:, 1:], q1[:, 1:])
    ang = 2 * np.arctan2(np.linalg.norm(v, axis=1), np.abs(w))
    axis = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12) * np.sign(w)[:, None]
    return (t[1:] + t[:-1]) / 2, np.degrees(axis * (ang / np.diff(t))[:, None])


rows = []
for f in sorted(glob.glob(os.path.join(root, "*", "*kinematics.csv"))):
    k = pd.read_csv(f)
    six = "time_ms" in k.columns
    tools = ("fenestrated", "curved") if six else ("tool1", "tool2")
    qn = ("q0", "q1", "q2", "q3") if six else ("qw", "qx", "qy", "qz")
    for tid, g in k.groupby("trial_id"):
        t = g["time_ms"].values / 1000.0 if six else g["time_s"].values
        vals = []
        for tool in tools:
            qq = g[[f"{tool}_{c}" for c in qn]].values.astype(float)
            ok = np.isfinite(qq).all(axis=1) & (np.linalg.norm(qq, axis=1) > 0.5)
            tt, qq = t[ok], qq[ok]
            dtt = np.diff(tt)
            dtt[dtt <= 1e-4] = np.median(dtt[dtt > 1e-4])
            tt = tt[0] + np.concatenate([[0.0], np.cumsum(dtt)])
            tm, om = ang_speed(tt, qq)
            good = np.linalg.norm(om, axis=1) < 2000
            tm, om = tm[good], om[good]
            tg = np.arange(tm[0], tm[-1], 1 / FS)
            y = np.column_stack([np.interp(tg, tm, om[:, i]) for i in range(3)])
            s = np.linalg.norm(np.round(y / 0.5729578) * 0.5729578, axis=1)
            sp = [sparc(s[i:i + N], FS, fc=5.0) for i in range(0, s.size - N + 1, N)]
            ld = [ldlj(s[i:i + N], FS) for i in range(0, s.size - N + 1, N)]
            vals.append((np.nanmean(sp), np.nanmean([x for x in ld if np.isfinite(x)])))
        rows.append({"trial_id": tid, "sparc_w4": np.mean([v[0] for v in vals]), "ldlj_w4": np.mean([v[1] for v in vals])})
d = pd.DataFrame(rows).merge(part, on="trial_id").merge(pd.read_csv("results/h3_lask_trials.csv")[["trial_id", "duration_s"]], on="trial_id")
d["log_exp"] = np.log10(1 + d.procedures_lifetime)
rng = np.random.default_rng(20261111)
out = []
for c in ("sparc_w4", "ldlj_w4"):
    rho, p = stats.spearmanr(d.log_exp, d[c])
    bs = [stats.spearmanr(*d.iloc[rng.integers(0, len(d), len(d))][["log_exp", c]].values.T)[0] for _ in range(2000)]
    out.append({"metric": c, "n": len(d), "rho_experience": rho, "p": p, "ci_lo": np.quantile(bs, .025), "ci_hi": np.quantile(bs, .975),
                "rho_duration": stats.spearmanr(d.duration_s, d[c])[0]})
o = pd.DataFrame(out)
o.to_csv("results/a1_lask_windowed.csv", index=False)
print(o.round(3).to_string())
