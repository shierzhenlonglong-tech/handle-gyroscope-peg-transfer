"""事后补充分析（2026-11-11，非预先指定，论文中标为 post hoc）：
(1) 逐试次信度：每人最后 20 个有效试次去线性趋势后的单因素 ICC、SEM、MDC95，以及达到 0.80 信度所需平均的试次数；
(2) 会话间离线变化：下一会话前 3 个试次均值 - 上一会话后 3 个试次均值；与会话内变化对比；
(3) 后半程斜率：每人后一半试次上对 log(序号) 的斜率。"""
import warnings
import numpy as np, pandas as pd
from scipy import stats
from h1_mixed_models import load, PRIMARY
warnings.filterwarnings("ignore")
df = load()
df["logT"] = np.log(df.T_log)
FE = ["logT"] + PRIMARY
pd.set_option("display.width", 250)

# ---------- (1) reliability on the last 20 trials ----------
rows = []
for hand in "LR":
    h = df[df.hand == hand]
    for f in FE:
        res, means, gains = [], [], []
        for pid, g in h.groupby("participant"):
            g = g.sort_values("order").dropna(subset=[f])
            if len(g) < 30:
                continue
            y = g[f].values[-20:]
            x = np.arange(20)
            r = y - np.polyval(np.polyfit(x, y, 1), x)
            res.append(r)
            means.append(y.mean())
            gains.append(g[f].values[-10:].mean() - g[f].values[:10].mean())
        res = np.array(res)
        msw = np.sum(res ** 2) / (res.size - 2 * len(res))        # within-person variance (trend removed)
        var_b = max(np.var(means, ddof=1) - msw / 20, 0.0)
        icc = var_b / (var_b + msw)
        sem = np.sqrt(msw)
        mdc = 1.96 * np.sqrt(2) * sem
        gain = np.mean(np.abs(gains))
        k80 = 0.8 * (1 - icc) / (0.2 * icc) if icc > 0 else np.inf
        k_mdc = (mdc / gain) ** 2                                 # trials per block so that MDC95 of a block mean equals the mean practice gain
        rows.append({"hand": hand, "feature": f, "n_subj": len(res), "ICC1": icc, "SEM": sem, "MDC95_single": mdc,
                     "mean_abs_gain": gain, "gain_over_MDC": gain / mdc, "k_for_ICC0.8": k80, "k_gain_eq_MDC": k_mdc})
rel = pd.DataFrame(rows)
rel.to_csv("../07_results/x1_reliability.csv", index=False)
print(rel.round(3).to_string())

# ---------- (2) offline (between-session) vs online (within-session) change ----------
rows = []
for hand in "LR":
    h = df[df.hand == hand]
    for f in FE:
        sd = h[f].std()
        off, on = {}, {}
        for pid, g in h.groupby("participant"):
            g = g.sort_values("order").dropna(subset=[f])
            ss = [s for s, gg in g.groupby("session_idx") if len(gg) >= 6]
            o1, o2 = [], []
            for s in ss:
                a = g[g.session_idx == s][f].values
                o2.append((a[-3:].mean() - a[:3].mean()) / sd)
                if s + 1 in ss:
                    b = g[g.session_idx == s + 1][f].values
                    o1.append((b[:3].mean() - a[-3:].mean()) / sd)
            if o1:
                off[pid] = np.mean(o1)
            if o2:
                on[pid] = np.mean(o2)
        a, b = np.array(list(off.values())), np.array(list(on.values()))
        rows.append({"hand": hand, "feature": f, "n_off": len(a), "offline_mean": a.mean(), "offline_p": stats.wilcoxon(a)[1],
                     "n_on": len(b), "online_mean": b.mean(), "online_p": stats.wilcoxon(b)[1],
                     "offline_frac_same_sign_as_online": float(np.mean(np.sign(a) == np.sign(b.mean())))})
ret = pd.DataFrame(rows)
ret.to_csv("../07_results/x2_offline_online.csv", index=False)
print(ret.round(3).to_string())

# ---------- (3) second-half slopes ----------
rows = []
for hand in "LR":
    h = df[df.hand == hand]
    for f in FE:
        sd = h[f].std()
        s1, s2 = [], []
        for pid, g in h.groupby("participant"):
            g = g.sort_values("order").dropna(subset=[f])
            if len(g) < 30:
                continue
            m = len(g) // 2
            for part, acc in ((g.iloc[:m], s1), (g.iloc[m:], s2)):
                acc.append(np.polyfit(np.log(part.order), part[f] / sd, 1)[0])
        s1, s2 = np.array(s1), np.array(s2)
        rows.append({"hand": hand, "feature": f, "n": len(s2), "slope_first_half": np.median(s1), "slope_second_half": np.median(s2),
                     "p_second_half": stats.wilcoxon(s2)[1], "n_same_sign_second": int(np.sum(np.sign(s2) == np.sign(np.median(s1))))})
lat = pd.DataFrame(rows)
lat.to_csv("../07_results/x3_second_half.csv", index=False)
print(lat.round(3).to_string())
