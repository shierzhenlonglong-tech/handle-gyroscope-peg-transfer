"""补充方案 A1 的检验：窗口化 SPARC / LDLJ 的时长系数、练习效应、SRM、单试次 ICC。输出 results/a1_windowed.csv。
若仓库根目录有 features_w_part{1,2,3}.csv（extract_windowed.py 的输出），则由其合并并写出 trial_features_windowed.csv；
否则直接读取随仓库发布的 trial_features_windowed.csv。"""
import warnings
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
from h1_mixed_models import load, fit
import os as _os
_os.makedirs("results", exist_ok=True)
_os.makedirs("figures", exist_ok=True)
warnings.filterwarnings("ignore")
if all(_os.path.exists(f"features_w_part{i}.csv") for i in (1, 2, 3)):
    df = load()
    w = pd.concat([pd.read_csv(f"features_w_part{i}.csv", dtype={"session": str}) for i in (1, 2, 3)])
    df = df.merge(w.drop(columns=["session"]), on=["participant", "trial", "hand"], how="left")
    df.to_csv("trial_features_windowed.csv", index=False)
else:
    df = pd.read_csv("trial_features_windowed.csv", dtype={"session": str})   # 已发布的合并表（= load() 的结果并入窗口化列）
FE = ["sparc", "sparc_w2", "sparc_w4", "sparc_w8", "ldlj", "ldlj_w2", "ldlj_w4", "ldlj_w8"]


def icc_last20(h, f):
    res, means = [], []
    for pid, g in h.groupby("participant"):
        g = g.sort_values("order").dropna(subset=[f])
        if len(g) < 30:
            continue
        y = g[f].values[-20:]
        x = np.arange(20)
        res.append(y - np.polyval(np.polyfit(x, y, 1), x))
        means.append(y.mean())
    res = np.array(res)
    msw = np.sum(res ** 2) / (res.size - 2 * len(res))
    vb = max(np.var(means, ddof=1) - msw / 20, 0.0)
    return vb / (vb + msw)


rows = []
for hand in "LR":
    h = df[df.hand == hand]
    for f in FE:
        m1 = fit(h, f, False)[0]
        m2 = fit(h, f, True)[0]
        d = []
        for pid, g in h.groupby("participant"):
            g = g.sort_values("order").dropna(subset=[f])
            if len(g) >= 20:
                d.append(g[f].iloc[-10:].mean() - g[f].iloc[:10].mean())
        d = np.array(d)
        ci = m2.conf_int().loc["lo"]
        rows.append({"hand": hand, "feature": f, "n": int(h[f].notna().sum()), "b1_M1": m1.params["lo"], "p_M1": m1.pvalues["lo"],
                     "b1_M2": m2.params["lo"], "ci_lo": ci[0], "ci_hi": ci[1], "p_M2": m2.pvalues["lo"],
                     "b3_logT": m2.params["lT"], "p_b3": m2.pvalues["lT"], "SRM": d.mean() / d.std(ddof=1), "ICC1": icc_last20(h, f)})
r = pd.DataFrame(rows)
k = r.feature.isin(["sparc_w4", "ldlj_w4"])
r.loc[k, "p_M2_fdr"] = multipletests(r.loc[k, "p_M2"], method="fdr_bh")[1]
r.to_csv("results/a1_windowed.csv", index=False)
pd.set_option("display.width", 250)
print(r.round(3).to_string())
print(df[["sparc", "sparc_w4", "ldlj", "ldlj_w4", "lT", "w_mean"]].corr(method="spearman").round(2).to_string())
