"""补充方案 B1：同步腕部传感器的双手协调指标 —— M1/M2、BH-FDR、SRM、ICC、未收缩逐人斜率。
输出 results/b1_bimanual.csv（B1_KEEP_ALL=1 时为 results/b1_bimanual_all.csv）。
若仓库根目录有 wristbi_part{0..11}.csv（extract_wrist_bimanual.py 的输出），则由其合并、筛选并写出 wrist_bimanual.csv；
否则直接读取随仓库发布的 wrist_bimanual.csv（已按覆盖率 0.9–1.2 筛选）。"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from h1_mixed_models import fit
import os as _os
_os.makedirs("results", exist_ok=True)
_os.makedirs("figures", exist_ok=True)
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
PRIM, SEC = ["bi_both", "bi_single", "bi_corr"], ["bi_idle", "bi_lag", "bi_xcmax", "bi_left_only", "bi_right_only", "bi_excess"]
idx = pd.read_csv("trial_index.csv")
idx = idx[idx.valid_trial.astype(bool)]
FROM_PARTS = all(_os.path.exists(f"wristbi_part{k}.csv") for k in range(12))
if FROM_PARTS:
    b = pd.concat([pd.read_csv(f"wristbi_part{k}.csv") for k in range(12)])
    n_raw = len(b)
    b = b.merge(idx[["participant", "trial", "order", "order_in_session", "session_idx", "T_log", "dropped"]], on=["participant", "trial"])
    n_valid = len(b)
else:
    b = pd.read_csv("wrist_bimanual.csv")   # 已发布的合并表；n_raw / n_valid 无法由它复现
    n_raw = n_valid = "n/a (from wrist_bimanual.csv)"
cov = b.overlap_s / b.T_log
b = b[(cov >= 0.9) & (cov <= 1.2)].copy()
b["lo"], b["lT"] = np.log(b.order), np.log(b.T_log)
b["first_in_session"] = (b.order_in_session == 1).astype(int)
b["drop"] = (b.dropped.fillna(0) > 0).astype(int)
if FROM_PARTS:
    b.to_csv("wrist_bimanual.csv", index=False)
# 2026-11-11 B2 之后：任一腕部传感器平均角速度 < 5 deg/s 的试次剔除（与 D1 同一规则）；B1_KEEP_ALL=1 复现剔除前结果（b1_bimanual_all.csv）。
import os
KEEP_ALL = os.environ.get("B1_KEEP_ALL") == "1"
if not KEEP_ALL:
    wf = pd.read_csv("wrist_features.csv")
    bad = wf[wf.w_mean < 5.0][["participant", "trial"]].drop_duplicates().assign(_x=1)
    b = b.merge(bad, on=["participant", "trial"], how="left")
    print("trials removed by the wrist 5 deg/s rule:", int(b._x.notna().sum()))
    b = b[b._x.isna()].drop(columns="_x")
pl_, pr_ = b.bi_both + b.bi_left_only, b.bi_both + b.bi_right_only
b["bi_excess"] = b.bi_both - pl_ * pr_          # exploratory, not pre-specified: simultaneous movement beyond independence
print("trials with both wrists passing QC:", n_raw, "| valid trials:", n_valid, "| overlap 0.9-1.2:", len(b), "| participants:", b.participant.nunique())
print("start offset between wrists (s): median |.|", round(b.start_offset_s.abs().median(), 3), "p99", round(b.start_offset_s.abs().quantile(.99), 3))
print(b[PRIM + SEC].describe().round(3).to_string())

rows = []
for f in PRIM + SEC:
    m1, m2 = fit(b, f, False)[0], fit(b, f, True)[0]
    ci = m2.conf_int().loc["lo"]
    sd, mu = b[f].std(), b[f].mean()
    s1, s2, d, res, means = [], [], [], [], []
    for pid, g in b.groupby("participant"):
        g = g.sort_values("order")
        if len(g) < 20:
            continue
        g = g.assign(y=(g[f] - mu) / sd)
        s1.append(smf.ols("y ~ lo", g).fit().params["lo"])
        s2.append(smf.ols("y ~ lo + lT", g).fit().params["lo"])
        d.append(g[f].iloc[-10:].mean() - g[f].iloc[:10].mean())
        if len(g) >= 30:
            y = g[f].values[-20:]
            x = np.arange(20)
            res.append(y - np.polyval(np.polyfit(x, y, 1), x))
            means.append(y.mean())
    s1, s2, d, res = np.array(s1), np.array(s2), np.array(d), np.array(res)
    msw = np.sum(res ** 2) / (res.size - 2 * len(res))
    vb = max(np.var(means, ddof=1) - msw / 20, 0.0)
    rows.append({"feature": f, "primary": f in PRIM, "n": len(b), "mean": mu, "sd": sd,
                 "b1_M1": m1.params["lo"], "p_M1": m1.pvalues["lo"], "b1_M2": m2.params["lo"], "ci_lo": ci[0], "ci_hi": ci[1], "p_M2": m2.pvalues["lo"],
                 "b3_logT": m2.params["lT"], "p_logT": m2.pvalues["lT"], "n_subj": len(s1),
                 "sign_M1": int(np.sum(np.sign(s1) == np.sign(m1.params["lo"]))), "sign_M2": int(np.sum(np.sign(s2) == np.sign(m2.params["lo"]))),
                 "mean_change": d.mean(), "SRM": d.mean() / d.std(ddof=1), "ICC1": vb / (vb + msw), "n_icc": len(res)})
r = pd.DataFrame(rows)
for c in ("p_M1", "p_M2"):
    r[c + "_fdr"] = np.nan
    r.loc[r.primary, c + "_fdr"] = multipletests(r.loc[r.primary, c], method="fdr_bh")[1]
r.to_csv("results/b1_bimanual_all.csv" if KEEP_ALL else "results/b1_bimanual.csv", index=False)
print(r.round(3).T.to_string())
