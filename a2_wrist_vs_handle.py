"""补充方案 A2：腕部 IMU 与手柄 IMU 的比较。输出 results/a2_*.csv。
若仓库根目录有 wrist_part{0..11}.csv（extract_wrist.py 的输出），则由其合并、筛选并写出 wrist_features.csv；
否则直接读取随仓库发布的 wrist_features.csv（已通过 QC 与覆盖率 0.9–1.2 筛选）。"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from h1_mixed_models import load, fit, PRIMARY, _admissible
import os as _os
_os.makedirs("results", exist_ok=True)
_os.makedirs("figures", exist_ok=True)
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
hd = load()
idx = pd.read_csv("trial_index.csv")
idx = idx[idx.valid_trial.astype(bool)]
FROM_PARTS = all(_os.path.exists(f"wrist_part{k}.csv") for k in range(12))
if FROM_PARTS:
    w = pd.concat([pd.read_csv(f"wrist_part{k}.csv", dtype={"session": str}) for k in range(12)])
    w = w.merge(idx[["participant", "trial", "order", "order_in_session", "session_idx", "T_log", "dropped"]], on=["participant", "trial"])
    n_all = len(w[w.dev != "none"])
else:
    w = pd.read_csv("wrist_features.csv", dtype={"session": str})   # 已发布的合并表；无文件的记录不在其中，n_all 无法复现
    n_all = "n/a (from wrist_features.csv)"
w = w[(w.qc == "pass")].copy()
cov = w.span_s / w.T_log
w = w[(cov >= 0.9) & (cov <= 1.2)].copy()
print("wrist device-trials: with file", n_all, "| pass QC + coverage 0.9-1.2:", len(w), w.groupby("dev").size().to_dict())

# ---- map each wrist sensor to a hand, per participant-session, from trial-level mean speed ----
hw = hd.pivot_table(index=["participant", "trial"], columns="hand", values="w_mean")
ww = w.pivot_table(index=["participant", "trial"], columns="dev", values="w_mean")
devs = [c for c in ww.columns]
j = hw.join(ww, how="inner").join(idx.set_index(["participant", "trial"]).session_idx)
maps, rows = {}, []
for pid, g in j.groupby("participant"):
    def score(gg):
        c = gg[["L", "R"] + devs].corr(method="spearman")
        a = c.loc["L", devs[0]] + c.loc["R", devs[1]]          # devs[0] = left
        b = c.loc["L", devs[1]] + c.loc["R", devs[0]]          # devs[1] = left
        return a, b
    a, b = score(g.dropna())
    maps[pid] = {devs[0]: "L", devs[1]: "R"} if a >= b else {devs[1]: "L", devs[0]: "R"}
    agree = []
    for s, gg in g.dropna().groupby("session_idx"):
        if len(gg) >= 8:
            sa, sb = score(gg)
            agree.append((sa >= sb) == (a >= b))
    rows.append({"participant": pid, "n_both": len(g.dropna()), "score_chosen": max(a, b) / 2, "score_other": min(a, b) / 2,
                 "left_device": [k for k, v in maps[pid].items() if v == "L"][0], "sessions_checked": len(agree), "sessions_agree": int(np.sum(agree))})
mp = pd.DataFrame(rows)
mp.to_csv("results/a2_device_mapping.csv", index=False)
print(mp.round(2).to_string())
w["hand"] = [maps[p][d] for p, d in zip(w.participant, w.dev)]
w["lo"], w["lT"] = np.log(w.order), np.log(w.T_log)
w["first_in_session"] = (w.order_in_session == 1).astype(int)
w["drop"] = (w.dropped.fillna(0) > 0).astype(int)
if FROM_PARTS:
    w.to_csv("wrist_features.csv", index=False)
# 2026-11-11 补充方案 B2 之后：腕部传感器平均角速度 < 5 deg/s 的记录按与 D1 相同的规则剔除（P16 第 1 会话双腕各 10 个、P21 右腕 1 个）。
# wrist_features.csv 保留全部记录供 B2 使用；设环境变量 A2_KEEP_ALL=1 可复现剔除前的结果（a2_*_all.csv）。
import os
KEEP_ALL = os.environ.get("A2_KEEP_ALL") == "1"
TAG = "_all" if KEEP_ALL else ""
if not KEEP_ALL:
    print("wrist records removed by the 5 deg/s rule:", int((w.w_mean < 5.0).sum()))
    w = w[w.w_mean >= 5.0].copy()

# ---- (1) M1 / M2 for wrist metrics ----
rows = []
for hand in "LR":
    h = w[w.hand == hand]
    for f in PRIMARY:
        m1, m2 = fit(h, f, False)[0], fit(h, f, True)[0]
        ci = m2.conf_int().loc["lo"]
        d = []
        for pid, g in h.groupby("participant"):
            g = g.sort_values("order")
            if len(g) >= 20:
                d.append(g[f].iloc[-10:].mean() - g[f].iloc[:10].mean())
        d = np.array(d)
        rows.append({"sensor": "wrist", "hand": hand, "feature": f, "n": len(h), "b1_M1": m1.params["lo"], "p_M1": m1.pvalues["lo"],
                     "b1_M2": m2.params["lo"], "ci_lo": ci[0], "ci_hi": ci[1], "p_M2": m2.pvalues["lo"], "b3_logT": m2.params["lT"], "SRM": d.mean() / d.std(ddof=1)})
wr = pd.DataFrame(rows)
wr.to_csv(f"results/a2_wrist_models{TAG}.csv", index=False)
print(wr.round(3).to_string())

# ---- (2) within-person handle-wrist correlation; (3) mutual adjustment ----
both = hd.merge(w[["participant", "trial", "hand"] + PRIMARY], on=["participant", "trial", "hand"], suffixes=("", "_wr"))
print("handle-wrist paired hand-trials:", both.groupby("hand").size().to_dict())


def best(form, d):
    b = None
    for meth in ("lbfgs", "bfgs", "powell"):
        try:
            m = smf.mixedlm(form, d, groups=d.participant, re_formula="~lo").fit(reml=True, method=meth, maxiter=2000)
        except Exception:
            continue
        if _admissible(m) and (b is None or m.llf > b.llf + 1e-6):
            b = m
    if b is None:
        for meth in ("lbfgs", "powell"):
            m = smf.mixedlm(form, d, groups=d.participant).fit(reml=True, method=meth)
            if _admissible(m) and (b is None or m.llf > b.llf + 1e-6):
                b = m
    return b


rows = []
for hand in "LR":
    h = both[both.hand == hand].copy()
    for f in PRIMARY:
        a = h[f] - h.groupby("participant")[f].transform("mean")
        b_ = h[f + "_wr"] - h.groupby("participant")[f + "_wr"].transform("mean")
        r_within = np.corrcoef(a, b_)[0, 1]
        h["yh"] = (h[f] - h[f].mean()) / h[f].std()
        h["yw"] = (h[f + "_wr"] - h[f + "_wr"].mean()) / h[f + "_wr"].std()
        mh = best("yh ~ lo + first_in_session + lT + drop + yw", h)     # handle given wrist
        mw = best("yw ~ lo + first_in_session + lT + drop + yh", h)     # wrist given handle
        rows.append({"hand": hand, "feature": f, "n": len(h), "r_within": r_within, "ratio_handle_over_wrist": float(np.median(h[f] / h[f + "_wr"])) if f in ("w_mean", "w_p95") else np.nan,
                     "handle_given_wrist_b1": mh.params["lo"], "p_h": mh.pvalues["lo"], "wrist_given_handle_b1": mw.params["lo"], "p_w": mw.pvalues["lo"]})
cmp = pd.DataFrame(rows)
cmp.to_csv(f"results/a2_handle_vs_wrist{TAG}.csv", index=False)
print(cmp.round(3).to_string())
print("median trial-mean speed: wrist", round(w.w_mean.median(), 2), "handle", round(hd.w_mean.median(), 2), "| handle under deg reading", round(hd.w_mean.median() * np.pi / 180, 2))
