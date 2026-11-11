"""H1 主分析（冻结方案第 5 节）。输出到 07_results/。"""
import warnings, json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
MIN_W_MEAN = 5.0
PRIMARY = ["w_mean", "w_p95", "qlow_5", "pause_frac", "sparc", "peaks_per_s"]
SECONDARY = ["w_cv", "pause_rate", "qlow_2", "qlow_10", "ldlj", "acc_mag_sd", "bi_corr", "bi_both_move", "bi_left_share"]


def load(path="../02_数据/trial_features.csv"):
    f = pd.read_csv(path, dtype={"session": str})
    f = f[f.valid_trial.astype(bool) & (f.qc == "pass")].copy()
    # 偏离记录 D1（2026-11-10）：平均角速度 < 5 deg/s 的手-试次判为传感器未随手柄运动（实际只命中 P26 右手第 2 会话的 30 个）
    f = f[f.w_mean >= MIN_W_MEAN].copy()
    f["lo"] = np.log(f.order)
    f["lT"] = np.log(f.T_log)
    f["first_in_session"] = (f.order_in_session == 1).astype(int)
    f["drop"] = (f.dropped.fillna(0) > 0).astype(int)
    return f


def fit(df, feat, with_time, std=True):
    d = df.dropna(subset=[feat]).copy()
    sd = d[feat].std()
    d["y"] = (d[feat] - d[feat].mean()) / sd if std else d[feat]
    form = "y ~ lo + first_in_session" + (" + lT + drop" if with_time else "")
    last = None
    for re_f in ("~lo", None):          # random slope first; random intercept only as fallback
        best = None
        for meth in ("lbfgs", "bfgs", "powell", "cg", "nm"):   # keep the optimiser with the highest REML likelihood
            try:
                m = smf.mixedlm(form, d, groups=d.participant, re_formula=re_f).fit(reml=True, method=meth, maxiter=2000)
                if np.isfinite(m.bse["lo"]) and (best is None or m.llf > best.llf + 1e-6):
                    best = m
            except Exception as e:
                last = e
        if best is not None:
            m = best
            slopes = {k: m.params["lo"] + v.get("lo", 0.0) for k, v in m.random_effects.items()} if re_f else None
            return m, re_f is not None, slopes, sd, len(d)
    raise RuntimeError(f"{feat}: {last}")


def run(df, feats, tag):
    rows, slopes_all = [], {}
    for hand in ("L", "R"):
        h = df[df.hand == hand]
        for feat in feats:
            for with_time in (False, True):
                m, rs, slopes, sd, n = fit(h, feat, with_time)
                ci = m.conf_int().loc["lo"]
                r = {"hand": hand, "feature": feat, "model": "M2" if with_time else "M1", "n": n, "random_slope": rs,
                     "beta_std": m.params["lo"], "ci_lo": ci[0], "ci_hi": ci[1], "p": m.pvalues["lo"], "beta_raw": m.params["lo"] * sd}
                if with_time:
                    r["beta_lT_std"] = m.params["lT"]
                    r["p_lT"] = m.pvalues["lT"]
                if slopes:
                    s = np.array(list(slopes.values()))
                    r["n_same_sign"] = int(np.sum(np.sign(s) == np.sign(m.params["lo"])))
                    r["n_subj"] = len(s)
                    slopes_all[f"{hand}|{feat}|{r['model']}"] = slopes
                rows.append(r)
    res = pd.DataFrame(rows)
    res["p_fdr"] = np.nan
    for mod in ("M1", "M2"):
        k = res.model == mod
        res.loc[k, "p_fdr"] = multipletests(res.loc[k, "p"], method="fdr_bh")[1]
    res.to_csv(f"../07_results/h1_{tag}.csv", index=False)
    json.dump(slopes_all, open(f"../07_results/h1_{tag}_slopes.json", "w"))
    return res


def srm(df):
    rows = []
    for hand in ("L", "R"):
        h = df[df.hand == hand]
        for feat in ["T_log"] + PRIMARY:
            diffs = []
            for p, g in h.groupby("participant"):
                g = g.sort_values("order").dropna(subset=[feat])
                if len(g) >= 20:
                    diffs.append(g[feat].iloc[-10:].mean() - g[feat].iloc[:10].mean())
            diffs = np.array(diffs)
            rows.append({"hand": hand, "feature": feat, "n_subj": len(diffs), "mean_change": diffs.mean(),
                         "sd_change": diffs.std(ddof=1), "SRM": diffs.mean() / diffs.std(ddof=1)})
    out = pd.DataFrame(rows)
    out.to_csv("../07_results/h1_srm.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    df = load()
    print("n hand-trials", df.groupby("hand").size().to_dict(), "participants", df.participant.nunique())
    res = run(df, PRIMARY, "primary")
    print(res.round(4).to_string())
    print(srm(df).round(3).to_string())
    sec = run(df, SECONDARY, "secondary")
    print(sec[["hand", "feature", "model", "n", "beta_std", "p", "p_fdr", "n_same_sign"]].round(4).to_string())
