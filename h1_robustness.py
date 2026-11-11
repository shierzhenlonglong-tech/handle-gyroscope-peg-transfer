"""H1 稳健性分析（冻结方案第 5 节 1–6 项）+ 逐人 OLS 斜率。输出 results/h1_robustness.csv、h1_ols_slopes.csv。"""
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from h1_mixed_models import load, PRIMARY
warnings.filterwarnings("ignore")


def m2(d, feat, form_extra="lT + drop", ycol=None):
    ycol = ycol or feat
    d = d.dropna(subset=[ycol]).copy()
    d["y"] = (d[ycol] - d[ycol].mean()) / d[ycol].std()
    form = "y ~ lo + first_in_session + " + form_extra
    for re_f in ("~lo", None):
        best = None
        for meth in ("lbfgs", "bfgs", "powell"):
            try:
                m = smf.mixedlm(form, d, groups=d.participant, re_formula=re_f).fit(reml=True, method=meth, maxiter=2000)
                if np.isfinite(m.bse["lo"]) and (best is None or m.llf > best.llf + 1e-6):
                    best = m
            except Exception:
                pass
        if best is not None:
            ci = best.conf_int().loc["lo"]
            return best.params["lo"], ci[0], ci[1], best.pvalues["lo"], len(d)
    return (np.nan,) * 5


df = load()
both = df.groupby(["participant", "trial"]).hand.transform("nunique") == 2
df["lT_mean"] = df.groupby("participant").lT.transform("mean")
df["lT_within"] = df.lT - df.lT_mean
variants = {
    "main": (df, "lT + drop", None),
    "spline_time": (df, "bs(lT, df=4) + drop", None),
    "within_between_time": (df, "lT_within + lT_mean + drop", None),
    "both_hands_only": (df[both], "lT + drop", None),
    "no_duplicate_ids": (df[~df.flag_dup_id.astype(bool)], "lT + drop", None),
    "no_drops": (df[df["drop"] == 0], "lT", None),
}
import h1_mixed_models as _h
_h.MIN_W_MEAN = 0.0
_raw = load()                      # without deviation D1
_h.MIN_W_MEAN = 5.0
variants["without_D1_exclusion"] = (_raw.assign(lT_mean=0, lT_within=0), "lT + drop", None)
for tag, lab in (("_n0165", "noise_0.165"), ("_n05", "noise_0.5")):
    variants[lab] = (load(f"trial_features{tag}.csv"), "lT + drop", None)
rows = []
for name, (d, extra, _) in variants.items():
    for hand in "LR":
        for feat in PRIMARY:
            b, lo, hi, p, n = m2(d[d.hand == hand], feat, extra)
            rows.append({"variant": name, "hand": hand, "feature": feat, "beta_std": b, "ci_lo": lo, "ci_hi": hi, "p": p, "n": n})
for hand in "LR":   # alternative definitions
    b, lo, hi, p, n = m2(df[df.hand == hand], "sparc", ycol="sparc_fc5")
    rows.append({"variant": "sparc_fc5", "hand": hand, "feature": "sparc", "beta_std": b, "ci_lo": lo, "ci_hi": hi, "p": p, "n": n})
    for th in ("qlow_2", "qlow_10"):
        b, lo, hi, p, n = m2(df[df.hand == hand], "qlow_5", ycol=th)
        rows.append({"variant": th, "hand": hand, "feature": "qlow_5", "beta_std": b, "ci_lo": lo, "ci_hi": hi, "p": p, "n": n})
# leave one participant out: range of beta
for hand in "LR":
    for feat in PRIMARY:
        bs = []
        for pid in sorted(df.participant.unique()):
            b = m2(df[(df.hand == hand) & (df.participant != pid)], feat)[0]
            bs.append((b, pid))
        bmin, bmax = min(bs), max(bs)
        rows.append({"variant": "leave_one_out_min", "hand": hand, "feature": feat, "beta_std": bmin[0], "note": bmin[1]})
        rows.append({"variant": "leave_one_out_max", "hand": hand, "feature": feat, "beta_std": bmax[0], "note": bmax[1]})
r = pd.DataFrame(rows)
r.to_csv("results/h1_robustness.csv", index=False)
pd.set_option("display.width", 250)
print(r.pivot_table(index=["feature", "hand"], columns="variant", values="beta_std").round(3).to_string())
print(r[(r.p > 0.05)][["variant", "hand", "feature", "beta_std", "p"]].round(3).to_string())
# per-participant OLS slopes (unshrunk), with and without time
rows = []
for hand in "LR":
    for feat in PRIMARY:
        h = df[df.hand == hand]
        sd = h[feat].std()
        for pid, g in h.groupby("participant"):
            if len(g) < 20:
                continue
            g = g.assign(y=(g[feat] - h[feat].mean()) / sd)
            b1 = smf.ols("y ~ lo", g).fit().params["lo"]
            b2 = smf.ols("y ~ lo + lT", g).fit().params["lo"]
            rows.append({"hand": hand, "feature": feat, "participant": pid, "n": len(g), "slope_M1": b1, "slope_M2": b2})
s = pd.DataFrame(rows)
s.to_csv("results/h1_ols_slopes.csv", index=False)
print(s.groupby(["feature", "hand"]).agg(n=("participant", "size"), pos_M1=("slope_M1", lambda x: (x > 0).sum()), pos_M2=("slope_M2", lambda x: (x > 0).sum()), med_M2=("slope_M2", "median")).round(3).to_string())
