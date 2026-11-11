"""合并三个分包的特征，建立试次索引与排除标记。输出：
trial_index.csv（每试次一行）、trial_features.csv（每手-试次一行，含索引列）。"""
import re
import sys
import numpy as np, pandas as pd
TAG = sys.argv[1] if len(sys.argv) > 1 else ""   # "" = baseline; e.g. "_n0165" for a noise-injected run
d = pd.concat([pd.read_csv(f"features{TAG}_part{i}.csv", dtype={"session": str, "ts": str}) for i in (1, 2, 3)], ignore_index=True)
t = d.drop_duplicates(["participant", "session", "trial"])[["participant", "session", "trial", "ts", "block", "trial_no", "T_log", "dropped", "comments"]].copy()
t = t.sort_values(["participant", "ts"]).reset_index(drop=True)
t["comments"] = t.comments.fillna("")
# 排除标记（规则见 01_规划/分析方案_冻结版.md 第 3 节）
t["ex_calibration"] = (t.block == 0) | ((t.participant == "P01") & (t.trial == "20230911124510_Trial1-1"))
t["ex_no_log"] = t.T_log.isna()
t["ex_short"] = t.T_log < 20
bad = r"malfunction|not working|broke|Discard|ceased|by accident|Incomplete|severe problems|not moving|grasper issue|can not open|blocked|not closing"
t["ex_comment"] = t.comments.str.contains(bad, case=False, regex=True)
t["flag_dup_id"] = t.duplicated(["participant", "session", "block", "trial_no"], keep=False) & ~t.ex_calibration
t["valid_trial"] = ~(t.ex_calibration | t.ex_no_log | t.ex_short | t.ex_comment)
v = t[t.valid_trial].copy()
v["order"] = v.groupby("participant").cumcount() + 1
v["session_idx"] = v.groupby("participant").session.transform(lambda s: pd.factorize(s)[0] + 1)
v["order_in_session"] = v.groupby(["participant", "session"]).cumcount() + 1
v["cum_practice_s"] = v.groupby("participant").T_log.cumsum() - v.T_log
v["proficient"] = (v.T_log <= 48) & (v.dropped.fillna(0) == 0)
v["cum_proficient"] = v.groupby("participant").proficient.cumsum()
v["reached_criterion"] = v.cum_proficient >= 10
t = t.merge(v[["participant", "trial", "order", "session_idx", "order_in_session", "cum_practice_s", "proficient", "cum_proficient", "reached_criterion"]], on=["participant", "trial"], how="left")
if not TAG:
    t.to_csv("trial_index.csv", index=False)
f = d.drop(columns=["ts", "block", "trial_no", "T_log", "dropped", "comments"]).merge(t, on=["participant", "session", "trial"])
f.to_csv(f"trial_features{TAG}.csv", index=False)
print("trial keys", len(t), "| valid", int(t.valid_trial.sum()))
print(t[["ex_calibration", "ex_no_log", "ex_short", "ex_comment", "flag_dup_id"]].sum().to_dict())
ok = f[f.valid_trial & (f.qc == "pass")]
print("hand-trials valid+pass:", ok.groupby("hand").size().to_dict(), "| both hands:", int((ok.groupby(["participant", "trial"]).size() == 2).sum()))
s = v.groupby("participant").agg(n=("order", "max"), sessions=("session_idx", "max"), n_prof=("proficient", "sum"),
                                 first_prof=("order", lambda o: o[v.loc[o.index, "proficient"]].min()),
                                 crit_at=("order", lambda o: o[v.loc[o.index, "reached_criterion"]].min()))
print(s.to_string())
