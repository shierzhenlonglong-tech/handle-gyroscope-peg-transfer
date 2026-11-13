"""补充方案 A1：窗口化 SPARC / LDLJ。每个手-试次一行，输出 features_w_partN.csv（仓库根目录；合并结果见 a1_windowed_smoothness.py）。
用法：python extract_windowed.py <SPD-FLS1 子集目录> <输出 csv> --only partN"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
import oh2026_qc_sensitivity as q
from extract_features import sparc, ldlj, FS, CFG

WINS = {"w2": 64, "w4": 128, "w8": 256}


def windowed(sps, n):
    sp_v, ld_v = [], []
    for s in sps:
        for i in range(0, s.size - n + 1, n):
            w = s[i:i + n]
            a, b = sparc(w, FS), ldlj(w, FS)
            if np.isfinite(a):
                sp_v.append(a)
            if np.isfinite(b):
                ld_v.append(b)
    if len(sp_v) < 3:
        return np.nan, np.nan, len(sp_v)
    return float(np.mean(sp_v)), float(np.mean(ld_v)) if ld_v else np.nan, len(sp_v)


def main(root, out, only):
    trials, _ = q.discover(Path(root))
    rows = []
    for tr in trials:
        if only and only not in str(tr.members.get("log", tr.members.get("imu", ("", "", 0)))[0]):
            continue
        mem = str(tr.members.get("log", tr.members.get("imu", ("", "", 0)))[1]).replace("\\", "/")
        sm = re.search(r"(P\d+)_FLS Task A_(\d+)", mem)
        pid, ses = (sm.group(1).upper(), f"{int(sm.group(2)):02d}") if sm else (tr.participant, tr.session)
        imu = tr.members.get("imu")
        if imu is None or imu[2] == 0:
            continue
        T_log = None
        if "log" in tr.members and tr.members["log"][2] > 0:
            T_log = q.parse_log(q._read(tr.members["log"]).decode("utf-8", errors="replace"))["T_log"]
        data, _ = q.parse_imu(q._read(imu).decode("utf-8", errors="replace"), CFG.gyro_scale)
        for h, hh in (("LH", "L"), ("RH", "R")):
            stage, _, _, cl = q.qc_hand(data[h], T_log, CFG)
            if stage != "S6":
                continue
            segs = q.segments_uniform(cl[0], cl[1], FS, CFG.gap_factor, CFG.min_segment_s)
            if not segs:
                continue
            sps = [np.linalg.norm(y, axis=1) for y in segs]
            r = {"participant": pid, "session": ses, "trial": tr.key, "hand": hh}
            for tag, n in WINS.items():
                a, b, k = windowed(sps, n)
                r[f"sparc_{tag}"], r[f"ldlj_{tag}"], r[f"nwin_{tag}"] = a, b, k
            rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out, len(rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None)
