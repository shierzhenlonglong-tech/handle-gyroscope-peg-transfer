"""补充方案 A2：腕部 Shimmer IMU -> 与手柄相同的指标。每个传感器-试次一行。
用法：python extract_wrist.py <shimmer_raw 目录> <输出 csv> [k n]   （只处理序号 % n == k 的文件，便于并行；分块输出命名为 wrist_part{k}.csv，k = 0..11）
需要 SPD-FLS1 腕部 Shimmer 原始文件（不随仓库发布）。
陀螺为 Shimmer 标定后的 deg/s；重采样到 31.25 Hz 后与手柄用同一套特征函数和阈值。"""
import sys, re, io, glob, os, dataclasses
import numpy as np, pandas as pd
import oh2026_qc_sensitivity as q
from extract_features import hand_features, FS

CFG = dataclasses.replace(q.QCConfig(), nominal_dt=0.0195, gyro_scale=1.0)
STRIP = re.compile(r"[^,;\n]*;CAL;[^;]*;|VideoFrame\d+,")   # later sessions add a VideoFrame column
NAMES = ["com", "dev", "t", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz", "bat"]


def main(root, out, k=0, n=1):
    files = sorted(glob.glob(os.path.join(root, "**", "*_Shimmer.txt"), recursive=True))
    rows = []
    for i, f in enumerate(files):
        if i % n != k:
            continue
        m = re.search(r"(P\d+)_FLS Task A_(\d+)", f.replace("\\", "/"))
        key = re.search(r"(\d{14})[_ ]Trial(\d+)(?:-(\d+))?_Shimmer", os.path.basename(f))
        if not m or not key:
            continue
        trial = f"{key.group(1)}_Trial{key.group(2)}-{int(key.group(3)) if key.group(3) else 0}"
        base = {"participant": m.group(1), "session": f"{int(m.group(2)):02d}", "trial": trial}
        if os.path.getsize(f) < 2000:
            rows.append({**base, "dev": "none", "qc": "empty file"})
            continue
        try:
            txt = STRIP.sub("", open(f, encoding="utf8", errors="replace").read())
            d = pd.read_csv(io.StringIO(txt), header=None, on_bad_lines="skip", usecols=range(13), names=NAMES)
            for c in NAMES[2:9]:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d = d.dropna(subset=NAMES[2:9])
        except Exception as e:
            rows.append({**base, "dev": "none", "qc": f"parse error: {type(e).__name__}"})
            continue
        for dev, g in d.groupby("dev"):
            arr = np.column_stack([g.t.values / 1000.0, g.gx.values, g.gy.values, g.gz.values])
            stage, reason, info, cl = q.qc_hand(arr, None, CFG)
            r = {**base, "dev": dev, "qc": reason, "n_rows": info.get("n_rows"), "span_s": info.get("span_s"),
                 "t_start": float(arr[:, 0].min()) if len(arr) else np.nan}
            if stage == "S6":
                segs = q.segments_uniform(cl[0], cl[1], FS, CFG.gap_factor, CFG.min_segment_s)
                if segs:
                    feat, _ = hand_features(segs)
                    r.update(feat)
                    am = np.linalg.norm(g[["ax", "ay", "az"]].values, axis=1)
                    r["acc_mag_sd"] = float(am.std())
                else:
                    r["qc"] = "no continuous segment"
            rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out, len(rows))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 0, int(a[4]) if len(a) > 4 else 1)
