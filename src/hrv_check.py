"""能不能把「每秒心率」換算成 HRV？用品質 good 的心電貼片直接比對。

同一個 5 分鐘窗，算兩種 RMSSD：
  真 RMSSD：逐拍 RR 間隔（ecgrr，只用相鄰兩拍都正常的差值）
  換算 RMSSD：每秒心率 → 60000 / HR 當作「間隔」，再算相鄰差值
若每秒心率能代表 HRV，兩者應該高度相關、數值接近。

執行：python src/hrv_check.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import data as D

RES = D.ROOT / "results"


def main():
    good = D.usable_ecg_recordings()
    rr = D.rmssd_windows()                       # 已只含有訊號的貼片節次，時段乾淨拍 ≥ 80%
    rows = []
    for f in D._ecg_files("ecghr"):
        dev = f.name.split("_")[1]
        h = pd.read_csv(f, parse_dates=["time"])
        if h.empty:
            continue
        h["win"] = h.time.dt.floor("5min")
        for win, g in h.groupby("win"):
            wearer = f"{dev} {D._session_id(pd.Series([win])).iloc[0]}"
            if wearer not in good or len(g) < 180:   # 逐時段品質由與 rmssd_windows（≥ 80% + Malik）合併時決定
                continue
            g = g.sort_values("time")
            consec = g.time.diff().dt.total_seconds().values[1:] == 1   # 只用相鄰兩秒都有資料的差值
            pseudo = 60000 / g.hr_ecg.values
            d = np.diff(pseudo)[consec]
            if len(d) < 100:
                continue
            rows.append({"device": dev, "win": win, "rmssd_from_hr": float(np.sqrt(np.mean(d ** 2)))})
    conv = pd.DataFrame(rows).groupby(["device", "win"], as_index=False).mean()
    m = rr.merge(conv, on=["device", "win"]).replace([np.inf, -np.inf], np.nan).dropna(subset=["rmssd", "rmssd_from_hr"])

    # 手環：只有平滑過的每秒心率（0 個封包含 RR），用同樣方法換算
    hr = D.load_hr_seconds()
    pol = hr[hr.device_type == "polar"].sort_values(["device", "time"]).copy()
    pol["win"] = pol.time.dt.floor("5min")
    prow = []
    for (dev, win), g in pol.groupby(["device", "win"]):
        if len(g) < 180:
            continue
        consec = g.time.diff().dt.total_seconds().values[1:] == 1
        d = np.diff(60000 / g.hr.values)[consec]
        if len(d) >= 100:
            prow.append(float(np.sqrt(np.mean(d ** 2))))
    polar_rmssd = np.array(prow)
    rho = stats.spearmanr(m.rmssd, m.rmssd_from_hr).statistic
    r = np.corrcoef(np.log(m.rmssd), np.log(m.rmssd_from_hr))[0, 1]
    ratio = (m.rmssd_from_hr / m.rmssd)
    out = {"windows": len(m), "wearers": m.wearer.nunique(),
           "median_true_rmssd_ms": m.rmssd.median(), "median_hr_derived_rmssd_ms": m.rmssd_from_hr.median(),
           "median_ratio_derived_over_true": ratio.median(), "ratio_q1": ratio.quantile(0.25),
           "ratio_q3": ratio.quantile(0.75), "spearman_rho": rho, "pearson_r_log": r,
           "polar_windows": len(polar_rmssd), "median_polar_hr_derived_rmssd_ms": float(np.median(polar_rmssd)),
           "polar_q1": float(np.quantile(polar_rmssd, 0.25)), "polar_q3": float(np.quantile(polar_rmssd, 0.75))}
    pd.DataFrame([out]).round(3).to_csv(RES / "hrv_conversion_check.csv", index=False, encoding="utf-8-sig")
    import figures as F
    F.hrv_conversion(m, out, polar_rmssd)
    print(out)


if __name__ == "__main__":
    main()
