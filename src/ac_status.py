"""冷氣狀態判定：研究代碼對照表 09-24 起才有冷氣註記，09-21 ～ 09-23 依 CO₂ 與溫濕度的變化推定。

依據：開冷氣時教室關窗，CO₂ 會持續累積；開窗時 CO₂ 幾乎不上升。
指標（兩臺感測器各算後平均；該節讀值 < 30 筆的感測器不計入）：
  前段斜率：收錄時間前 1/3 的 CO₂ 上升速度（ppm／分）
  後段斜率：收錄時間後 1/3 的 CO₂ 上升速度
  最高濃度：5 筆滑動中位數的最大值
判定規則（先用 7 個已有註記的節次驗證，全部判對才採用）：
  「在累積」＝斜率 ≥ 8 ppm／分。依據：開窗節次的斜率最高 1.5，有冷氣註記的節次最低 12.9，8 落在兩者之間。
  前段沒在累積、後段在累積 → 後半冷氣
  前段在累積，或最高濃度 ≥ 2000 → 冷氣
  前後段都沒在累積，且最高濃度 < 1000 → 無冷氣
  其餘 → 無法判定
輸出：data/ac_status.csv（環境資料，不含個資）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data as D

OUT = D.ROOT / "data" / "ac_status.csv"
RISING = 8.0   # ppm／分


def _slope(g: pd.DataFrame) -> float:
    t = (g.time - g.time.iloc[0]).dt.total_seconds() / 60
    return float(np.polyfit(t, g.co2, 1)[0]) if len(g) >= 5 else np.nan


def features() -> pd.DataFrame:
    rows = []
    for s, c in D.load_all_co2().items():
        c = c[c.time >= "2026-09-21"].copy()
        c["session"] = D._session_id(c.time)
        for ses, g in c.groupby("session"):
            if len(g) < 30:
                continue
            g = g.sort_values("time")
            g = g[g.time >= g.time.iloc[0] + pd.Timedelta(minutes=2)]       # 去掉開機暖機
            t0, t1 = g.time.iloc[0], g.time.iloc[-1]
            third = (t1 - t0) / 3
            rows.append({"session": ses, "sensor": s, "start": t0, "end": t1,
                         "early_slope": _slope(g[g.time <= t0 + third]),
                         "late_slope": _slope(g[g.time >= t1 - third]),
                         "co2_max": g.co2.rolling(5, center=True, min_periods=1).median().max(),
                         "temp_mean": g.temp.mean(), "rh_mean": g.rh.mean()})
    f = pd.DataFrame(rows)
    agg = f.groupby("session").agg(start=("start", "min"), end=("end", "max"), early_slope=("early_slope", "mean"),
                                   late_slope=("late_slope", "mean"), co2_max=("co2_max", "max"),
                                   temp_mean=("temp_mean", "mean"), rh_mean=("rh_mean", "mean"),
                                   sensors=("sensor", lambda x: "、".join(sorted(x))))
    return agg.reset_index()


def classify(r) -> str:
    if r.early_slope < RISING <= r.late_slope:
        return "後半冷氣"
    if r.early_slope >= RISING or r.co2_max >= 2000:
        return "冷氣"
    if r.early_slope < RISING and r.late_slope < RISING and r.co2_max < 1000:
        return "無冷氣"
    return "無法判定"


def build() -> pd.DataFrame:
    f = features()
    _, bi = D.load_code_table()
    rec = {b: a for b, a in zip(bi.block, bi.ac) if isinstance(a, str) and a}
    f["rule"] = f.apply(classify, axis=1)
    f["recorded"] = f.session.map(rec).fillna("")
    check = f[f.recorded != ""]
    wrong = check[check.rule != check.recorded]
    if len(wrong):
        raise RuntimeError(f"判定規則與對照表不符，不採用：\n{wrong[['session', 'recorded', 'rule']]}")
    f["status"] = np.where(f.recorded != "", f.recorded, f.rule)
    f["source"] = np.where(f.recorded != "", "研究代碼對照表", "由 CO₂／溫濕度推定")
    f["label"] = np.where(f.recorded != "", f.status, f.status + "（推定）")
    # 09-17 沒有 CO₂，無法判定；一併列出讓表格完整
    if "09-17 AM" not in set(f.session):
        f = pd.concat([pd.DataFrame([{"session": "09-17 AM", "status": "無法判定", "source": "沒有 CO₂ 資料",
                                      "label": "無法判定", "recorded": "", "rule": ""}]), f], ignore_index=True)
    f = f.sort_values("session").reset_index(drop=True)
    out = f[["session", "status", "source", "label", "early_slope", "late_slope", "co2_max", "temp_mean", "rh_mean",
             "sensors"]].round(1)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"規則驗證：{len(check)} 個已有註記的節次全部判對")
    print(out.to_string())
    return out


def load() -> dict[str, str]:
    """節次 → 顯示用標籤（冷氣／後半冷氣／無冷氣，推定者加「（推定）」）。"""
    if not OUT.exists():
        build()
    d = pd.read_csv(OUT, encoding="utf-8-sig")
    return dict(zip(d.session, d.label))


if __name__ == "__main__":
    build()
