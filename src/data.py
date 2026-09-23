"""讀取並整理 CO2、Stroop、心率（手環 + 心電貼片）資料。

原始受試者資料不在這個 repo 裡：Stroop 與心電資料依 REC 核定的保管方式不能上 GitHub，
程式直接讀兩個姊妹專案資料夾（路徑可用環境變數覆寫）。
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GITHUB = ROOT.parent
STROOP_DIR = Path(os.environ.get("STROOP_DIR", GITHUB / "cognitive_performance_test" / "output"))
ECG_DIR = Path(os.environ.get("ECG_DIR", GITHUB / "ECG" / "data"))
CO2_DIR = ROOT / "data" / "co2"
PRIVATE_DIR = ROOT / "data" / "private"  # 逐人資料，已列入 .gitignore

SENSORS = ["Wa1", "Wa2"]
# 同一次施測前後各延伸 60 秒取感測器讀值（感測器每 30 秒一筆，40 秒的測驗區間內可能只有 1 筆）
CO2_PAD = pd.Timedelta(seconds=60)


# ---------------------------------------------------------------- CO2
def load_co2(sensor: str) -> pd.DataFrame:
    d = pd.read_csv(CO2_DIR / f"{sensor}.csv", encoding="utf-8-sig")
    d.columns = ["date", "clock", "co2", "temp", "rh"]
    d["time"] = pd.to_datetime(d["date"] + " " + d["clock"])
    d = d[["time", "co2", "temp", "rh"]].sort_values("time").drop_duplicates("time")
    # SCD30 量程 400–10000 ppm；室內不可能低於戶外（約 400），低於 350 視為傳輸錯誤
    d = d[(d.co2 >= 350) & (d.co2 <= 10000)]
    return d.reset_index(drop=True)


def load_all_co2() -> dict[str, pd.DataFrame]:
    return {s: load_co2(s) for s in SENSORS}


def window_mean(co2: pd.DataFrame, t0, t1, pad=CO2_PAD) -> pd.Series:
    m = (co2.time >= t0 - pad) & (co2.time <= t1 + pad)
    sub = co2.loc[m, ["co2", "temp", "rh"]]
    out = sub.mean() if len(sub) else pd.Series({"co2": np.nan, "temp": np.nan, "rh": np.nan})
    out["n_readings"] = len(sub)
    return out


# ---------------------------------------------------------------- Stroop
def load_stroop_trials() -> pd.DataFrame:
    files = sorted(glob.glob(str(STROOP_DIR / "stroop_*.csv")))
    files = [f for f in files if "TEST" not in Path(f).name]
    d = pd.concat([pd.read_csv(f, encoding="utf-8-sig", dtype={"Student_ID": str}) for f in files])
    d = d[(d.Test_Set == "Stroop") & (d.Page_Hidden == "否")]
    d["time"] = pd.to_datetime(d.Stimulus_Onset)
    d["correct"] = (d.Is_Correct == "是").astype(int)
    d["rt"] = d.Reaction_Time_ms.astype(float)
    d["condition"] = d.Condition.map({"中性": "neutral", "一致": "congruent", "不一致": "incongruent"})
    # 以「日期 + 上午/下午」當施測區塊（09-23 下午有一位同學另開一場，合併到同一區塊）
    d["block"] = d.time.dt.strftime("%m-%d") + np.where(d.time.dt.hour < 12, " AM", " PM")
    return d.reset_index(drop=True)


def pseudonyms(ids) -> dict[str, str]:
    return {sid: f"S{i + 1:02d}" for i, sid in enumerate(sorted(set(ids)))}


def stroop_person_sessions(trials: pd.DataFrame, co2: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """每位學生每次施測一列：平均反應時間、正確率、干擾分數，與該 40 秒區間的環境值。"""
    # 反應時間只用答對、且在 200–3000 ms 之間的題目（剔除誤觸與分心）
    ok = trials[(trials.correct == 1) & trials.rt.between(200, 3000)]
    rows = []
    for (block, sid), g in trials.groupby(["block", "Student_ID"]):
        gk = ok[(ok.block == block) & (ok.Student_ID == sid)]
        by = gk.groupby("condition").rt.mean()
        t0, t1 = g.time.min(), g.time.max()
        r = {
            "block": block, "student_id": sid, "t0": t0, "t1": t1,
            "slot": "PM" if t0.hour >= 12 else "AM",
            "rt_mean": gk.rt.mean(),
            "accuracy": g.correct.mean(),
            "interference": by.get("incongruent", np.nan) - by.get("neutral", np.nan),
            "n_trials": len(g),
        }
        for s, c in co2.items():
            w = window_mean(c, t0, t1)
            r[f"co2_{s}"], r[f"temp_{s}"], r[f"rh_{s}"], r[f"nread_{s}"] = w.co2, w.temp, w.rh, w.n_readings
        rows.append(r)
    d = pd.DataFrame(rows).sort_values(["student_id", "t0"])
    d["test_no"] = d.groupby("student_id").cumcount() + 1  # 第幾次做 Stroop（練習效應）
    d["student"] = d.student_id.map(pseudonyms(d.student_id))
    return d.reset_index(drop=True)


def add_trial_env(trials: pd.DataFrame, ps: pd.DataFrame) -> pd.DataFrame:
    keep = ["block", "student_id", "student", "test_no", "slot"] + [
        c for c in ps.columns if c.split("_")[0] in ("co2", "temp", "rh")]
    t = trials.rename(columns={"Student_ID": "student_id"}).merge(ps[keep], on=["block", "student_id"])
    return t


# ---------------------------------------------------------------- 心率
def load_hr_seconds() -> pd.DataFrame:
    """所有手環與貼片的每秒心率，長格式：time, device, hr。同一秒同一顆出現在多個檔就取平均。"""
    files = [f for f in glob.glob(str(ECG_DIR / "merged*.csv"))]
    parts = []
    for f in files:
        d = pd.read_csv(f, parse_dates=["time"])
        long = d.melt(id_vars="time", var_name="device", value_name="hr").dropna()
        long["device"] = long.device.str.replace("hr_", "", regex=False)
        parts.append(long)
    hr = pd.concat(parts).groupby(["time", "device"], as_index=False).hr.mean()
    hr = hr[hr.hr.between(40, 200)]
    hr["device_type"] = np.where(hr.device.str.startswith("E"), "ecg", "polar")
    return hr


def _session_id(t: pd.Series) -> pd.Series:
    # 同一天上午／下午各一節收錄
    return t.dt.strftime("%m-%d") + np.where(t.dt.hour < 12, " AM", " PM")


def hr_windows(hr: pd.DataFrame, minutes: int = 5, min_cover: float = 0.6) -> pd.DataFrame:
    """每顆裝置每 5 分鐘一列的平均心率（計畫書：生理訊號以 5 分鐘為單位）。"""
    h = hr.copy()
    h["win"] = h.time.dt.floor(f"{minutes}min")
    w = h.groupby(["device", "device_type", "win"]).agg(hr=("hr", "mean"), n=("hr", "size")).reset_index()
    w = w[w.n >= min_cover * minutes * 60]
    w["session"] = _session_id(w.win)
    w["wearer"] = w.device + " " + w.session  # 裝置 × 節次 = 同一位配戴者
    w["elapsed_min"] = w.groupby("wearer").win.transform(lambda s: (s - s.min()).dt.total_seconds() / 60)
    return w.reset_index(drop=True)


def rmssd_windows(minutes: int = 5, min_good: float = 0.8, min_beats: int = 150) -> pd.DataFrame:
    """心電貼片每 5 分鐘的 RMSSD。只用相鄰兩拍都沒被標記為異常的差值。"""
    rows = []
    for f in glob.glob(str(ECG_DIR / "ecgrr_*.csv")):
        dev = Path(f).stem.split("_")[1]
        d = pd.read_csv(f, parse_dates=["time"])
        if d.empty:
            continue
        d["win"] = d.time.dt.floor(f"{minutes}min")
        for win, g in d.groupby("win"):
            good = g.bad.values == 0
            if len(g) < min_beats or good.mean() < min_good:
                continue
            rr = g.rr_ms.values
            pair = good[1:] & good[:-1]
            diff = np.diff(rr)[pair]
            if len(diff) < min_beats // 2:
                continue
            rows.append({"device": dev, "win": win, "rmssd": float(np.sqrt(np.mean(diff ** 2))),
                         "hr_ecg": 60000 / rr[good].mean(), "good_pct": good.mean(), "beats": len(g)})
    w = pd.DataFrame(rows).groupby(["device", "win"], as_index=False).mean()
    w["session"] = _session_id(w.win)
    w["wearer"] = w.device + " " + w.session
    w["elapsed_min"] = w.groupby("wearer").win.transform(lambda s: (s - s.min()).dt.total_seconds() / 60)
    return w


def add_window_env(w: pd.DataFrame, co2: dict[str, pd.DataFrame], minutes: int = 5) -> pd.DataFrame:
    w = w.copy()
    for s, c in co2.items():
        cc = c.copy()
        cc["win"] = cc.time.dt.floor(f"{minutes}min")
        agg = cc.groupby("win").agg(**{f"co2_{s}": ("co2", "mean"), f"temp_{s}": ("temp", "mean"),
                                        f"rh_{s}": ("rh", "mean")}).reset_index()
        w = w.merge(agg, on="win", how="left")
    return w


# ---------------------------------------------------------------- 兩台感測器比對
def paired_sensors(co2: dict[str, pd.DataFrame], tol_s: int = 20) -> pd.DataFrame:
    a = co2["Wa1"].rename(columns=lambda c: c if c == "time" else f"{c}_Wa1")
    b = co2["Wa2"].rename(columns=lambda c: c if c == "time" else f"{c}_Wa2")
    p = pd.merge_asof(a.sort_values("time"), b.sort_values("time"), on="time",
                      direction="nearest", tolerance=pd.Timedelta(seconds=tol_s))
    p = p.dropna(subset=["co2_Wa2"])
    p["session"] = _session_id(p.time)
    return p.reset_index(drop=True)
