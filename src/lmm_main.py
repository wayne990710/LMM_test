"""主要分析：環境（CO₂、溫度、濕度）對學生疲勞、認知表現與心率的 LMM。

變因設定（依研究設計）
  自變量：CO₂（每 100 ppm）、溫度（°C）、濕度（%）——三者同時放入模型
  依變量：疲勞程度、反應時間、正確率、干擾分數、瞬時心率、心率變異度（各一個模型）
  控制變因：現場人數、冷氣（開／關）、睡眠時長、睡眠品質
            睡眠來自疲勞量表，只放在疲勞模型（做法 C）；其他模型若也控制睡眠，沒填量表的人會被排除，
            這個做法（A）當作敏感度分析。
  隨機效應：學生隨機截距；心率／HRV 為每 5 分鐘一筆，另加「學生 × 節次」隨機截距
            （同一節課內的多個時段彼此相關）

資料使用原則：有資料的都納入（各依變量的人數、筆數不同，見 lmm_sample_size.csv）。
  - 疲勞量表：排除注意力檢核未通過的問卷。
  - 心率／HRV：只有能從研究代碼對照表對應到學生的節次（貼片編號、手環編號）。
  - HRV：5 分鐘時段內乾淨心跳 ≥ 95%（異常 ≤ 5%，依 Kubios 建議），並以 Malik 20% 準則剔除錯拍。
  - 每節課固定一個 CO₂ 來源：兩臺都完整就用平均，否則用有完整資料的那一臺。

執行：python src/lmm_main.py（需先跑過 run_analysis.py 產生 data/private/ 的資料表）
輸出：results/LMM/（模型層級結果，可公開）
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import ac_status as AC
import data as D

warnings.filterwarnings("ignore")
OUT = D.ROOT / "results" / "LMM"

IVS = [("co2h", "CO₂（每 100 ppm）"), ("temp", "溫度（°C）"), ("rh", "濕度（%）")]
CONTROLS = [("headcount", "現場人數"), ("ac", "冷氣（開 = 1）"), ("sleep_h", "睡眠時長（小時）"),
            ("sleep_q", "睡眠品質")]
HRV_CLEAN = 0.95
SLEEP = ("sleep_h", "sleep_q")


def controls_for(y: str, sleep_all: bool = False) -> list[str]:
    """主要分析（C）：睡眠只控制在疲勞模型；sleep_all=True 為敏感度分析（A）：每個模型都控制睡眠。"""
    return [c for c, _ in CONTROLS if c not in SLEEP or sleep_all or y == "fatigue_now"]


DVS = [  # 代號, 名稱, 單位, 資料層級
    ("fatigue_now", "疲勞程度", "分（1–7）", "人次"),
    ("rt_mean", "反應時間", "ms", "人次"),
    ("accuracy", "正確率", "比例", "人次"),
    ("interference", "干擾分數", "ms", "人次"),
    ("hr", "瞬時心率", "bpm", "5 分鐘"),
    ("log_rmssd", "心率變異度 log(RMSSD)", "log ms", "5 分鐘"),
]


# ---------------------------------------------------------------- 冷氣
def ac_switch_time(co2: dict, session: str) -> pd.Timestamp:
    """「後半冷氣」的節次：用兩段折線擬合 CO₂，轉折點視為開冷氣（開始累積）的時間。"""
    c = pd.concat([v[D._session_id(v.time) == session] for v in co2.values()]).sort_values("time")
    c = c.set_index("time").co2.resample("1min").mean().dropna()
    t = (c.index - c.index[0]).total_seconds().values / 60
    best, tk = np.inf, None
    for k in range(5, len(t) - 5):
        X = np.column_stack([np.ones_like(t), t, np.clip(t - t[k], 0, None)])
        sse = np.sum((c.values - X @ np.linalg.lstsq(X, c.values, rcond=None)[0]) ** 2)
        if sse < best:
            best, tk = sse, c.index[k]
    return tk


def ac_lookup(co2: dict) -> tuple[dict, dict]:
    """節次 → 冷氣（1/0）；後半冷氣的節次另回傳開啟時間。Stroop 與量表在下課後施測，後半冷氣一律算開。"""
    status = {k: v.replace("（推定）", "") for k, v in AC.load().items()}
    ac = {s: (1.0 if v in ("冷氣", "後半冷氣") else 0.0 if v == "無冷氣" else np.nan) for s, v in status.items()}
    switch = {s: ac_switch_time(co2, s) for s, v in status.items() if v == "後半冷氣"}
    return ac, switch


# ---------------------------------------------------------------- 資料
def person_level(co2, ac, headcount) -> dict[str, pd.DataFrame]:
    """疲勞（每份問卷）與 Stroop（每次施測）各一張表，附上當下的環境值與該節的控制變因。"""
    fat = D.load_fatigue(co2)
    fat = fat[fat.attention_ok].copy()
    trials = D.load_stroop_trials()
    st = D.stroop_person_sessions(trials, co2)
    # 睡眠取同一節課的問卷（量表與 Stroop 在同一個下課時間施測）
    sleep = fat.groupby(["student_id", "block"])[["sleep_h", "sleep_q"]].mean().reset_index()
    st = st.merge(sleep, on=["student_id", "block"], how="left")
    out = {}
    for name, d in (("fatigue", fat), ("stroop", st)):
        d = D.add_session_co2(d, session_col="block")
        d = d.rename(columns={"temp_Sess": "temp", "rh_Sess": "rh"})
        d["co2h"] = d.co2_Sess / 100
        d["ac"] = d.block.map(ac)
        d["headcount"] = d.block.map(headcount)
        d["student"] = d.student_id.astype(str)
        out[name] = d
    return out


def window_level(co2, ac, switch, headcount, sleep) -> dict[str, pd.DataFrame]:
    """心率與 HRV：每 5 分鐘一筆，只用能對應到學生的裝置。"""
    dm, _ = D.load_code_table()
    dm["dev"] = dm.device.map(D.device_column)
    dm = dm.dropna(subset=["dev"])
    dm["wearer"] = dm.dev + " " + dm.block
    hw = D.add_session_co2(D.add_window_env(D.hr_windows(D.load_hr_seconds()), co2))
    rw = D.add_session_co2(D.add_window_env(D.rmssd_windows(min_good=HRV_CLEAN), co2))
    rw["log_rmssd"] = np.log(rw.rmssd)
    out = {}
    for name, w in (("hr", hw), ("hrv", rw)):
        w = w.merge(dm[["wearer", "student_id", "block"]], on="wearer")
        w = w.rename(columns={"temp_Sess": "temp", "rh_Sess": "rh"})
        w["co2h"] = w.co2_Sess / 100
        w["ac"] = w.block.map(ac)
        for s, t in switch.items():   # 後半冷氣：開啟前的時段算關
            w.loc[(w.block == s) & (w.win + pd.Timedelta(minutes=5) <= t), "ac"] = 0.0
        w["headcount"] = w.block.map(headcount)
        w = w.merge(sleep, on=["student_id", "block"], how="left")
        w["student"] = w.student_id.astype(str)
        w["stu_block"] = w.student + "_" + w.block
        out[name] = w
    return out


# ---------------------------------------------------------------- 模型
def fit(d: pd.DataFrame, y: str, window: bool, controls: list[str]):
    rhs = " + ".join([v for v, _ in IVS] + [c for c in controls if d[c].nunique() > 1])
    f = f"{y} ~ {rhs}"
    vc = {"blk": "0 + C(stu_block)"} if window else None
    m = smf.mixedlm(f, d, groups=d.student, vc_formula=vc)
    for method in ("lbfgs", "powell", "nm"):
        try:
            r = m.fit(reml=True, method=method)
            if np.isfinite(r.llf):
                return r, f
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"無法收斂：{f}")


def vif(d: pd.DataFrame, cols: list[str]) -> dict:
    X = d[cols].astype(float)
    out = {}
    for c in cols:
        others = [o for o in cols if o != c]
        A = np.column_stack([np.ones(len(X))] + [X[o] for o in others])
        resid = X[c] - A @ np.linalg.lstsq(A, X[c], rcond=None)[0]
        r2 = 1 - resid.var() / X[c].var()
        out[c] = 1 / (1 - r2) if r2 < 1 else np.inf
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    co2 = D.load_all_co2()
    ac, switch = ac_lookup(co2)
    _, bi = D.load_code_table()
    headcount = dict(zip(bi.block, bi.headcount))
    pl = person_level(co2, ac, headcount)
    sleep = pl["fatigue"].groupby(["student_id", "block"])[["sleep_h", "sleep_q"]].mean().reset_index()
    wl = window_level(co2, ac, switch, headcount, sleep)
    source = {"fatigue_now": pl["fatigue"], "rt_mean": pl["stroop"], "accuracy": pl["stroop"],
              "interference": pl["stroop"], "hr": wl["hr"], "log_rmssd": wl["hrv"]}

    coef, fits, sizes, vifs = [], [], [], []
    labels = dict(IVS + CONTROLS)
    for y, name, unit, level in DVS:
        ctrl = controls_for(y)
        need = [v for v, _ in IVS] + ctrl
        raw = source[y].dropna(subset=[y])
        d = raw.dropna(subset=need).reset_index(drop=True)
        window = level == "5 分鐘"
        # 缺漏原因：沒有 CO₂（09-17 兩臺都沒收錄）、沒有睡眠（該次沒填量表）
        sizes.append({"依變量": name, "資料層級": level, "有測到的筆數": len(raw),
                      "有測到的學生": raw.student.nunique(), "缺 CO₂／溫濕度": int(raw[["co2h", "temp", "rh"]].isna().any(axis=1).sum()),
                      "控制睡眠": "是" if "sleep_h" in ctrl else "否",
                      "納入模型的筆數": len(d), "納入模型的學生": d.student.nunique(),
                      "節次數": d.block.nunique(), "每位學生筆數（最少–最多）":
                      f"{d.groupby('student').size().min()}–{d.groupby('student').size().max()}"})
        r, f = fit(d, y, window, ctrl)
        used = [v for v in need if v in r.fe_params.index]
        vifs.append({"依變量": name, **{labels[k]: v for k, v in vif(d, used).items()}})
        ci = r.conf_int()
        sd_y = d[y].std()
        for term in used:
            coef.append({"依變量": name, "單位": unit, "變項": labels[term],
                         "角色": "自變量" if term in dict(IVS) else "控制變因",
                         "係數": r.fe_params[term], "SE": r.bse[term], "95% CI 下限": ci.loc[term, 0],
                         "95% CI 上限": ci.loc[term, 1], "p": r.pvalues[term],
                         "標準化係數": r.fe_params[term] * d[term].std() / sd_y})
        for term in [c for c in ctrl if c not in used]:
            coef.append({"依變量": name, "單位": unit, "變項": labels[term], "角色": "控制變因",
                         "係數": np.nan, "p": np.nan, "備註": "此資料中沒有變化，無法估計"})
        X = r.model.exog
        var_f = np.var(X @ r.fe_params.values)
        var_u = float(np.sum(r.cov_re.values)) if r.cov_re.size else 0.0
        var_b = float(np.sum(r.vcomp)) if window else np.nan
        var_re = var_u + (var_b if window else 0.0)
        tot = var_f + var_re + r.scale
        fits.append({"依變量": name, "公式": f + ("  +  (1 | 學生) + (1 | 學生×節次)" if window else "  +  (1 | 學生)"),
                     "筆數": len(d), "學生": d.student.nunique(), "邊際 R²": var_f / tot,
                     "條件 R²": (var_f + var_re) / tot, "學生間變異": var_u, "學生×節次變異": var_b,
                     "殘差變異": r.scale})
        d.drop(columns=[c for c in d.columns if c.startswith(("nread", "co2h_Wa"))], errors="ignore").to_csv(
            D.PRIVATE_DIR / f"lmm_{y}.csv", index=False, encoding="utf-8-sig")

    # 敏感度（做法 A）：每個模型都控制睡眠；沒填量表的人次會被排除
    sens = []
    for y, name, unit, level in DVS:
        ctrl = controls_for(y, sleep_all=True)
        d = source[y].dropna(subset=[y] + [v for v, _ in IVS] + ctrl).reset_index(drop=True)
        r, _ = fit(d, y, level == "5 分鐘", ctrl)
        ci = r.conf_int()
        for term, lab in IVS:
            sens.append({"依變量": name, "變項": lab, "筆數": len(d), "學生": d.student.nunique(),
                         "係數": r.fe_params[term], "95% CI 下限": ci.loc[term, 0], "95% CI 上限": ci.loc[term, 1],
                         "p": r.pvalues[term], "標準化係數": r.fe_params[term] * d[term].std() / d[y].std()})
    pd.DataFrame(sens).round(4).to_csv(OUT / "lmm_sensitivity_sleep_all.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(sizes).to_csv(OUT / "lmm_sample_size.csv", index=False, encoding="utf-8-sig")
    coef = pd.DataFrame(coef)
    coef.round(4).to_csv(OUT / "lmm_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fits).round(4).to_csv(OUT / "lmm_model_fit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(vifs).round(2).to_csv(OUT / "lmm_vif.csv", index=False, encoding="utf-8-sig")
    # 各節次各依變量的學生數（說明樣本數為何不同）
    per = {name: source[y].dropna(subset=[y]).groupby("block").student.nunique() for y, name, _, _ in DVS}
    per = pd.DataFrame(per).fillna(0).astype(int)
    per.insert(0, "冷氣", pd.Series(ac).reindex(per.index).map({1.0: "開", 0.0: "關"}).fillna("—"))
    per.insert(1, "現場人數", pd.Series(headcount).reindex(per.index).astype("Int64"))
    per.to_csv(OUT / "lmm_n_by_session.csv", encoding="utf-8-sig")
    pd.Series({k: str(v) for k, v in switch.items()}, name="冷氣開啟時間（推定）").to_csv(
        OUT / "ac_switch_time.csv", encoding="utf-8-sig")
    import lmm_figures
    lmm_figures.main()
    pd.set_option("display.width", 250)
    print(pd.DataFrame(sizes).to_string())
    print(coef.round(4).to_string())
    print(pd.DataFrame(fits).round(3).to_string())
    print(pd.DataFrame(vifs).round(1).to_string())
    print(per.to_string())
    print(switch)


if __name__ == "__main__":
    main()
