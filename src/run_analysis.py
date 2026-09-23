"""主程式：整理資料 → 分別用 Wa1、Wa2（與兩台平均）的 CO2 建 LMM → 比較哪一台與結果最貼合。

執行：python src/run_analysis.py
輸出：results/（模型層級結果，可公開）、data/private/（逐人資料，已 gitignore）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import data as D
import figures as F
import models as M

RES = D.ROOT / "results"
EXPOSURES = ["Wa1", "Wa2", "Avg"]  # Avg = 兩台平均，代表「整間教室」


def add_exposures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for v in ("co2", "temp", "rh"):
        df[f"{v}_Avg"] = df[[f"{v}_Wa1", f"{v}_Wa2"]].mean(axis=1, skipna=False)
    for s in EXPOSURES:
        df[f"co2h_{s}"] = df[f"co2_{s}"] / 100  # 以每 100 ppm 為單位，係數比較好讀
    return df


def compare(df, y, group, covars, outcome, unit, cv=True, repeats=10, extra=""):
    """同一批資料列上，對每個暴露來源各跑：基準模型 vs 基準 + CO2。"""
    rows = []
    base_f = f"{y} ~ {covars}" if covars else f"{y} ~ 1"
    for s in EXPOSURES:
        term = f"co2h_{s}"
        f = f"{y} ~ {term}" + (f" + {covars}" if covars else "") + extra.format(s=s)
        r0, r1 = M.fit_lmm(base_f + extra.format(s=s), df, group), M.fit_lmm(f, df, group)
        lr = 2 * (r1.llf - r0.llf)
        r2m, r2c = M.nakagawa_r2(r1)
        row = {"outcome": outcome, "unit": unit, "exposure": s, "n": len(df), "groups": df[group].nunique(),
               **M.coef_row(r1, term), "aic": M.aic(r1), "delta_aic_vs_base": M.aic(r1) - M.aic(r0),
               "lr_p": stats.chi2.sf(max(lr, 0), 1), "r2_marginal": r2m, "r2_conditional": r2c}
        if cv:
            row["loso_rmse"] = M.loso_rmse(f, df, group, y)
            row["loso_rmse_base"] = M.loso_rmse(base_f + extra.format(s=s), df, group, y)
            row["cv5_rmse"], row["cv5_r2"] = M.kfold_rmse(f, df, group, y, repeats=repeats)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["delta_aic_vs_best"] = out.aic - out.aic.min()
    return out


def main():
    RES.mkdir(exist_ok=True)
    D.PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    co2 = D.load_all_co2()

    # ---------- 1. 兩台感測器一致性
    pair = D.paired_sensors(co2)
    pair = pair[pair.time >= "2026-09-21"]
    agree = []
    for sess, g in [("全部", pair)] + list(pair.groupby("session")):
        d = g.co2_Wa1 - g.co2_Wa2
        agree.append({"session": sess, "n_pairs": len(g), "wa1_mean": g.co2_Wa1.mean(), "wa2_mean": g.co2_Wa2.mean(),
                      "mean_diff_wa1_minus_wa2": d.mean(), "sd_diff": d.std(),
                      "loa_low": d.mean() - 1.96 * d.std(), "loa_high": d.mean() + 1.96 * d.std(),
                      "pearson_r": np.corrcoef(g.co2_Wa1, g.co2_Wa2)[0, 1],
                      "temp_diff": (g.temp_Wa1 - g.temp_Wa2).mean(), "rh_diff": (g.rh_Wa1 - g.rh_Wa2).mean()})
    pd.DataFrame(agree).round(3).to_csv(RES / "sensor_agreement.csv", index=False, encoding="utf-8-sig")

    # ---------- 2. 暴露情形（對照計畫書 1000 / 1500 / 2000 ppm 門檻）
    expo = []
    for s, c in co2.items():
        c = c[c.time >= "2026-09-21"].copy()
        c["session"] = D._session_id(c.time)
        for sess, g in c.groupby("session"):
            expo.append({"sensor": s, "session": sess, "start": g.time.min().strftime("%H:%M"),
                         "end": g.time.max().strftime("%H:%M"), "co2_mean": g.co2.mean(), "co2_max": g.co2.max(),
                         "pct_over_1000": (g.co2 > 1000).mean() * 100, "pct_over_1500": (g.co2 > 1500).mean() * 100,
                         "pct_over_2000": (g.co2 > 2000).mean() * 100})
    pd.DataFrame(expo).round(1).to_csv(RES / "co2_exposure_by_session.csv", index=False, encoding="utf-8-sig")

    # ---------- 3. Stroop
    trials = D.load_stroop_trials()
    ps = add_exposures(D.stroop_person_sessions(trials, co2))
    ps.to_csv(D.PRIVATE_DIR / "stroop_person_sessions.csv", index=False, encoding="utf-8-sig")
    both = ps.dropna(subset=["co2_Wa1", "co2_Wa2"]).reset_index(drop=True)

    res = []
    for y, name in [("rt_mean", "Stroop 平均反應時間 (ms)"), ("interference", "Stroop 干擾分數 (ms)"),
                    ("accuracy", "Stroop 正確率")]:
        res.append(compare(both, y, "student", "test_no", name, "人次"))
    # 敏感度：再控制溫濕度
    res.append(compare(both, "rt_mean", "student", "test_no", "Stroop 平均反應時間（+溫濕度）", "人次",
                       cv=False, extra=" + temp_{s} + rh_{s}"))

    tr = D.add_trial_env(trials, ps)
    tr = add_exposures(tr.dropna(subset=["co2_Wa1", "co2_Wa2"]))
    tr = tr[(tr.correct == 1) & tr.rt.between(200, 3000)].reset_index(drop=True)
    tr["log_rt"] = np.log(tr.rt)
    tr["incong"] = (tr.condition == "incongruent").astype(int)
    tr["cong"] = (tr.condition == "congruent").astype(int)
    trial_rows = []
    for s in EXPOSURES:
        r = M.fit_lmm(f"log_rt ~ co2h_{s} * incong + co2h_{s} * cong + test_no", tr, "student")
        r0 = M.fit_lmm("log_rt ~ incong + cong + test_no", tr, "student")
        r2m, r2c = M.nakagawa_r2(r)
        for term, lab in [(f"co2h_{s}", "CO2 主效應（中性題，log RT）"), (f"co2h_{s}:incong", "CO2 × 不一致（干擾）")]:
            trial_rows.append({"outcome": lab, "unit": "單題", "exposure": s, "n": len(tr), "groups": tr.student.nunique(),
                               **M.coef_row(r, term), "aic": M.aic(r), "delta_aic_vs_base": M.aic(r) - M.aic(r0),
                               "lr_p": stats.chi2.sf(max(2 * (r.llf - r0.llf), 0), 3),
                               "r2_marginal": r2m, "r2_conditional": r2c})
    tt = pd.DataFrame(trial_rows)
    tt["delta_aic_vs_best"] = tt.aic - tt[tt.outcome == tt.outcome.iloc[0]].aic.min()
    res.append(tt)

    # 補充：Wa1 單獨可多用 09-21 上午那一場（Wa2 當時沒資料）
    wa1_all = ps.dropna(subset=["co2_Wa1"]).reset_index(drop=True)
    r = M.fit_lmm("rt_mean ~ co2h_Wa1 + test_no", wa1_all, "student")
    wa1_extra = {"outcome": "Stroop 平均反應時間（Wa1 全部場次）", "unit": "人次", "exposure": "Wa1",
                 "n": len(wa1_all), "groups": wa1_all.student.nunique(), **M.coef_row(r, "co2h_Wa1")}
    res.append(pd.DataFrame([wa1_extra]))

    # ---------- 4. 心率（5 分鐘）與 HRV
    hr = D.load_hr_seconds()
    hw = add_exposures(D.add_window_env(D.hr_windows(hr), co2))
    hw.to_csv(D.PRIVATE_DIR / "hr_5min_windows.csv", index=False, encoding="utf-8-sig")
    hwb = hw.dropna(subset=["co2_Wa1", "co2_Wa2"]).reset_index(drop=True)
    res.append(compare(hwb, "hr", "wearer", "C(device_type)", "心率 (bpm)，未控制時間", "5 分鐘", repeats=5))
    res.append(compare(hwb, "hr", "wearer", "elapsed_min + C(device_type)", "心率 (bpm)，控制上課經過時間",
                       "5 分鐘", repeats=5))

    rw = add_exposures(D.add_window_env(D.rmssd_windows(), co2))
    rw.to_csv(D.PRIVATE_DIR / "rmssd_5min_windows.csv", index=False, encoding="utf-8-sig")
    rwb = rw.dropna(subset=["co2_Wa1", "co2_Wa2"]).reset_index(drop=True)
    rwb["log_rmssd"] = np.log(rwb.rmssd)
    res.append(compare(rwb, "log_rmssd", "wearer", "", "HRV log(RMSSD)，未控制時間", "5 分鐘", repeats=5))
    res.append(compare(rwb, "log_rmssd", "wearer", "elapsed_min", "HRV log(RMSSD)，控制上課經過時間", "5 分鐘",
                       repeats=5))

    allres = pd.concat(res, ignore_index=True)
    allres.round(4).to_csv(RES / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # ---------- 5. 區塊層級摘要（可公開：每區塊 ≥ 5 人的平均）
    blk = ps.groupby("block").agg(n_students=("student", "nunique"), co2_Wa1=("co2_Wa1", "mean"),
                                  co2_Wa2=("co2_Wa2", "mean"), temp_Wa1=("temp_Wa1", "mean"),
                                  temp_Wa2=("temp_Wa2", "mean"), rh_Wa1=("rh_Wa1", "mean"), rh_Wa2=("rh_Wa2", "mean"),
                                  rt_mean=("rt_mean", "mean"), rt_sd=("rt_mean", "std"),
                                  interference=("interference", "mean"), accuracy=("accuracy", "mean"),
                                  mean_test_no=("test_no", "mean")).reset_index()
    hrb = hw.groupby("session").agg(hr_windows=("hr", "size"), wearers=("wearer", "nunique"), hr_mean=("hr", "mean"))
    blk = blk.merge(hrb, left_on="block", right_index=True, how="left")
    blk.round(2).to_csv(RES / "session_summary.csv", index=False, encoding="utf-8-sig")

    # 各區塊 CO2 與「第幾次施測」的相關：練習效應與 CO2 是否糾纏在一起
    conf = {s: np.corrcoef(both[f"co2_{s}"], both.test_no)[0, 1] for s in ("Wa1", "Wa2")}
    conf["slot_vs_co2_Wa1"] = np.corrcoef(both.co2_Wa1, (both.slot == "PM").astype(int))[0, 1]
    pd.Series(conf, name="r").round(3).to_csv(RES / "confounding_check.csv", encoding="utf-8-sig")

    F.make_all(co2, pair, blk, hw, allres, trials)
    print(allres.round(3).to_string())


if __name__ == "__main__":
    main()
