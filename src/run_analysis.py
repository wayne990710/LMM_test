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
import seating_test as ST

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
    fat = D.load_fatigue(co2)
    # 假名：用所有資料來源的學生代碼聯集編號，兩份資料的 S01 才會是同一個人
    pseudo = D.pseudonyms(set(trials.Student_ID) | set(fat.student_id))
    ps = add_exposures(D.stroop_person_sessions(trials, co2))
    ps["student"] = ps.student_id.map(pseudo)
    # 睡眠（試前調查）併進 Stroop：同一區塊的問卷
    sleep = fat[fat.attention_ok].groupby(["student_id", "block"])[["sleep_h", "sleep_q", "fatigue_now"]].mean()
    ps = ps.merge(sleep, left_on=["student_id", "block"], right_index=True, how="left")
    ps.to_csv(D.PRIVATE_DIR / "stroop_person_sessions.csv", index=False, encoding="utf-8-sig")
    both = ps.dropna(subset=["co2_Wa1", "co2_Wa2"]).reset_index(drop=True)

    res = []
    for y, name in [("rt_mean", "Stroop 平均反應時間 (ms)"), ("interference", "Stroop 干擾分數 (ms)"),
                    ("accuracy", "Stroop 正確率")]:
        res.append(compare(both, y, "student", "test_no", name, "人次"))
    # 敏感度：再控制溫濕度
    res.append(compare(both, "rt_mean", "student", "test_no", "Stroop 平均反應時間（+溫濕度）", "人次",
                       cv=False, extra=" + temp_{s} + rh_{s}"))

    # 敏感度：再控制睡眠時數與品質（計畫書的共變量）
    bs = both.dropna(subset=["sleep_h", "sleep_q"]).reset_index(drop=True)
    res.append(compare(bs, "rt_mean", "student", "test_no + sleep_h + sleep_q", "Stroop 平均反應時間（+睡眠）",
                       "人次", cv=False))

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

    # ---------- 3b. 自覺疲勞量表
    fat["student"] = fat.student_id.map(pseudo)
    fat = add_exposures(fat)
    fat.to_csv(D.PRIVATE_DIR / "fatigue_with_co2.csv", index=False, encoding="utf-8-sig")
    fat_ok = fat[fat.attention_ok]
    fboth = fat_ok.dropna(subset=["co2_Wa1", "co2_Wa2"]).reset_index(drop=True)
    fcov = "resp_no + sleep_h + sleep_q"
    res.append(compare(fboth, "fatigue_now", "student", fcov, "自覺疲勞（現在，1–7）", "人次"))
    # 負對照：FSS 問的是「過去 24 小時」，不該跟填寫當下的 CO2 有關；若有關代表有別的混淆
    res.append(compare(fboth, "fss", "student", fcov, "FSS 過去 24 小時（負對照）", "人次", cv=False))
    # 上午／下午能不能解釋掉：加入 slot
    fboth["pm"] = (fboth.block.str.endswith("PM")).astype(int)
    res.append(compare(fboth, "fatigue_now", "student", fcov + " + pm", "自覺疲勞（+上午/下午）", "人次", cv=False))
    fwa1 = fat_ok.dropna(subset=["co2_Wa1"]).reset_index(drop=True)
    fwa1["pm"] = (fwa1.block.str.endswith("PM")).astype(int)
    for cov, lab in [(fcov, "自覺疲勞（Wa1 全部場次）"), (fcov + " + pm", "自覺疲勞（Wa1 全部場次，+上午/下午）")]:
        r = M.fit_lmm(f"fatigue_now ~ co2h_Wa1 + {cov}", fwa1, "student")
        res.append(pd.DataFrame([{"outcome": lab, "unit": "人次", "exposure": "Wa1", "n": len(fwa1),
                                  "groups": fwa1.student.nunique(), **M.coef_row(r, "co2h_Wa1")}]))

    # ---------- 3c. 座位表有沒有用
    seat_ex = pd.DataFrame([ST.exhaustive(both, "rt_mean", "test_no"),
                            ST.exhaustive(fboth, "fatigue_now", fcov)])
    seat_ex.round(4).to_csv(RES / "seating_exhaustive.csv", index=False, encoding="utf-8-sig")
    seat_sim = pd.concat([ST.simulate(both, "rt_mean", "test_no"),
                          ST.simulate(fboth, "fatigue_now", fcov, betas=(0, 0.1, 0.2, 0.4))])
    seat_sim.round(3).to_csv(RES / "seating_simulation.csv", index=False, encoding="utf-8-sig")

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

    # ---------- 4b. 逐人生理（需要貼片字母 → 編號對照；沒有就跳過）
    letters = D.load_patch_letters()
    if letters is not None:
        dm = D.load_device_map().merge(letters, left_on=["block", "device"], right_on=["block", "letter"])
        dm["wearer"] = dm.patch + " " + dm.block
        dm["student"] = dm.student_id.map(pseudo)
        pw = hwb.merge(dm[["wearer", "student"]], on="wearer")
        res.append(compare(pw, "hr", "student", "elapsed_min", "貼片心率，逐人（控制上課經過時間）", "5 分鐘",
                           cv=False))
        # 施測前 5 分鐘的心率當中介變項：CO2 → 心率 → 反應時間
        pre = hw.merge(dm[["wearer", "student_id", "block"]], on="wearer")
        pre = pre.merge(ps[["student_id", "block", "t0"]], on=["student_id", "block"])
        pre = pre[(pre.win >= pre.t0 - pd.Timedelta(minutes=10)) & (pre.win < pre.t0)]
        hr_pre = pre.groupby(["student_id", "block"]).hr.mean().rename("hr_pre")
        med = both.merge(hr_pre, left_on=["student_id", "block"], right_index=True).reset_index(drop=True)
        if med.student.nunique() >= 3:
            res.append(compare(med, "rt_mean", "student", "test_no + hr_pre", "反應時間（+施測前心率）", "人次",
                               cv=False))
    else:
        print("※ 沒有 data/private/patch_letters.csv，跳過逐人生理分析")

    allres = pd.concat(res, ignore_index=True)
    allres.round(4).to_csv(RES / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # ---------- 5. 區塊層級摘要（可公開：每區塊 ≥ 5 人的平均）
    blk = ps.groupby("block").agg(n_students=("student", "nunique"), co2_Wa1=("co2_Wa1", "mean"),
                                  co2_Wa2=("co2_Wa2", "mean"), temp_Wa1=("temp_Wa1", "mean"),
                                  temp_Wa2=("temp_Wa2", "mean"), rh_Wa1=("rh_Wa1", "mean"), rh_Wa2=("rh_Wa2", "mean"),
                                  rt_mean=("rt_mean", "mean"), rt_sd=("rt_mean", "std"),
                                  interference=("interference", "mean"), accuracy=("accuracy", "mean"),
                                  mean_test_no=("test_no", "mean")).reset_index()
    fb = fat_ok.groupby("block").agg(fatigue_n=("fatigue_now", "size"), fatigue_now=("fatigue_now", "mean"),
                                     fss=("fss", "mean"))
    blk = blk.merge(fb, left_on="block", right_index=True, how="left")
    hrb = hw.groupby("session").agg(hr_windows=("hr", "size"), wearers=("wearer", "nunique"), hr_mean=("hr", "mean"))
    blk = blk.merge(hrb, left_on="block", right_index=True, how="left")
    blk.round(2).to_csv(RES / "session_summary.csv", index=False, encoding="utf-8-sig")

    # 各區塊 CO2 與「第幾次施測」的相關：練習效應與 CO2 是否糾纏在一起
    conf = {s: np.corrcoef(both[f"co2_{s}"], both.test_no)[0, 1] for s in ("Wa1", "Wa2")}
    conf["slot_vs_co2_Wa1"] = np.corrcoef(both.co2_Wa1, (both.slot == "PM").astype(int))[0, 1]
    pd.Series(conf, name="r").round(3).to_csv(RES / "confounding_check.csv", encoding="utf-8-sig")

    F.make_all(co2, pair, blk, hw, allres, trials, seat_sim)
    print(allres.round(3).to_string())


if __name__ == "__main__":
    main()
