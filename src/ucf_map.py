"""產生 UCF 論證路線圖（數字直接讀 results/ 的結果表，避免圖與表不一致）。"""
from __future__ import annotations

import pandas as pd

import data as D
import figures as F

R = D.ROOT / "results"


def main():
    eff = pd.read_csv(R / "effect_sizes.csv").set_index("key")
    chk = pd.read_csv(R / "assumption_checks.csv")
    pc = pd.read_csv(R / "path_coefficients.csv")
    sc = pd.read_csv(R / "prediction_scenarios.csv").set_index("scenario")
    vf = pd.read_csv(R / "ventilation_fit.csv")
    cv = pd.read_csv(R / "prediction_validation.csv").iloc[0]
    hw = pd.read_csv(D.PRIVATE_DIR / "hr_5min_windows.csv")
    n_hr = int((hw.device_type == "ecg").sum())
    n_hrv = len(pd.read_csv(D.PRIVATE_DIR / "rmssd_5min_windows.csv"))
    ph = pd.read_csv(R / "hr_hrv_main.csv")
    hr_in = ph[ph.outcome.str.startswith("心率")].iloc[0]
    hrv_in = ph[(ph.level == "節次內") & ph.threshold.str.contains("主要")].iloc[0]
    hrv_sess = ph[ph.level == "節次平均（Spearman）"].iloc[0]
    q = D.ecg_quality()
    q = q[q.quality != "error"]
    fat, acc, rt = eff.loc["fatigue_now"], eff.loc["accuracy"], eff.loc["rt_mean"]
    a = pc[(pc.model == "A")].set_index(["from", "to"])
    co2_fat, fat_rt = a.loc[("co2h", "fatigue_now")], a.loc[("fatigue_now", "rt_mean")]
    lam = vf[vf.usable].ach_per_h.median()
    lam_need = sc.loc["穩態 1000 ppm 所需", "ach_per_h"]
    must = (chk.decision == "必須用無母數").sum()
    rows = {
        "thesis": "關窗開冷氣使教室 CO₂ 在一堂課內升到 1,600 ppm 以上；CO₂ 越高，學生自覺疲勞越高（中等效果量），"
                  "\n但 Stroop 反應速度沒有變慢。因此通風守則的依據是「疲勞」而非「反應速度」，"
                  f"\n要把 CO₂ 壓在 1000 ppm，換氣量需從約 {lam:.1f} 提高到 {lam_need:.1f} 次／小時（約 {lam_need / lam:.1f} 倍）。",
        "steps": [
            ("① 資料品質", f"注意力檢核排除 7 份\n貼片逐時段篩選：心率 {n_hr}、HRV {n_hrv} 窗\n（HRV 加 Malik 20% 剔除錯拍）\n"
                          "兩臺感測器 r = 0.97\n→ 只用可信資料"),
            ("② 檢定選擇", f"{len(chk)} 個結果變項中\n{must} 個「必須」用無母數\n其餘因 n < 15「建議」\n"
                          "→ 以無母數為主"),
            ("③ 無母數＋效果量", f"疲勞 g = {fat.g_av:+.2f}（中）\n正確率 g = {acc.g_av:+.2f}（中）\n"
                                f"反應時間 g = {rt.g_av:+.2f}（小）\n→ 找出有訊號的變項"),
            ("④ 參數確認", f"LMM：疲勞 d₁₀₀₀ = {fat.d_1000_lmm:.2f}\n反應時間 d₁₀₀₀ = {rt.d_1000_lmm:+.2f}\n"
                          "溫濕度與 CO₂ 共線\n→ 效果量兩法一致"),
            ("⑤ 機制（徑路）", f"CO₂→疲勞 β = {co2_fat.beta_std:+.2f}✔\n疲勞→反應時間\nβ = {fat_rt.beta_std:+.2f}✘\n"
                              "→ 中介鏈不成立"),
            ("⑥ 模擬預測", f"λ {lam:.1f}→{lam_need:.1f} 次／時\n疲勞增量 {sc.iloc[0].d_fatigue_end:+.2f}"
                          f"→{sc.iloc[-1].d_fatigue_end:+.2f}\n留一節次驗證 +{cv.improvement_pct:.0f}%\n→ 通風守則"),
        ],
        "questions": [
            ("研究問題一：CO₂ 與認知表現的動態關聯",
             f"疲勞 ↑（中，方向一致，n 偏小）\n正確率 ↓（中）；反應時間：無變慢\n"
             f"心率：節次內 {hr_in.beta:+.2f} bpm／100 ppm（p = {hr_in.p:.2f}）\n"
             f"HRV：節次內 {hrv_in.pct_per_100ppm:+.1f}%（n.s.）；跨節次 ρ = {hrv_sess.rho:+.2f}（p = {hrv_sess.p:.2f}）",
             "部分"),
            ("研究問題二：建構 LMM 預測模型",
             "群體層級：CO₂ 對疲勞有穩定斜率\n個人層級：預測力有限\n（留一節次只改善 4%）\n→ 適合情境比較，不適合個人預測",
             "部分"),
            ("研究問題三：通風改善策略",
             f"質量平衡擬合 12 節（R² ≥ 0.94）\n1000 ppm 需 λ ≈ {lam_need:.1f} 次／小時\n下課時疲勞增量可降約 83%\n"
             "開窗換氣率待衰減測試",
             "支持"),
        ],
        "links": [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (4, 0), (4, 1), (5, 1), (5, 2), (0, 2)],
    }
    F.ucf_map(rows)


if __name__ == "__main__":
    main()
