"""模擬預測：CO2 質量平衡（通風）→ CO2 軌跡 → 自覺疲勞（LMM + 以學生為單位的 bootstrap）。

1. 質量平衡：關窗開冷氣時 C(t) = Css − (Css − C0)·e^(−λt)（Persily & de Jonge, 2017 的單區模型）。
   擬合每節課的上升段，得到換氣率 λ（次/小時）與穩態濃度 Css；
   g = (Css − C_out)·λ 為「人數 × 每人 CO2 產生率 ÷ 教室體積」，單位 ppm/小時。
2. 情境模擬：在相同 g 下改變 λ，算出一堂 50 分鐘課的 CO2 軌跡。
3. 預測疲勞：疲勞 LMM 的 CO2 係數以學生為單位 bootstrap 2000 次，把係數的不確定性帶進預測。
4. 驗證：留一節次交叉驗證（CO2 是全班共用、以節次為單位變動，所以要整節拿掉）。

執行：python src/prediction.py（需先跑過 run_analysis.py）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import data as D
import models as M

RES = D.ROOT / "results"
C_OUT = 420.0          # 戶外 CO2（ppm）；臺北近年背景值約 420–430
CLASS_MIN = 50
FCOV = "resp_no + sleep_h + sleep_q"


def _rise(t, css, lam, c0):
    return css - (css - c0) * np.exp(-lam * t)


def fit_mass_balance(co2: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for s, c in co2.items():
        c = c[c.time >= "2026-09-21"].copy()
        c["session"] = D._session_id(c.time)
        for ses, g in c.groupby("session"):
            g = g.sort_values("time")
            t = ((g.time - g.time.iloc[0]).dt.total_seconds() / 3600).values
            y = g.co2.values.astype(float)
            keep = t > 2 / 60                       # 去掉開機暖機的前 2 分鐘
            t, y = t[keep], y[keep]
            if len(t) < 30:
                continue
            peak = np.argmax(pd.Series(y).rolling(5, center=True, min_periods=1).median().values)
            t, y = t[:peak + 1] - t[0], y[:peak + 1]   # 只用上升段（之後通常是下課開門）
            if len(t) < 20:
                continue
            try:
                p, _ = curve_fit(_rise, t, y, p0=[2500, 1.0, y[0]],
                                 bounds=([C_OUT, 0.01, 300], [20000, 20, 3000]), maxfev=20000)
            except RuntimeError:
                continue
            r2 = 1 - np.sum((y - _rise(t, *p)) ** 2) / np.sum((y - y.mean()) ** 2)
            rows.append({"sensor": s, "session": ses, "minutes": round(t[-1] * 60), "c_start": y[0], "c_peak": y[-1],
                         "css": p[0], "ach_per_h": p[1], "r2": r2, "g_ppm_per_h": (p[0] - C_OUT) * p[1]})
    f = pd.DataFrame(rows)
    # 可信的擬合：曲線真的有在上升、R² 高、參數沒有卡在邊界
    f["usable"] = (f.r2 >= 0.93) & (f.css < 15000) & (f.ach_per_h > 0.2) & (f.c_peak - f.c_start > 300)
    return f


def trajectory(c0, g, lam, minutes=CLASS_MIN):
    t = np.linspace(0, minutes / 60, minutes + 1)
    css = C_OUT + g / lam
    return t * 60, css - (css - c0) * np.exp(-lam * t)


def boot_fatigue_slope(fat: pd.DataFrame, n_boot=2000, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    students = fat.student.unique()
    groups = {s: fat[fat.student == s] for s in students}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(students, len(students), replace=True)
        bd = pd.concat([groups[s].assign(student=f"{s}_{i}") for i, s in enumerate(pick)], ignore_index=True)
        try:
            out.append(M.fit_lmm(f"fatigue_now ~ co2h + {FCOV}", bd, "student").fe_params["co2h"])
        except Exception:  # noqa: BLE001
            continue
    return np.array(out)


def leave_one_session_out(fat: pd.DataFrame) -> dict:
    err_m, err_0 = [], []
    for b in fat.block.unique():
        tr, te = fat[fat.block != b], fat[fat.block == b]
        r1 = M.fit_lmm(f"fatigue_now ~ co2h + {FCOV}", tr, "student")
        r0 = M.fit_lmm(f"fatigue_now ~ {FCOV}", tr, "student")
        re1 = {k: float(v.iloc[0]) for k, v in r1.random_effects.items()}
        re0 = {k: float(v.iloc[0]) for k, v in r0.random_effects.items()}
        err_m.append(te.fatigue_now - (np.asarray(r1.predict(te)) + te.student.map(re1).fillna(0)))
        err_0.append(te.fatigue_now - (np.asarray(r0.predict(te)) + te.student.map(re0).fillna(0)))
    rm = float(np.sqrt(np.mean(np.concatenate(err_m) ** 2)))
    r0 = float(np.sqrt(np.mean(np.concatenate(err_0) ** 2)))
    return {"sessions": fat.block.nunique(), "n": len(fat), "rmse_with_co2": rm, "rmse_without_co2": r0,
            "improvement_pct": (r0 - rm) / r0 * 100}


def main():
    co2 = D.load_all_co2()
    mb = fit_mass_balance(co2)
    mb.round(3).to_csv(RES / "ventilation_fit.csv", index=False, encoding="utf-8-sig")
    u = mb[mb.usable]
    g_med, lam_med = u.g_ppm_per_h.median(), u.ach_per_h.median()
    c0 = u.c_start.median()
    lam_need = g_med / (1000 - C_OUT)       # 穩態剛好 1000 ppm 所需的換氣率
    print(f"可用擬合 {len(u)} 筆：λ 中位數 {lam_med:.2f} 次/時，g 中位數 {g_med:.0f} ppm/時，"
          f"起始 {c0:.0f} ppm；穩態 ≤ 1000 ppm 需 λ ≥ {lam_need:.2f} 次/時")

    fat = pd.read_csv(D.PRIVATE_DIR / "fatigue_with_co2.csv")
    fat = fat[fat.attention_ok].dropna(subset=["co2_Wa1", "co2_Wa2"]).reset_index(drop=True)
    fat["co2h"] = fat.co2_Avg / 100
    b_hat = M.fit_lmm(f"fatigue_now ~ co2h + {FCOV}", fat, "student").fe_params["co2h"]
    boot = boot_fatigue_slope(fat)
    cv = leave_one_session_out(fat)
    pd.DataFrame([{**cv, "b_co2_per_100ppm": b_hat, "boot_ci_low": np.quantile(boot, 0.025),
                   "boot_ci_high": np.quantile(boot, 0.975), "boot_reps": len(boot)}]).round(4).to_csv(
        RES / "prediction_validation.csv", index=False, encoding="utf-8-sig")

    scen = [("目前（λ 中位數）", lam_med), ("換氣加倍", lam_med * 2), ("穩態 1000 ppm 所需", lam_need)]
    rows, traj = [], {}
    for name, lam in scen:
        t, c = trajectory(c0, g_med, lam)
        d_fat = (c[-1] - c0) / 100 * boot           # 下課時相對上課開始的疲勞變化
        rows.append({"scenario": name, "ach_per_h": lam, "co2_at_50min": c[-1], "co2_mean": c.mean(),
                     "minutes_over_1000": int((c > 1000).sum()),
                     "d_fatigue_end": (c[-1] - c0) / 100 * b_hat,
                     "d_fatigue_ci_low": np.quantile(d_fat, 0.025), "d_fatigue_ci_high": np.quantile(d_fat, 0.975)})
        traj[name] = (t, c)
    sc = pd.DataFrame(rows)
    sc.round(3).to_csv(RES / "prediction_scenarios.csv", index=False, encoding="utf-8-sig")
    import figures as F
    F.prediction_simulation(traj, sc, u)
    print(sc.round(2).to_string())
    print(cv)


if __name__ == "__main__":
    main()
