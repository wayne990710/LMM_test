"""徑路分析（piecewise SEM）：每條結構方程用「學生隨機截距」LMM 估計，間接效果用以學生為單位的 cluster bootstrap。

模型 A（全部人次）：
    疲勞      ~ CO2 + 溫度 + 濕度 + 睡眠時數 + 睡眠品質
    反應時間  ~ 疲勞 + CO2 + 溫度 + 濕度 + 第幾次施測
    干擾分數  ~ 疲勞 + CO2 + 溫度 + 濕度 + 第幾次施測
模型 B（有可信心率的子樣本）：在 CO2 與疲勞之間加入「施測前 10 分鐘心率」
    心率      ~ CO2 + 溫度 + 濕度
    疲勞      ~ 心率 + CO2 + 睡眠時數 + 睡眠品質
    反應時間  ~ 疲勞 + 心率 + CO2 + 第幾次施測
整體適配用 d-separation（Shipley's Fisher's C）：模型沒畫的路徑，加進去後應該不顯著。

執行：python src/path_analysis.py（需先跑過 run_analysis.py 產生 data/private/ 的資料表）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import data as D
import models as M

RES = D.ROOT / "results"
N_BOOT = 2000

LABELS = {"co2h": "CO₂（每100 ppm）", "temp": "溫度 (°C)", "rh": "濕度 (%)", "sleep_h": "睡眠時數",
          "sleep_q": "睡眠品質", "test_no": "第幾次施測", "fatigue_now": "自覺疲勞", "hr_pre": "施測前心率",
          "rt_mean": "反應時間 (ms)", "interference": "干擾分數 (ms)"}

MODELS = {
    "A": {"eqs": {"fatigue_now": ["co2h", "temp", "rh", "sleep_h", "sleep_q"],
                  "rt_mean": ["fatigue_now", "co2h", "temp", "rh", "test_no"],
                  "interference": ["fatigue_now", "co2h", "temp", "rh", "test_no"]},
          # 模型沒畫的路徑（d-separation 檢驗）：(結果, 被省略的預測變項)
          "missing": [("rt_mean", "sleep_h"), ("rt_mean", "sleep_q"), ("interference", "sleep_h"),
                      ("interference", "sleep_q"), ("fatigue_now", "test_no")],
          "indirect": [("co2h", "fatigue_now", "rt_mean"), ("co2h", "fatigue_now", "interference"),
                       ("temp", "fatigue_now", "rt_mean"), ("rh", "fatigue_now", "rt_mean"),
                       ("sleep_q", "fatigue_now", "rt_mean")]},
    "B": {"eqs": {"hr_pre": ["co2h", "temp", "rh"],
                  "fatigue_now": ["hr_pre", "co2h", "sleep_h", "sleep_q"],
                  "rt_mean": ["fatigue_now", "hr_pre", "co2h", "test_no"]},
          "missing": [("fatigue_now", "temp"), ("fatigue_now", "rh"), ("rt_mean", "temp"), ("rt_mean", "rh"),
                      ("rt_mean", "sleep_q")],
          "indirect": [("co2h", "hr_pre", "rt_mean"), ("co2h", "hr_pre", "fatigue_now"),
                       ("co2h", "fatigue_now", "rt_mean"), ("hr_pre", "fatigue_now", "rt_mean")]},
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ps = pd.read_csv(D.PRIVATE_DIR / "stroop_person_sessions.csv", parse_dates=["t0"])
    ps = ps.rename(columns={"temp_Avg": "temp", "rh_Avg": "rh"})
    ps["co2h"] = ps.co2_Avg / 100
    a = ps.dropna(subset=["co2h", "temp", "rh", "fatigue_now", "sleep_h", "sleep_q", "rt_mean",
                          "interference"]).reset_index(drop=True)

    # 施測前 10 分鐘心率：研究代碼對照表 → 裝置；hr_5min_windows 已只含品質 good 的貼片
    hw = pd.read_csv(D.PRIVATE_DIR / "hr_5min_windows.csv", parse_dates=["win"])
    dm, _ = D.load_code_table()
    dm["dev"] = dm.device.map(D.device_column)
    dm = dm.dropna(subset=["dev"])
    dm["wearer"] = dm.dev + " " + dm.block
    pre = hw.merge(dm[["wearer", "student_id", "block"]], on="wearer")
    pre["student_id"] = pre.student_id.astype(str)
    a["student_id"] = a.student_id.astype(str)
    pre = pre.merge(a[["student_id", "block", "t0"]], on=["student_id", "block"])
    pre = pre[(pre.win >= pre.t0 - pd.Timedelta(minutes=10)) & (pre.win < pre.t0)]
    hr_pre = pre.groupby(["student_id", "block"]).hr.mean().rename("hr_pre").reset_index()
    b = a.merge(hr_pre, on=["student_id", "block"]).reset_index(drop=True)
    return a, b


def fit_paths(df: pd.DataFrame, eqs: dict) -> dict:
    out = {}
    for y, xs in eqs.items():
        r = M.fit_lmm(f"{y} ~ {' + '.join(xs)}", df, "student")
        out[y] = r
    return out


def path_table(df, eqs, fits) -> pd.DataFrame:
    rows = []
    sd = df.std(numeric_only=True)
    for y, xs in eqs.items():
        r = fits[y]
        ci = r.conf_int()
        for x in xs:
            b = r.fe_params[x]
            rows.append({"to": y, "from": x, "b": b, "se": r.bse[x], "ci_low": ci.loc[x, 0], "ci_high": ci.loc[x, 1],
                         "p": r.pvalues[x], "beta_std": b * sd[x] / sd[y]})
    return pd.DataFrame(rows)


def dsep(df, eqs, missing) -> dict:
    ps = []
    for y, x in missing:
        r = M.fit_lmm(f"{y} ~ {' + '.join(eqs[y] + [x])}", df, "student")
        ps.append(max(r.pvalues[x], 1e-12))
    C = -2 * np.sum(np.log(ps))
    return {"fisher_C": C, "df": 2 * len(ps), "p": stats.chi2.sf(C, 2 * len(ps)), "claims": len(ps)}


def cluster_boot(df, eqs, indirect, n_boot=N_BOOT, seed=0) -> pd.DataFrame:
    """以學生為單位重抽（同一人被抽到兩次視為兩個不同的人），算每條路徑與間接效果的百分位信賴區間。"""
    rng = np.random.default_rng(seed)
    students = df.student.unique()
    groups = {s: df[df.student == s] for s in students}
    rec = []
    for k in range(n_boot):
        pick = rng.choice(students, len(students), replace=True)
        bd = pd.concat([groups[s].assign(student=f"{s}_{i}") for i, s in enumerate(pick)], ignore_index=True)
        try:
            fits = fit_paths(bd, eqs)
        except Exception:  # noqa: BLE001 — 偶爾重抽樣本共線，跳過
            continue
        row = {f"{x}->{y}": fits[y].fe_params[x] for y, xs in eqs.items() for x in xs}
        for x, m, y in indirect:
            row[f"{x}->{m}->{y}"] = fits[m].fe_params[x] * fits[y].fe_params[m]
        rec.append(row)
    return pd.DataFrame(rec)


def run_model(name, df) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    spec = MODELS[name]
    fits = fit_paths(df, spec["eqs"])
    paths = path_table(df, spec["eqs"], fits)
    boot = cluster_boot(df, spec["eqs"], spec["indirect"])
    paths["boot_ci_low"] = [boot[f"{r['from']}->{r['to']}"].quantile(0.025) for _, r in paths.iterrows()]
    paths["boot_ci_high"] = [boot[f"{r['from']}->{r['to']}"].quantile(0.975) for _, r in paths.iterrows()]
    ind = []
    for x, m, y in spec["indirect"]:
        est = fits[m].fe_params[x] * fits[y].fe_params[m]
        v = boot[f"{x}->{m}->{y}"]
        ind.append({"path": f"{x} → {m} → {y}", "indirect_b": est, "boot_ci_low": v.quantile(0.025),
                    "boot_ci_high": v.quantile(0.975),
                    # 雙尾 bootstrap p：重抽分布落在 0 另一側的比例 × 2
                    "boot_p": min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))})
    fit = {"model": name, "n": len(df), "students": df.student.nunique(), "boot_reps": len(boot),
           **dsep(df, spec["eqs"], spec["missing"])}
    for t in (paths, ind := pd.DataFrame(ind)):
        t.insert(0, "model", name)
    return paths, ind, fit


def main():
    a, b = load_data()
    allp, alli, allf = [], [], []
    for name, df in (("A", a), ("B", b)):
        p, i, f = run_model(name, df)
        allp.append(p), alli.append(i), allf.append(f)
        print(f"模型 {name}: n={f['n']}，{f['students']} 人，Fisher's C={f['fisher_C']:.2f}, p={f['p']:.3f}")
    paths, ind, fit = pd.concat(allp), pd.concat(alli), pd.DataFrame(allf)
    paths.round(4).to_csv(RES / "path_coefficients.csv", index=False, encoding="utf-8-sig")
    ind.round(4).to_csv(RES / "path_indirect_effects.csv", index=False, encoding="utf-8-sig")
    fit.round(4).to_csv(RES / "path_model_fit.csv", index=False, encoding="utf-8-sig")
    import figures as F
    exo = {n: {("co2h", "temp"): d.co2h.corr(d.temp), ("co2h", "rh"): d.co2h.corr(d.rh)}
           for n, d in (("A", a), ("B", b))}
    F.path_diagram(paths, fit, LABELS, exo)
    print(paths.round(3).to_string())
    print(ind.round(3).to_string())


if __name__ == "__main__":
    main()
