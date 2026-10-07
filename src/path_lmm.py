"""徑路分析（多層次中介，piecewise SEM）：環境 → 中介因子 → 結果。

依據前面的 LMM 與兩兩相關，候選的中介路徑：
  中介 1：自覺疲勞    環境 → 疲勞 → Stroop（反應時間、正確率、干擾分數）
  中介 2：瞬時心率    環境 → 心率 → Stroop／疲勞
  序列中介：         環境 → 心率 → 疲勞 → Stroop
  心率變異度：同一人次同時有 HRV 與 Stroop／疲勞的資料太少，無法做中介（見 path_feasibility.csv）。

模型設定（與主要 LMM 相同）
  - 分析單位：每位學生每節課一筆；心率取該生該節所有 5 分鐘時段的平均。
  - 環境值：Stroop 施測當下（沒有 Stroop 時用疲勞量表填答前 3 分鐘）。
  - 每條方程式都放入三個自變量（CO₂、溫度、濕度）與控制變因（人數、冷氣）；睡眠只放在疲勞的方程式。
  - 每條方程式都是學生隨機截距 LMM；所有連續變項先標準化（z 分數），係數即標準化 β。
  - 間接效果 = a × b（序列中介 = a × d × b），95% CI 以「學生」為單位 bootstrap 重抽 2,000 次。
  - 模型含所有直接效果（飽和模型），所以不做整體適配檢定，重點放在間接效果。

執行：python src/path_lmm.py（需先跑過 lmm_main.py）
輸出：results/LMM/path_*.csv、04_徑路分析.png
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import data as D
import lmm_main as L

warnings.filterwarnings("ignore")
OUT = D.ROOT / "results" / "LMM"
N_BOOT = 2000
# 各模型的 bootstrap 結果（每個模型約 3–25 分鐘）。保留不刪：中斷後重跑只會補跑還沒完成的模型
PARTS = D.PRIVATE_DIR / "path_parts"
IVS = ["co2", "temp", "rh"]
CTRL = ["headcount", "ac"]
SLEEP = ["sleep_h", "sleep_q"]
LAB = {"co2": "CO₂", "temp": "溫度", "rh": "濕度", "fatigue_now": "疲勞程度", "hr": "瞬時心率",
       "log_rmssd": "心率變異度", "rt_mean": "反應時間", "accuracy": "正確率", "interference": "干擾分數"}
STROOP = ["rt_mean", "accuracy", "interference"]

# 模型：名稱 → (中介因子順序, 結果變項)
# 每個模型各自取「所需變項都有值」的人次，盡量用到最多資料
MODELS = {
    "A 疲勞中介": (["fatigue_now"], STROOP),
    "B1 心率中介（→ Stroop）": (["hr"], STROOP),
    "B2 心率中介（→ 疲勞）": (["hr"], ["fatigue_now"]),
    "C 心率 → 疲勞 序列中介": (["hr", "fatigue_now"], STROOP),
}


def person_sessions() -> pd.DataFrame:
    co2 = D.load_all_co2()
    ac, switch = L.ac_lookup(co2)
    _, bi = D.load_code_table()
    hc = dict(zip(bi.block, bi.headcount))
    pl = L.person_level(co2, ac, hc)
    sleep = pl["fatigue"].groupby(["student_id", "block"])[SLEEP].mean().reset_index()
    wl = L.window_level(co2, ac, switch, hc, sleep)
    key = ["student", "block"]
    st = pl["stroop"].assign(co2=lambda d: d.co2h * 100)[key + STROOP + IVS + CTRL]
    fa = pl["fatigue"].assign(co2=lambda d: d.co2h * 100)
    fa = fa.groupby(key, as_index=False).agg(fatigue_now=("fatigue_now", "mean"), sleep_h=("sleep_h", "mean"),
                                             sleep_q=("sleep_q", "mean"), co2_f=("co2", "mean"),
                                             temp_f=("temp", "mean"), rh_f=("rh", "mean"),
                                             headcount_f=("headcount", "first"), ac_f=("ac", "first"))
    hr = wl["hr"].groupby(key, as_index=False).hr.mean()
    hrv = wl["hrv"].groupby(key, as_index=False).log_rmssd.mean()
    d = st.merge(fa, on=key, how="outer").merge(hr, on=key, how="outer").merge(hrv, on=key, how="outer")
    for v in IVS + CTRL:   # 沒有 Stroop 的人次用疲勞量表時的環境值
        d[v] = d[v].fillna(d[f"{v}_f"])
    return d.drop(columns=[c for c in d.columns if c.endswith("_f")])


def equations(meds: list[str], outs: list[str]) -> dict[str, list[str]]:
    """每個中介與結果各一條方程式：前面的中介因子都指向後面的變項。"""
    eqs = {}
    for i, m in enumerate(meds):
        eqs[m] = meds[:i] + IVS + CTRL + (SLEEP if m == "fatigue_now" else [])
    for y in outs:
        eqs[y] = meds + IVS + CTRL + (SLEEP if y == "fatigue_now" else [])
    return eqs


def fit_all(d: pd.DataFrame, eqs: dict) -> dict[str, pd.Series]:
    out = {}
    for y, xs in eqs.items():
        xs = [x for x in xs if d[x].nunique() > 1]
        m = smf.mixedlm(f"{y} ~ {' + '.join(xs)}", d, groups=d.student)
        r = None
        for method in ("lbfgs", "powell"):
            try:
                r = m.fit(reml=False, method=method)
                break
            except Exception:  # noqa: BLE001
                continue
        out[y] = r
    return out


def indirect_paths(meds, outs):
    """所有「自變量 → … → 結果」經過至少一個中介的路徑。"""
    paths = []
    for x in IVS:
        for y in outs:
            for m in meds:
                paths.append((x, m, y))
            if len(meds) == 2:
                paths.append((x, meds[0], meds[1], y))
    return paths


def effect(fits, path) -> float:
    e = 1.0
    for a, b in zip(path[:-1], path[1:]):
        e *= fits[b].fe_params.get(a, np.nan)
    return e


def z(d: pd.DataFrame, cols) -> pd.DataFrame:
    d = d.copy()
    for c in cols:
        if d[c].nunique() > 1:
            d[c] = (d[c] - d[c].mean()) / d[c].std()
    return d


def run(name, d_all, meds, outs, rng):
    eqs = equations(meds, outs)
    need = sorted({v for y, xs in eqs.items() for v in [y] + xs})
    d = d_all.dropna(subset=need).reset_index(drop=True)
    d = z(d, [c for c in need if c not in ("ac",)])
    fits = fit_all(d, eqs)
    paths = indirect_paths(meds, outs)
    students = d.student.unique()
    groups = {s: d[d.student == s] for s in students}
    boot_coef, boot_ind = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(students, len(students), replace=True)
        bd = pd.concat([groups[s].assign(student=f"{s}_{i}") for i, s in enumerate(pick)], ignore_index=True)
        try:
            bf = fit_all(bd, eqs)
            boot_coef.append({f"{x}->{y}": bf[y].fe_params.get(x, np.nan) for y, xs in eqs.items() for x in xs})
            boot_ind.append({" → ".join(p): effect(bf, p) for p in paths})
        except Exception:  # noqa: BLE001
            continue
    bc, bi = pd.DataFrame(boot_coef), pd.DataFrame(boot_ind)
    coef = []
    for y, xs in eqs.items():
        r = fits[y]
        for x in xs:
            if x not in r.fe_params:
                continue
            v = bc[f"{x}->{y}"].dropna()
            coef.append({"模型": name, "從": LAB.get(x, x), "到": LAB[y], "β": r.fe_params[x], "p": r.pvalues[x],
                         "bootstrap 下限": v.quantile(0.025), "bootstrap 上限": v.quantile(0.975),
                         "角色": "控制" if x in CTRL + SLEEP else "路徑", "from": x, "to": y})
    ind = []
    for p in paths:
        k = " → ".join(p)
        v = bi[k].dropna()
        ind.append({"模型": name, "路徑": " → ".join(LAB[s] for s in p), "間接效果 β": effect(fits, p),
                    "bootstrap 下限": v.quantile(0.025), "bootstrap 上限": v.quantile(0.975),
                    "bootstrap p": min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())),
                    "顯著": bool(v.quantile(0.025) > 0 or v.quantile(0.975) < 0)})
    info = {"模型": name, "人次": len(d), "學生": d.student.nunique(), "bootstrap 成功次數": len(bc)}
    return pd.DataFrame(coef), pd.DataFrame(ind), info


def feasibility(d: pd.DataFrame) -> pd.DataFrame:
    """每一種中介組合，同一人次所有變項都有資料的筆數。"""
    rows = []
    combos = [("環境 → 疲勞 → Stroop", ["fatigue_now", "rt_mean"] + SLEEP),
              ("環境 → 心率 → Stroop", ["hr", "rt_mean"]),
              ("環境 → 心率 → 疲勞", ["hr", "fatigue_now"] + SLEEP),
              ("環境 → 心率 → 疲勞 → Stroop", ["hr", "fatigue_now", "rt_mean"] + SLEEP),
              ("環境 → 心率變異度 → Stroop", ["log_rmssd", "rt_mean"]),
              ("環境 → 心率變異度 → 疲勞", ["log_rmssd", "fatigue_now"] + SLEEP),
              ("環境 → 疲勞 → 心率變異度", ["fatigue_now", "log_rmssd"] + SLEEP)]
    for lab, cols in combos:
        x = d.dropna(subset=cols + IVS + CTRL)
        n, k = len(x), x.student.nunique()
        rows.append({"路徑": lab, "人次": n, "學生": k,
                     "判斷": "可做" if n >= 70 and k >= 10 else "可做（樣本偏小，只能偵測較大的效果）"
                     if n >= 40 and k >= 8 else "樣本不足，不做"})
    return pd.DataFrame(rows)


def run_one(k: int):
    """只跑第 k 個模型（bootstrap 很慢，四個模型可以分開平行跑）。"""
    name, (meds, outs) = list(MODELS.items())[k]
    d = person_sessions()
    c, i, info = run(name, d, meds, outs, np.random.default_rng(k))
    part = PARTS
    part.mkdir(parents=True, exist_ok=True)
    c.to_csv(part / f"coef_{k}.csv", index=False, encoding="utf-8-sig")
    i.to_csv(part / f"ind_{k}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([info]).to_csv(part / f"info_{k}.csv", index=False, encoding="utf-8-sig")
    print(info, flush=True)


def main():
    d = person_sessions()
    feas = feasibility(d)
    feas.to_csv(OUT / "path_feasibility.csv", index=False, encoding="utf-8-sig")
    print(feas.to_string())
    part = PARTS
    missing = [k for k in range(len(MODELS)) if not (part / f"info_{k}.csv").exists()]
    for k in missing:
        run_one(k)
    rd = lambda p: pd.read_csv(p, encoding="utf-8-sig")   # noqa: E731
    coef = pd.concat([rd(part / f"coef_{k}.csv") for k in range(len(MODELS))])
    ind = pd.concat([rd(part / f"ind_{k}.csv") for k in range(len(MODELS))])
    infos = [rd(part / f"info_{k}.csv").iloc[0].to_dict() for k in range(len(MODELS))]
    coef.round(4).to_csv(OUT / "path_coefficients.csv", index=False, encoding="utf-8-sig")
    ind.round(4).to_csv(OUT / "path_indirect.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(infos).to_csv(OUT / "path_models.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(coef[coef["角色"] == "路徑"].drop(columns=["from", "to"]).round(3).to_string())
    print(ind.round(3).to_string())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_one(int(sys.argv[1]))   # 平行跑：python path_lmm.py 0 / 1 / 2 / 3，再跑 python path_lmm.py 合併
    else:
        main()
