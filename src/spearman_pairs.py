"""兩兩相關：3 個自變量與 6 個依變量（控制變因不列入），個人內 Spearman 等級相關。

做法
  - 分析單位：每位學生每節課一筆。心率、HRV 先取該生該節所有 5 分鐘時段的平均（環境值也取同一批時段的平均）。
  - 環境 × 依變量：用該依變量自己的環境值（疲勞取填答前 3 分鐘、Stroop 取施測當下、心率取同一批時段）。
  - 依變量 × 依變量：同一位學生、同一節課兩者都有資料才配對。
  - 個人內：每位學生先減去自己的平均，再算 Spearman ρ，排除「人與人之間」的差異。
    只測到 1 次的學生減完平均都是 0，不提供資訊，先排除。
  - p 值：學生內置換檢定（在每位學生內部打亂其中一個變項，10,000 次），不假設常態、也考慮了重複測量。
  - 樣本數要求：至少 20 筆配對、5 位學生，否則標示「樣本不足」不做檢定。
  - 多重比較：Benjamini–Hochberg FDR，只在實際做了檢定的配對之間校正（不含自變量彼此之間）。

執行：python src/spearman_pairs.py（需先跑過 lmm_main.py 的資料前處理）
輸出：results/LMM/spearman_pairs.csv、03_兩兩相關.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import data as D
import lmm_main as L

OUT = D.ROOT / "results" / "LMM"
MIN_N, MIN_STUDENTS, N_PERM = 20, 5, 10000
ENV = [("co2", "CO₂"), ("temp", "溫度"), ("rh", "濕度")]
DV = [("fatigue_now", "疲勞程度"), ("rt_mean", "反應時間"), ("accuracy", "正確率"), ("interference", "干擾分數"),
      ("hr", "瞬時心率"), ("log_rmssd", "心率變異度")]
LAB = dict(ENV + DV)


def tables() -> dict[str, pd.DataFrame]:
    """每個依變量一張「學生 × 節次」表：student, block, 依變量, co2, temp, rh。"""
    co2 = D.load_all_co2()
    ac, switch = L.ac_lookup(co2)
    _, bi = D.load_code_table()
    hc = dict(zip(bi.block, bi.headcount))
    pl = L.person_level(co2, ac, hc)
    sleep = pl["fatigue"].groupby(["student_id", "block"])[["sleep_h", "sleep_q"]].mean().reset_index()
    wl = L.window_level(co2, ac, switch, hc, sleep)
    out = {}
    src = {"fatigue_now": pl["fatigue"], "rt_mean": pl["stroop"], "accuracy": pl["stroop"],
           "interference": pl["stroop"], "hr": wl["hr"], "log_rmssd": wl["hrv"]}
    for y, _ in DV:
        d = src[y].assign(co2=lambda t: t.co2h * 100)
        d = d.groupby(["student", "block"], as_index=False)[[y, "co2", "temp", "rh"]].mean()
        out[y] = d
    return out


def within(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = d.dropna(subset=cols)
    d = d[d.groupby("student").student.transform("size") >= 2].copy()
    for c in cols:
        d[c] = d[c] - d.groupby("student")[c].transform("mean")
    return d


def perm_p(x: np.ndarray, y: np.ndarray, groups: np.ndarray, rho: float, rng) -> float:
    """學生內置換：每位學生內部打亂 y，重算 ρ。"""
    idx = [np.where(groups == g)[0] for g in np.unique(groups)]
    rx = stats.rankdata(x)
    cnt = 0
    for _ in range(N_PERM):
        yy = y.copy()
        for i in idx:
            yy[i] = y[rng.permutation(i)]
        r = np.corrcoef(rx, stats.rankdata(yy))[0, 1]
        cnt += abs(r) >= abs(rho) - 1e-12
    return (cnt + 1) / (N_PERM + 1)


def detectable_rho(n_eff: int) -> float:
    """80% 檢定力、雙尾 α = .05 時能偵測的最小 |ρ|（Fisher z 近似）。"""
    return float(np.tanh((1.96 + 0.8416) / np.sqrt(n_eff - 3))) if n_eff > 3 else np.nan


def pair(d: pd.DataFrame, a: str, b: str, rng) -> dict:
    w = within(d, [a, b])
    n, k = len(w), w.student.nunique()
    row = {"變項 1": LAB[a], "變項 2": LAB[b], "筆數": n, "學生": k}
    if n < MIN_N or k < MIN_STUDENTS:
        return {**row, "備註": f"樣本不足（需 ≥ {MIN_N} 筆、≥ {MIN_STUDENTS} 人）"}
    rho = stats.spearmanr(w[a], w[b]).statistic
    p = perm_p(w[a].values, w[b].values.copy(), w.student.values, rho, rng)
    return {**row, "ρ": rho, "p": p, "可偵測的最小 |ρ|": detectable_rho(n - k)}


def bh(p: pd.Series) -> pd.Series:
    """Benjamini–Hochberg FDR 校正。"""
    q = p.dropna().sort_values()
    m = len(q)
    adj = (q * m / np.arange(1, m + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    return adj.reindex(p.index)


def main():
    rng = np.random.default_rng(0)
    t = tables()
    rows = []
    # 自變量 × 依變量：用依變量自己的環境值
    for y, _ in DV:
        for e, _ in ENV:
            rows.append({"類型": "自變量 × 依變量", **pair(t[y], e, y, rng)})
    # 依變量 × 依變量：同一位學生、同一節課
    for i, (a, _) in enumerate(DV):
        for b, _ in DV[i + 1:]:
            m = t[a][["student", "block", a]].merge(t[b][["student", "block", b]], on=["student", "block"])
            rows.append({"類型": "依變量 × 依變量", **pair(m, a, b, rng)})
    res = pd.DataFrame(rows)
    res["FDR 校正 p"] = bh(res.p)
    res.round(4).to_csv(OUT / "spearman_pairs.csv", index=False, encoding="utf-8-sig")

    # 自變量彼此之間：只印出來，不進報告（用 Stroop 的人次，資料最多）
    iv = []
    for i, (a, _) in enumerate(ENV):
        for b, _ in ENV[i + 1:]:
            r = pair(t["rt_mean"], a, b, rng)
            iv.append(r)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(iv).round(3).to_string())
    print(res.round(3).to_string())
    import lmm_figures
    lmm_figures.spearman_heatmap()


if __name__ == "__main__":
    main()
