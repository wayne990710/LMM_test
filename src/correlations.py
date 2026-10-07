"""變項相關矩陣：整體 Spearman 與個人內 Spearman（每人先減去自己的平均，去掉學生之間的差異）。

執行：python src/correlations.py（需先跑過 run_analysis.py 與 path_analysis.py 的資料表）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import data as D
import figures as F
from path_analysis import LABELS, load_data

# 睡眠為控制變項、第幾次施測不列入；心率與 HRV 為主要變項（只有能對應到學生的節次才有，逐對刪除）
VARS = ["co2h", "temp", "rh", "hr", "log_rmssd", "fatigue_now", "rt_mean", "interference", "accuracy"]


def spearman_matrix(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = pd.DataFrame(np.nan, cols, cols)
    p = r.copy()
    for i, a in enumerate(cols):
        for b in cols[i:]:
            d = df[[a, b]].dropna()
            if a == b:
                r.loc[a, b], p.loc[a, b] = 1.0, 0.0
                continue
            rho, pv = stats.spearmanr(d[a], d[b])
            r.loc[a, b] = r.loc[b, a] = rho
            p.loc[a, b] = p.loc[b, a] = pv
    return r, p


def main():
    a, _ = load_data()
    overall_r, overall_p = spearman_matrix(a, VARS)
    # 個人內：每位學生各自減去平均
    within = a.copy()
    within[VARS] = a[VARS] - a.groupby("student")[VARS].transform("mean")
    within_r, within_p = spearman_matrix(within, VARS)
    out = []
    for kind, r, p in (("overall", overall_r, overall_p), ("within_student", within_r, within_p)):
        for i, x in enumerate(VARS):
            for y in VARS[i + 1:]:
                n = len((a if kind == "overall" else within)[[x, y]].dropna())
                out.append({"kind": kind, "var1": x, "var2": y, "n": n, "rho": r.loc[x, y], "p": p.loc[x, y]})
    pd.DataFrame(out).round(4).to_csv(D.ROOT / "results" / "correlation_matrix.csv", index=False,
                                      encoding="utf-8-sig")
    F.correlation_heatmap({"整體 Spearman（含學生之間的差異）": (overall_r, overall_p),
                           "個人內 Spearman（每位學生減去自己的平均）": (within_r, within_p)},
                          [LABELS[v] for v in VARS], n=len(a), students=a.student.nunique(),
                          sub={LABELS[v]: (int(a[v].notna().sum()), int(a.dropna(subset=[v]).student.nunique()))
                               for v in ("hr", "log_rmssd")})
    print(f"n = {len(a)} 人次，{a.student.nunique()} 人；心率 {a.hr.notna().sum()}、HRV {a.log_rmssd.notna().sum()} 人次")


if __name__ == "__main__":
    main()
