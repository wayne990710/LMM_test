"""座位表有沒有用？用兩種方法直接檢驗。

1. 窮舉：把每位學生指派到 Wa1 或 Wa2（2^n 種座位安排全部試一遍），
   以「最近那台」當個人暴露量重跑 LMM，看係數、p 值、AIC 最多能變多少。
2. 模擬：假設真相就是「學生吸到的是最近那台的濃度」，而且 CO2 真的有效應，
   用正確座位表建模是否能比兩台平均（不需座位表）明顯更好（ΔAIC > 2）。
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import models as M


def _fit(df, y, covars, expo):
    d = df.assign(_x=expo / 100)
    r = M.fit_lmm(f"{y} ~ _x + {covars}", d, "student")
    return r.fe_params["_x"], r.pvalues["_x"], M.aic(r)


def exhaustive(df: pd.DataFrame, y: str, covars: str) -> dict:
    students = sorted(df.student.unique())
    idx = df.student.map({s: i for i, s in enumerate(students)}).values
    avg = df[["co2_Wa1", "co2_Wa2"]].mean(axis=1).values
    b_avg, p_avg, aic_avg = _fit(df, y, covars, avg)
    rows = []
    for seat in itertools.product([0, 1], repeat=len(students)):
        s = np.array(seat)[idx]
        expo = np.where(s == 1, df.co2_Wa2.values, df.co2_Wa1.values)
        b, p, a = _fit(df, y, covars, expo)
        rows.append((b, p, a, np.corrcoef(expo, avg)[0, 1], np.abs(expo - avg).max()))
    r = np.array(rows)
    return {"outcome": y, "n": len(df), "students": len(students), "seatings_tested": len(r),
            "beta_avg": b_avg, "p_avg": p_avg,
            "beta_min": r[:, 0].min(), "beta_max": r[:, 0].max(),
            "p_min": r[:, 1].min(), "p_max": r[:, 1].max(),
            "best_delta_aic_vs_avg": r[:, 2].min() - aic_avg, "worst_delta_aic_vs_avg": r[:, 2].max() - aic_avg,
            "min_corr_with_avg": r[:, 3].min(), "max_abs_diff_ppm": r[:, 4].max()}


def simulate(df: pd.DataFrame, y: str, covars: str, betas=(0, 5, 10, 20), nsim=200, seed=1) -> pd.DataFrame:
    """真相 = 最近那台。beta 單位：每 100 ppm 的效應。"""
    rng = np.random.default_rng(seed)
    base = M.fit_lmm(f"{y} ~ {covars}", df, "student")
    fixed = np.asarray(base.predict(df), dtype=float)
    sd_u, sd_e = np.sqrt(float(base.cov_re.iloc[0, 0])), np.sqrt(base.scale)
    students = sorted(df.student.unique())
    avg = df[["co2_Wa1", "co2_Wa2"]].mean(axis=1).values
    out = []
    for beta in betas:
        for _ in range(nsim):
            seat = dict(zip(students, rng.integers(0, 2, len(students))))
            true = np.where(df.student.map(seat) == 1, df.co2_Wa2, df.co2_Wa1)
            u = dict(zip(students, rng.normal(0, sd_u, len(students))))
            ys = fixed + df.student.map(u).values + beta * true / 100 + rng.normal(0, sd_e, len(df))
            d = df.assign(**{y: ys})
            _, _, a_true = _fit(d, y, covars, true)
            b_avg, p_avg, a_avg = _fit(d, y, covars, avg)
            out.append({"true_beta_per_100ppm": beta, "delta_aic_avg_minus_true": a_avg - a_true,
                        "avg_detects_effect": p_avg < 0.05, "avg_beta": b_avg})
    s = pd.DataFrame(out).groupby("true_beta_per_100ppm").agg(
        mean_delta_aic=("delta_aic_avg_minus_true", "mean"),
        pct_seating_wins_by_2=("delta_aic_avg_minus_true", lambda v: (v > 2).mean() * 100),
        pct_avg_detects_effect=("avg_detects_effect", lambda v: v.mean() * 100),
        mean_avg_beta=("avg_beta", "mean")).reset_index()
    s.insert(0, "outcome", y)
    return s
