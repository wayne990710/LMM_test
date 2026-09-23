"""線性混合效應模型（LMM）與交叉驗證工具。"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")  # 小樣本時 statsmodels 常警告隨機效應變異在邊界上，結果表另外報告


def fit_lmm(formula: str, df: pd.DataFrame, group: str, reml: bool = False):
    """隨機截距 LMM。模型比較（AIC、似然比）必須用 ML（reml=False）。"""
    m = smf.mixedlm(formula, df, groups=df[group])
    for method in ("lbfgs", "powell", "nm"):
        try:
            r = m.fit(reml=reml, method=method)
            if np.isfinite(r.llf):
                return r
        except Exception:  # noqa: BLE001 — 換下一個最佳化方法
            continue
    raise RuntimeError(f"LMM 無法收斂：{formula}")


def aic(r) -> float:
    # statsmodels 的 MixedLM 在 ML 下 aic 屬性可用；保險起見自己算（固定效應 + 隨機截距變異 + 殘差變異）
    k = len(r.fe_params) + 2
    return -2 * r.llf + 2 * k


def nakagawa_r2(r) -> tuple[float, float]:
    """Nakagawa & Schielzeth (2013) 邊際 R²（只看固定效應）與條件 R²（固定 + 隨機）。"""
    X = r.model.exog
    var_f = np.var(X @ r.fe_params.values)
    var_u = float(r.cov_re.iloc[0, 0])
    var_e = float(r.scale)
    tot = var_f + var_u + var_e
    return var_f / tot, (var_f + var_u) / tot


def _predict(r, test: pd.DataFrame, group: str, use_re: bool) -> np.ndarray:
    pred = np.asarray(r.predict(test), dtype=float)
    if use_re:
        re = {k: float(v.iloc[0]) for k, v in r.random_effects.items()}
        pred = pred + test[group].map(re).fillna(0).values
    return pred


def loso_rmse(formula: str, df: pd.DataFrame, group: str, y: str) -> float:
    """留一受試者交叉驗證：新受試者沒有隨機截距可用，只用固定效應預測。"""
    err = []
    for g in df[group].unique():
        tr, te = df[df[group] != g], df[df[group] == g]
        r = fit_lmm(formula, tr, group)
        err.append(te[y].values - _predict(r, te, group, use_re=False))
    e = np.concatenate(err)
    return float(np.sqrt(np.mean(e ** 2)))


def kfold_rmse(formula: str, df: pd.DataFrame, group: str, y: str, k: int = 5, repeats: int = 20) -> tuple[float, float]:
    """重複 5 折交叉驗證（隨機切列）：已見過的受試者加上其隨機截距。回傳 RMSE 與 R²。"""
    rmse, r2 = [], []
    for seed in range(repeats):
        pred = np.full(len(df), np.nan)
        for tr, te in KFold(k, shuffle=True, random_state=seed).split(df):
            r = fit_lmm(formula, df.iloc[tr], group)
            pred[te] = _predict(r, df.iloc[te], group, use_re=True)
        e = df[y].values - pred
        rmse.append(np.sqrt(np.mean(e ** 2)))
        r2.append(1 - np.sum(e ** 2) / np.sum((df[y] - df[y].mean()) ** 2))
    return float(np.mean(rmse)), float(np.mean(r2))


def coef_row(r, term: str) -> dict:
    ci = r.conf_int().loc[term]
    return {"beta": r.fe_params[term], "se": r.bse[term], "ci_low": ci[0], "ci_high": ci[1], "p": r.pvalues[term]}
