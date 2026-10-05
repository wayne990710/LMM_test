"""無母數分析：以「學生」為分析單位（12 人、重複測量），不假設常態分配。

三種方法互相印證：
1. 個人內 Spearman：每位學生算一個 CO2 與結果的等級相關 ρ，再用 Wilcoxon 符號等級檢定 ρ 的中位數是否為 0。
2. 高／低 CO2 配對：每位學生在 CO2 ≥ 門檻與 < 門檻時的平均各一個，Wilcoxon 配對檢定 + Hodges–Lehmann 中位差。
3. 節次置換檢定：同一節課所有人共用一個 CO2，所以整組打亂「各節課的 CO2」，統計量 = 個人內 ρ 的平均。
另做 Friedman／Page 趨勢檢定（CO2 三分位：低／中／高）給三個水準都有資料的學生。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MIN_OBS = 4  # 每位學生至少要有幾次觀測才算個人內相關


def _rho_by_group(df: pd.DataFrame, y: str, x: str, group: str) -> pd.Series:
    out = {}
    for g, d in df.groupby(group):
        d = d[[x, y]].dropna()
        if len(d) >= MIN_OBS and d[x].nunique() > 1 and d[y].nunique() > 1:
            out[g] = stats.spearmanr(d[x], d[y]).statistic
    return pd.Series(out, dtype=float)


def _wilcoxon(v: np.ndarray) -> float:
    v = v[~np.isnan(v)]
    v = v[v != 0]
    if len(v) < 2:
        return np.nan
    return float(stats.wilcoxon(v).pvalue)


def hodges_lehmann(d: np.ndarray) -> float:
    """配對差值的 Hodges–Lehmann 估計：所有 Walsh 平均 (d_i + d_j)/2 (i ≤ j) 的中位數。"""
    d = np.asarray(d, float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def within_spearman(df, y, x, group="student") -> dict:
    rho = _rho_by_group(df, y, x, group)
    return {"method": "個人內 Spearman → Wilcoxon", "n_students": len(rho),
            "median_rho": rho.median(), "q1_rho": rho.quantile(0.25), "q3_rho": rho.quantile(0.75),
            "n_positive": int((rho > 0).sum()), "n_negative": int((rho < 0).sum()), "p": _wilcoxon(rho.values),
            "_rhos": rho}


def paired_high_low(df, y, x, threshold, group="student") -> dict:
    d = df[[group, x, y]].dropna().assign(high=lambda t: t[x] >= threshold)
    m = d.groupby([group, "high"])[y].mean().unstack()
    m = m.dropna() if {True, False} <= set(m.columns) else m.iloc[0:0]
    diff = (m[True] - m[False]).values if len(m) else np.array([])
    out = {"method": f"高／低 CO2 配對 Wilcoxon（門檻 {threshold} ppm）", "n_students": len(diff),
           "median_low": float(np.median(m[False])) if len(m) else np.nan,
           "median_high": float(np.median(m[True])) if len(m) else np.nan,
           "hl_diff_high_minus_low": hodges_lehmann(diff) if len(diff) else np.nan,
           "n_positive": int((diff > 0).sum()), "n_negative": int((diff < 0).sum()),
           "p": _wilcoxon(diff)}
    if len(diff) >= 2:  # 配對等級二系列相關：+1 = 所有人高 CO2 時都比較大
        r = stats.rankdata(np.abs(diff[diff != 0]))
        s = np.sign(diff[diff != 0])
        out["rank_biserial"] = float((r[s > 0].sum() - r[s < 0].sum()) / r.sum()) if r.sum() else np.nan
    out["_pairs"] = m
    return out


def session_permutation(df, y, x, group="student", session="block", n_perm=10000, seed=0) -> dict:
    """打亂「節次 → CO2」的對應（同一節課的人一起換），統計量為個人內 ρ 的平均。"""
    d = df[[group, session, x, y]].dropna().copy()
    sess_x = d.groupby(session)[x].mean()
    # 學生之間在同一節課的 CO2 只差在 40 秒區間位置，置換時用節次平均，觀察值也用節次平均才公平
    sessions, vals = sess_x.index.values, sess_x.values
    code = pd.Categorical(d[session], categories=sessions).codes
    # 預先切好每位學生的「節次代碼」與 y 的等級，置換時只需重算 x 的等級
    per = []
    for _, idx in d.groupby(group).indices.items():
        if len(idx) >= MIN_OBS and d[y].values[idx].std() > 0:
            ry = stats.rankdata(d[y].values[idx])
            per.append((code[idx], ry - ry.mean()))

    def stat(xv):
        rs = []
        for c, ry in per:
            rx = stats.rankdata(xv[c])
            rx = rx - rx.mean()
            den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
            if den > 0:
                rs.append((rx * ry).sum() / den)
        return np.mean(rs)

    obs = stat(vals)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(vals)) for _ in range(n_perm)])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"method": f"節次置換檢定（{len(sessions)} 節，{n_perm} 次）", "n_students": d[group].nunique(),
            "mean_rho": obs, "null_2.5%": np.quantile(null, 0.025), "null_97.5%": np.quantile(null, 0.975),
            "p": float(p)}


def tertile_trend(df, y, x, group="student", session="block") -> dict:
    """依節次 CO2 分成低／中／高三組，Friedman（有無差異）與 Page（是否單調上升）。"""
    d = df[[group, session, x, y]].dropna().copy()
    sess_x = d.groupby(session)[x].mean()
    lvl = pd.qcut(sess_x, 3, labels=["low", "mid", "high"])
    d["lvl"] = d[session].map(lvl)
    m = d.groupby([group, "lvl"], observed=False)[y].mean().unstack().dropna()
    out = {"method": f"{x} 三分位 Friedman / Page 趨勢", "n_students": len(m),
           "tertile_cut_ppm": f"{sess_x[lvl == 'low'].max():.0f} / {sess_x[lvl == 'mid'].max():.0f}",
           "median_low": m["low"].median() if len(m) else np.nan,
           "median_mid": m["mid"].median() if len(m) else np.nan,
           "median_high": m["high"].median() if len(m) else np.nan}
    if len(m) >= 3:
        out["p"] = float(stats.friedmanchisquare(m["low"], m["mid"], m["high"]).pvalue)
        out["page_p_increasing"] = float(stats.page_trend_test(m[["low", "mid", "high"]].values).pvalue)
        out["page_p_decreasing"] = float(stats.page_trend_test(m[["high", "mid", "low"]].values).pvalue)
    return out


def run_all(specs, threshold=1000, n_perm=10000) -> tuple[pd.DataFrame, dict]:
    """specs: (結果名稱, df, y, x, group, session)。回傳摘要表與畫圖用的個人資料（不含代碼）。"""
    rows, plot = [], {}
    for lab, df, y, x, group, session in specs:
        ws = within_spearman(df, y, x, group)
        # 1000 ppm 門檻只對 CO2 有意義；溫度、濕度只做等級相關與三分位
        hl = paired_high_low(df, y, x, threshold, group) if x.startswith("co2") else {"_pairs": np.empty((0, 2))}
        pm = session_permutation(df, y, x, group, session, n_perm=n_perm)
        tt = tertile_trend(df, y, x, group, session)
        plot[lab] = {"rho": np.asarray(ws.pop("_rhos")), "pairs": np.asarray(hl.pop("_pairs"))}
        for r in (ws, hl, pm, tt) if x.startswith("co2") else (ws, pm, tt):
            rows.append({"outcome": lab, "exposure": x, "n_obs": len(df), **r})
    return pd.DataFrame(rows), plot
