"""檢定選擇（母數 vs 無母數）＋效果量（Cohen's d／Hedges' g、等級二系列相關）。

分析單位是「學生」：每位學生在 CO2 ≥ 1000 ppm（環境部室內空氣品質標準）與 < 1000 ppm 的節次各取一個平均，形成配對資料。

步驟
1. 假設檢查：測量尺度、配對差值的常態性（Shapiro–Wilk）、偏態、離群值（中位數 ± 3×MAD）、天花板效應、LMM 殘差常態性。
2. 決策：
   - 「必須用無母數」：順序尺度、天花板 > 30%、差值 Shapiro p < .10、有離群值、|偏態| > 1 任一成立。
   - 「建議用無母數」：上述都沒有，但配對人數 < 15（常態檢定在此樣本下檢定力不足，無法證明常態）。
3. 主要檢定：Wilcoxon 符號等級（精確 p），主要家族以 Holm 校正；配對 t 檢定只列出來對照。
4. 效果量：
   - Cohen's d_z = 平均差 ÷ 差值標準差；d_av = 平均差 ÷ 兩條件標準差的平均（Lakens, 2013）。
   - 兩者都乘上 Hedges 小樣本校正 J = 1 − 3 / (4·df − 1)，df = n − 1，得到 g_z、g_av。
   - 無母數效果量：配對等級二系列相關 r_rb（Kerby, 2014）。
   - 95% CI：以學生為單位 bootstrap 5000 次（百分位法）。
5. 連續暴露的效果量：LMM 斜率換算成「每 1000 ppm 的標準化差異」d_1000 = 10·b ÷ 個人內殘差標準差。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import data as D
import models as M

RES = D.ROOT / "results"
THRESH = 1000
N_BOOT = 5000

# (代號, 名稱, 尺度, 家族, LMM 共變項, 上限值)
OUTCOMES = [
    # 睡眠時數／品質只在疲勞模型當控制變項
    ("fatigue_now", "自覺疲勞（1–7）", "順序（單題 Likert）", "主要", "sleep_h + sleep_q", 7),
    ("rt_mean", "Stroop 反應時間 (ms)", "比率（連續）", "主要", "", None),
    ("interference", "Stroop 干擾分數 (ms)", "等距（差異分數）", "主要", "", None),
    ("accuracy", "Stroop 正確率", "比例（0–1）", "主要", "", 1.0),
    ("hr", "心率，每節平均 (bpm)", "比率（連續）", "主要", "", None),
    ("log_rmssd", "HRV log(RMSSD)，每節平均", "連續（對數）", "主要", "bad", None),
    ("fss", "FSS 過去 24 小時（負對照）", "順序（Likert 平均）", "負對照", "sleep_h + sleep_q", 7),
]


# ---------------------------------------------------------------- 資料
def _person_device_map() -> pd.DataFrame:
    dm, _ = D.load_code_table()
    dm["dev"] = dm.device.map(D.device_column)
    dm = dm.dropna(subset=["dev"])
    dm["wearer"] = dm.dev + " " + dm.block
    return dm


def load_long() -> dict[str, pd.DataFrame]:
    """每個結果變項一張長表：student, block, co2, y（以及 LMM 需要的共變項）。"""
    pseudo = None
    ps = pd.read_csv(D.PRIVATE_DIR / "stroop_person_sessions.csv").dropna(subset=["co2_Wa1", "co2_Wa2"])
    fa = pd.read_csv(D.PRIVATE_DIR / "fatigue_with_co2.csv")
    fa = fa[fa.attention_ok].dropna(subset=["co2_Wa1", "co2_Wa2"])
    pseudo = dict(zip(ps.student_id.astype(str), ps.student))
    out = {}
    for y in ("rt_mean", "interference", "accuracy"):
        out[y] = ps.assign(y=ps[y])
    for y in ("fatigue_now", "fss"):
        out[y] = fa.assign(y=fa[y])
    dm = _person_device_map()
    # 生理：CO2 用「每節固定來源」（co2_Sess），只有一臺有資料的節次也能納入；欄名沿用 co2_Avg 方便後續共用
    hw = pd.read_csv(D.PRIVATE_DIR / "hr_5min_windows.csv").dropna(subset=["co2_Sess"])
    hp = hw.merge(dm[["wearer", "student_id"]], on="wearer")
    hp = hp.groupby(["student_id", "session"]).agg(y=("hr", "mean"), co2_Avg=("co2_Sess", "mean")).reset_index()
    out["hr"] = hp.rename(columns={"session": "block"})
    rw = pd.read_csv(D.PRIVATE_DIR / "rmssd_5min_windows.csv").dropna(subset=["co2_Sess"])
    rw = rw[rw.clean >= D.HRV_MIN_CLEAN]
    rp = rw.merge(dm[["wearer", "student_id"]], on="wearer")
    rp = rp.groupby(["student_id", "session"]).agg(y=("log_rmssd", "mean"), bad=("bad", "mean"),
                                                    co2_Avg=("co2_Sess", "mean")).reset_index()
    out["log_rmssd"] = rp.rename(columns={"session": "block"})
    for k, d in out.items():
        d = d.copy()
        d["student_id"] = d.student_id.astype(str)
        d["student"] = d.student_id.map(pseudo).fillna(d.student_id.map(lambda s: f"X{hash(s) % 100:02d}"))
        d["co2h"] = d.co2_Avg / 100
        out[k] = d.reset_index(drop=True)
    return out


def paired(d: pd.DataFrame) -> pd.DataFrame:
    """每位學生：低 CO2 平均、高 CO2 平均（兩種條件都有資料者）。"""
    m = d.assign(high=d.co2_Avg >= THRESH).groupby(["student", "high"]).y.mean().unstack()
    if not {True, False} <= set(m.columns):
        return pd.DataFrame(columns=["low", "high"])
    return m.dropna().rename(columns={False: "low", True: "high"})[["low", "high"]]


# ---------------------------------------------------------------- 效果量
def hedges_j(df: int) -> float:
    return 1 - 3 / (4 * df - 1) if df > 1 else np.nan


def d_family(low: np.ndarray, high: np.ndarray) -> dict:
    diff = high - low
    n = len(diff)
    sd_d = diff.std(ddof=1)
    sd_av = (low.std(ddof=1) + high.std(ddof=1)) / 2
    j = hedges_j(n - 1)
    dz = diff.mean() / sd_d if sd_d > 0 else np.nan
    dav = diff.mean() / sd_av if sd_av > 0 else np.nan
    nz = diff[diff != 0]
    r = stats.rankdata(np.abs(nz))
    rrb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum() if len(nz) else np.nan
    return {"d_z": dz, "g_z": dz * j, "d_av": dav, "g_av": dav * j, "r_rb": rrb}


def boot_ci(low, high, n_boot=N_BOOT, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(low)
    rec = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        rec.append(d_family(low[i], high[i]))
    b = pd.DataFrame(rec)
    return {f"{k}_ci_low": b[k].quantile(0.025) for k in ("g_z", "g_av", "r_rb")} | \
           {f"{k}_ci_high": b[k].quantile(0.975) for k in ("g_z", "g_av", "r_rb")}


def magnitude(g: float) -> str:
    a = abs(g)
    return "微小" if a < 0.2 else "小" if a < 0.5 else "中" if a < 0.8 else "大"


# ---------------------------------------------------------------- 主流程
def analyse() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = load_long()
    checks, effects, qq = [], [], {}
    for key, name, scale, family, cov, ceiling in OUTCOMES:
        d = data[key]
        pr = paired(d)
        low, high = pr["low"].values.astype(float), pr["high"].values.astype(float)
        n = len(pr)
        diff = high - low
        sw_p = stats.shapiro(diff).pvalue if n >= 3 else np.nan
        skew = stats.skew(diff, bias=False) if n >= 3 else np.nan
        mad = stats.median_abs_deviation(diff, scale="normal") if n >= 3 else np.nan
        outliers = int(np.sum(np.abs(diff - np.median(diff)) > 3 * mad)) if mad and mad > 0 else 0
        ceil_pct = float((d.y >= ceiling).mean() * 100) if ceiling is not None else 0.0
        try:
            r = M.fit_lmm(f"y ~ co2h{' + ' + cov if cov else ''}", d, "student")
            resid_p = stats.shapiro(np.asarray(r.resid)[:5000]).pvalue
            d1000 = 10 * r.fe_params["co2h"] / np.sqrt(r.scale)
            ci = r.conf_int().loc["co2h"].values * 10 / np.sqrt(r.scale)
        except Exception:  # noqa: BLE001
            resid_p, d1000, ci = np.nan, np.nan, (np.nan, np.nan)
        reasons = []
        if scale.startswith("順序"):
            reasons.append("順序尺度")
        if ceil_pct > 30:
            reasons.append(f"天花板 {ceil_pct:.0f}%")
        if sw_p < 0.10:
            reasons.append(f"差值非常態（SW p = {sw_p:.3f}）")
        if outliers:
            reasons.append(f"{outliers} 個離群值")
        if abs(skew) > 1:
            reasons.append(f"偏態 {skew:+.2f}")
        level = "必須" if reasons else "建議"
        if not reasons:
            reasons.append(f"配對 n = {n} < 15，常態檢定檢定力不足")
        checks.append({"key": key, "outcome": name, "family": family, "scale": scale, "n_students": n,
                       "n_obs": len(d), "shapiro_diff_p": sw_p, "skew_diff": skew, "outliers": outliers,
                       "ceiling_pct": ceil_pct, "shapiro_lmm_resid_p": resid_p,
                       "decision": f"{level}用無母數", "reasons": "；".join(reasons)})
        qq[name] = diff
        fam = d_family(low, high) if n >= 3 else {}
        e = {"key": key, "outcome": name, "family": family, "n_students": n,
             "median_low": np.median(low) if n else np.nan, "median_high": np.median(high) if n else np.nan,
             "mean_diff": diff.mean() if n else np.nan, "hl_diff": np.nan,
             "n_up": int((diff > 0).sum()), "n_down": int((diff < 0).sum()),
             "wilcoxon_p": stats.wilcoxon(diff[diff != 0]).pvalue if (diff != 0).sum() >= 2 else np.nan,
             "paired_t_p": stats.ttest_rel(high, low).pvalue if n >= 2 else np.nan,
             **fam, "d_1000_lmm": d1000, "d_1000_ci_low": ci[0], "d_1000_ci_high": ci[1]}
        if n >= 3:
            i, j = np.triu_indices(n)
            e["hl_diff"] = float(np.median((diff[i] + diff[j]) / 2))
            e.update(boot_ci(low, high))
            e["magnitude_g_av"] = magnitude(e["g_av"])
        effects.append(e)
    eff = pd.DataFrame(effects)
    # Holm 校正：只在「主要」家族內
    prim = eff.family == "主要"
    p = eff.loc[prim, "wilcoxon_p"].values
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adj[idx] = running
    eff.loc[prim, "wilcoxon_p_holm"] = adj
    return pd.DataFrame(checks), eff, qq


def env_checks() -> pd.DataFrame:
    """環境變項（節次層級）的分配：決定相關用 Pearson 還是 Spearman。"""
    ps = pd.read_csv(D.PRIVATE_DIR / "stroop_person_sessions.csv").dropna(subset=["co2_Wa1", "co2_Wa2"])
    s = ps.groupby("block")[["co2_Avg", "temp_Avg", "rh_Avg"]].mean()
    rows = []
    for c, lab in (("co2_Avg", "CO₂"), ("temp_Avg", "溫度"), ("rh_Avg", "濕度")):
        v = s[c].values
        rows.append({"variable": lab, "n_sessions": len(v), "shapiro_p": stats.shapiro(v).pvalue,
                     "skew": stats.skew(v, bias=False), "min": v.min(), "max": v.max()})
    return pd.DataFrame(rows)


def main():
    checks, eff, qq = analyse()
    env = env_checks()
    checks.round(4).to_csv(RES / "assumption_checks.csv", index=False, encoding="utf-8-sig")
    eff.round(4).to_csv(RES / "effect_sizes.csv", index=False, encoding="utf-8-sig")
    env.round(4).to_csv(RES / "environment_distribution.csv", index=False, encoding="utf-8-sig")
    import figures as F
    F.assumption_qq(qq, checks)
    F.effect_size_forest(eff)
    pd.set_option("display.width", 250)
    print(checks.drop(columns=["key"]).round(3).to_string())
    print(eff.round(3).to_string())
    print(env.round(3).to_string())


if __name__ == "__main__":
    main()
