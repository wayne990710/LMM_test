"""補齊值得做的圖表，並把全部圖表依研究流程整理成一份 PDF 圖表集（給指導教授）。

執行：python src/figure_book.py（需先跑過 run_analysis.py）
輸出：results/figures/*.png（300 dpi）、results/figures/svg/*.svg，以及依序編號的 results/圖表/
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestPower

import ac_status as AC
import data as D
import figures as F
import prediction as P
from figures import COL, INK, MUTED, _save

R = D.ROOT / "results"
FIG = R / "figures"
STATUS = ["#2f8a4c", "#d4a12a", "#e07b2e", "#c2413a"]   # <1000、1000–1500、1500–2000、>2000 ppm


# ---------------------------------------------------------------- 新圖
def data_coverage():
    """每個場次有哪些資料：格內數字為人數或裝置數，顏色深淺代表相對完整度。"""
    trials = D.load_stroop_trials()
    fat = pd.read_csv(D.PRIVATE_DIR / "fatigue_with_co2.csv")
    ss = pd.read_csv(R / "session_summary.csv").set_index("block")
    hw = pd.read_csv(D.PRIVATE_DIR / "hr_5min_windows.csv")
    q = D.ecg_quality()
    q = q[q.quality != "error"]
    blocks = sorted(set(trials.block))
    rows = {
        "Stroop（人）": trials.groupby("block").Student_ID.nunique(),
        "疲勞量表（份，已排除未通過檢核）": fat[fat.attention_ok].groupby("block").size(),
        "CO₂ Wa1（施測時有資料）": ss.co2_Wa1.notna().astype(int),
        "CO₂ Wa2（施測時有資料）": ss.co2_Wa2.notna().astype(int),
        "手環（顆）": hw[hw.device_type == "polar"].groupby("session").device.nunique(),
        "心電貼片（收錄顆數）": q.groupby("session").label.nunique(),
        "心電貼片（有可用心率時段）": hw[hw.device_type == "ecg"].groupby("session").device.nunique(),
    }
    mat = pd.DataFrame({k: v.reindex(blocks) for k, v in rows.items()}).T.fillna(0)
    norm = mat.div(mat.max(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.imshow(norm.values, cmap=plt.cm.colors.LinearSegmentedColormap.from_list("c", ["#f1efe9", COL["Avg"]]),
              aspect="auto", vmin=0, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            txt = ("✓" if v else "—") if "CO₂" in mat.index[i] else (f"{int(v)}" if v else "—")
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="#fcfcfb" if norm.values[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(blocks)), blocks, rotation=35, ha="right")
    ax.set_yticks(range(len(mat.index)), mat.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("各場次資料完整度（數字＝人數或裝置數；— ＝沒有資料）", color=INK)
    fig.tight_layout()
    _save(fig, "data_coverage.png")


def co2_exposure():
    """每節課的 CO₂ 分級時間比例：兩臺感測器的讀值合併計算；該節讀值 < 30 筆的感測器不計入。"""
    co2 = D.load_all_co2()
    parts = []
    for s, c in co2.items():
        c = c[c.time >= "2026-09-21"].copy()
        c["session"] = D._session_id(c.time)
        n = c.groupby("session").co2.transform("size")
        parts.append(c[n >= 30])
    allc = pd.concat(parts)
    bins = [0, 1000, 1500, 2000, np.inf]
    labs = ["< 1000", "1000–1500", "1500–2000", "> 2000"]
    allc["band"] = pd.cut(allc.co2, bins, labels=labs, right=False)
    seg = allc.groupby(["session", "band"], observed=False).size().unstack().fillna(0)
    seg = seg.div(seg.sum(axis=1), axis=0) * 100
    ac = AC.load()
    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(seg))
    left = np.zeros(len(seg))
    for col, c in zip(labs, STATUS):
        ax.barh(y, seg[col], left=left, color=c, height=0.62, label=f"{col} ppm", edgecolor="#fcfcfb", lw=1.5)
        left += seg[col].values
    ax.set_yticks(y, [f"{s}　{ac.get(s, '') or ''}" for s in seg.index])
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="該節課收錄時間的比例 (%)", title="各節課 CO₂ 濃度分級")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "co2_exposure.png")
    seg.round(1).to_csv(R / "co2_exposure_pooled.csv", encoding="utf-8-sig")


def ac_inference():
    """冷氣狀態判定的證據：每節前 1/3 與後 1/3 的 CO₂ 上升速度，以及最高濃度。"""
    # reset_index：去掉沒有 CO₂ 的 09-17 後重新編號，否則資料列會和 y 軸標籤錯開一列
    d = pd.read_csv(D.ROOT / "data" / "ac_status.csv", encoding="utf-8-sig").dropna(subset=["early_slope"])
    d = d.reset_index(drop=True)
    col = {"冷氣": COL["Wa2"], "後半冷氣": "#d4a12a", "無冷氣": COL["Wa1"]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(d))
    for i, r in d.iterrows():
        c = col.get(r.status, MUTED)
        inferred = r.source.startswith("由")
        a.plot([r.early_slope, r.late_slope], [i, i], color=c, lw=2, alpha=0.6)
        a.scatter(r.early_slope, i, s=70, marker="o", color="#fcfcfb" if inferred else c, edgecolors=c, lw=2, zorder=3)
        a.scatter(r.late_slope, i, s=90, marker=">", color="#fcfcfb" if inferred else c, edgecolors=c, lw=2, zorder=3)
        b.barh(i, r.co2_max, color=c, height=0.6, alpha=0.45 if inferred else 0.9, edgecolor=c, lw=1.5)
    a.axvline(AC.RISING, color=INK, ls="--", lw=1.2)
    a.text(AC.RISING + 0.5, len(d) - 0.4, f"{AC.RISING:g} ppm／分（判定門檻）", fontsize=9, color=INK, va="top")
    a.axvline(0, color=MUTED, lw=0.8)
    a.set(xlabel="CO₂ 上升速度（ppm／分）　●＝前 1/3　▶＝後 1/3", title="CO₂ 累積速度")
    b.axvline(1000, color=MUTED, ls="--", lw=1)
    b.axvline(2000, color=INK, ls="--", lw=1)
    b.set(xlabel="最高 CO₂ (ppm)", title="最高濃度（虛線：1000、2000 ppm）")
    a.set_yticks(y, [f"{r.session}　{r.label}" for _, r in d.iterrows()])
    a.invert_yaxis()
    for ax in (a, b):
        ax.grid(axis="y", visible=False)
    h = [plt.Line2D([], [], color=c, lw=6, label=k) for k, c in col.items()] +         [plt.Line2D([], [], marker="s", ls="", color=MUTED, ms=9, label="研究代碼對照表"),
         plt.Line2D([], [], marker="s", ls="", color="#fcfcfb", markeredgecolor=MUTED, mew=2, ms=9, label="由數據推定")]
    fig.legend(handles=h, ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(fig, "ac_inference.png")


def ecg_quality_map():
    """每顆貼片每節課：可用於心率的 5 分鐘時段數（乾淨拍 ≥ 80%）／總時段數，以及可算出 HRV 的時段數
    （乾淨拍 ≥ 80%，且 Malik 篩選後仍有 ≥ 60 組相鄰拍差）。"""
    q = D.ecg_window_quality()
    q = q[q.beats >= 150]
    q["session"] = D._session_id(q.win)
    q["rec"] = q.device + " " + q.session
    q = q[q.rec.isin(D.usable_ecg_recordings())]
    g = q.groupby(["device", "session"]).agg(total=("clean", "size"),
                                             hr=("clean", lambda c: int((c >= D.HR_MIN_CLEAN).sum())),
                                             ).reset_index()
    rw = pd.read_csv(D.PRIVATE_DIR / "rmssd_5min_windows.csv")
    rw = rw[rw.clean >= D.HRV_MIN_CLEAN].groupby(["device", "session"]).size().rename("hrv")
    g = g.merge(rw, left_on=["device", "session"], right_index=True, how="left").fillna({"hrv": 0})
    g["hrv"] = g.hrv.astype(int)
    g["frac"] = g.hr / g.total
    frac = g.pivot(index="device", columns="session", values="frac")
    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("q", ["#e9b2ad", "#f3e3b5", "#a8d8bf"])
    ax.imshow(frac.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    for _, r in g.iterrows():
        i, j = frac.index.get_loc(r.device), frac.columns.get_loc(r.session)
        ax.text(j, i - 0.12, f"{r.hr}/{r.total}", ha="center", va="center", fontsize=9.5, color=INK)
        ax.text(j, i + 0.22, f"HRV {r.hrv}", ha="center", va="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(frac.shape[1]), frac.columns, rotation=35, ha="right")
    ax.set_yticks(range(frac.shape[0]), frac.index)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"心電貼片可用時段（上：可用於心率／總時段，乾淨拍 ≥ 80%；下：可算 HRV，≥ 80% + Malik 20%）　"
                 f"心率 {int(g.hr.sum())}/{int(g.total.sum())}、HRV {int(g.hrv.sum())} 個時段", color=INK, fontsize=11)
    fig.tight_layout()
    _save(fig, "ecg_quality.png")


def stroop_conditions():
    """測驗效度檢核：不一致題應比中性題慢（Stroop 效應）。"""
    t = D.load_stroop_trials()
    t = t[(t.correct == 1) & t.rt.between(200, 3000)]
    m = t.groupby(["Student_ID", "condition"]).rt.mean().unstack()[["neutral", "congruent", "incongruent"]]
    lab = {"neutral": "中性", "congruent": "一致", "incongruent": "不一致"}
    fr = stats.friedmanchisquare(*[m[c] for c in m.columns]).pvalue
    wi = stats.wilcoxon(m.incongruent, m.neutral).pvalue
    by = t.groupby(["block", "Student_ID", "condition"]).rt.mean().groupby(["block", "condition"]).median().unstack()
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.5]})
    for _, r in m.iterrows():
        a.plot(range(3), r.values, color=MUTED, alpha=0.35, lw=1)
    a.plot(range(3), m.median().values, color=INK, lw=3, marker="o", ms=8)
    a.set_xticks(range(3), [lab[c] for c in m.columns])
    a.set(ylabel="平均反應時間 (ms)", title=f"每線一位學生（全部場次平均）\nFriedman p = {fr:.3f}；不一致 vs 中性 Wilcoxon p = {wi:.3f}")
    for c, col in zip(["neutral", "congruent", "incongruent"], [COL["Avg"], COL["Wa1"], COL["Wa2"]]):
        b.plot(by.index, by[c], marker="o", color=col, label=lab[c])
    b.legend(frameon=False)
    b.set(ylabel="各場次學生中位數 (ms)", title="各場次三種情境的反應時間")
    b.tick_params(axis="x", rotation=35)
    fig.suptitle("Stroop 測驗效度檢核：三種情境的反應時間", color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, "stroop_conditions.png")


def env_collinearity():
    ss = pd.read_csv(R / "session_summary.csv")
    ss["co2"] = ss[["co2_Wa1", "co2_Wa2"]].mean(axis=1)
    ss["temp"] = ss[["temp_Wa1", "temp_Wa2"]].mean(axis=1)
    ss["rh"] = ss[["rh_Wa1", "rh_Wa2"]].mean(axis=1)
    ss = ss.dropna(subset=["co2"])
    lab_ac = AC.load()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, k, lab in [(axes[0], "temp", "溫度 (°C)"), (axes[1], "rh", "相對濕度 (%)")]:
        for _, r in ss.iterrows():
            col = COL["Wa2"] if r.block.endswith("PM") else COL["Wa1"]
            st = lab_ac.get(r.block, "")
            ac = st.startswith(("冷氣", "後半冷氣"))   # 兩種「後半冷氣」節次的 Stroop 都在冷氣開啟後施測
            inferred = "推定" in st
            ax.scatter(r.co2, r[k], s=110, marker="s" if ac else "o", zorder=3, lw=2,
                       color="#fcfcfb" if inferred else col, edgecolors=col)
            ax.annotate(r.block, (r.co2, r[k]), textcoords="offset points", xytext=(7, 5), fontsize=8, color=MUTED)
        rho = stats.spearmanr(ss.co2, ss[k]).statistic
        ax.set(xlabel="Stroop 施測時 CO₂（兩臺平均，ppm）", ylabel=lab, title=f"CO₂ vs {lab}：Spearman ρ = {rho:.2f}")
    h = [plt.Line2D([], [], marker="o", ls="", color=COL["Wa1"], ms=9, label="上午"),
         plt.Line2D([], [], marker="o", ls="", color=COL["Wa2"], ms=9, label="下午"),
         plt.Line2D([], [], marker="s", ls="", color=MUTED, ms=9, label="冷氣（研究代碼對照表）"),
         plt.Line2D([], [], marker="s", ls="", color="#fcfcfb", markeredgecolor=MUTED, mew=2, ms=9,
                    label="冷氣（由數據推定）")]
    axes[1].legend(handles=h, frameon=False, loc="best")
    fig.suptitle("環境變項共線性：CO₂ 高的節次同時較涼、較乾（開冷氣、關窗）", color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, "env_collinearity.png")


def temp_rh_adjustment():
    e = pd.read_csv(R / "temp_humidity_adjustment.csv")
    e = e[(e.exposure == "Avg") & e.adjust.isin(["none", "+temp", "+temp+rh"])]
    outs = list(dict.fromkeys(e.outcome))
    lab = {"none": "不調整", "+temp": "+ 溫度", "+temp+rh": "+ 溫度 + 濕度"}
    cols = [INK, COL["Wa1"], COL["Wa2"]]
    fig, axes = plt.subplots(1, len(outs), figsize=(3.6 * len(outs), 4.2))
    for ax, o in zip(axes, outs):
        g = e[e.outcome == o].reset_index(drop=True)
        for i, r in g.iterrows():
            ax.plot([r.ci_low, r.ci_high], [i, i], color=cols[i], lw=2.6, solid_capstyle="round")
            ax.scatter(r.beta, i, color=cols[i], s=60, zorder=3, edgecolors="#fcfcfb")
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(len(g)), [lab[a] for a in g.adjust])
        ax.invert_yaxis()
        ax.set_title(o, fontsize=10)
        ax.set_xlabel("CO₂ 係數（每 100 ppm）", fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.suptitle("加入溫度、濕度後 CO₂ 係數的變化（點＝估計值，線＝95% CI）", color=INK, y=1.03)
    fig.tight_layout()
    _save(fig, "temp_rh_adjustment.png")


def ventilation_fits():
    co2 = D.load_all_co2()
    mb = P.fit_mass_balance(co2)
    u = mb[mb.usable].reset_index(drop=True)
    ncol = 4
    nrow = int(np.ceil(len(u) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False, sharey=True)
    for ax in axes.flat[len(u):]:
        ax.set_visible(False)
    for ax, (_, r) in zip(axes.flat, u.iterrows()):
        c = co2[r.sensor]
        g = c[D._session_id(c.time) == r.session].sort_values("time")
        t = (g.time - g.time.iloc[0]).dt.total_seconds() / 60
        g, t = g[t > 2], t[t > 2] - 2
        g, t = g[t <= r.minutes], t[t <= r.minutes]
        ax.scatter(t, g.co2, s=8, color=COL["Wa1" if r.sensor == "Wa1" else "Wa2"], alpha=0.6)
        tt = np.linspace(0, r.minutes, 100)
        ax.plot(tt, P._rise(tt / 60, r.css, r.ach_per_h, r.c_start), color=INK, lw=2)
        ax.axhline(1000, color=MUTED, ls="--", lw=0.8)
        ax.set_title(f"{r.session} {r.sensor}：λ = {r.ach_per_h:.2f}／h，R² = {r.r2:.2f}", fontsize=9.5)
        ax.set_xlabel("分鐘", fontsize=8.5)
    for row in axes:
        row[0].set_ylabel("CO₂ (ppm)")
    fig.suptitle("質量平衡模型擬合（點＝實測，黑線＝C(t) = Css − (Css − C₀)·e^(−λt)；虛線＝1000 ppm）",
                 color=INK, y=1.01)
    fig.tight_layout()
    _save(fig, "ventilation_fits.png")


def power_curve():
    eff = pd.read_csv(R / "effect_sizes.csv").set_index("key")
    p = TTestPower()
    ns = np.arange(5, 61)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    items = [("fatigue_now", "自覺疲勞", COL["Avg"]), ("accuracy", "Stroop 正確率", COL["Wa2"]),
             ("rt_mean", "Stroop 反應時間", COL["Wa1"]), ("hr", "心率", INK),
             ("log_rmssd", "HRV", "#9b5fc0")]
    for k, lab, c in items:
        if k not in eff.index or not np.isfinite(eff.loc[k, "d_z"]) or eff.loc[k, "d_z"] == 0:
            continue
        dz = abs(eff.loc[k, "d_z"])
        pw = [p.power(effect_size=dz, nobs=n, alpha=0.05) for n in ns]
        n80 = int(np.ceil(p.solve_power(effect_size=dz, alpha=0.05, power=0.8)))
        now = int(eff.loc[k, "n_students"])
        ax.plot(ns, pw, color=c, lw=2.4, label=f"{lab}（|d_z| = {dz:.2f}；80% 需 {n80} 人）")
        ax.scatter(now, p.power(effect_size=dz, nobs=now, alpha=0.05), color=c, s=70, zorder=3, edgecolors="#fcfcfb")
    ax.axhline(0.8, color=MUTED, ls="--", lw=1)
    ax.text(ns[-1], 0.81, "80%", ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.set(xlabel="配對人數", ylabel="檢定力（雙尾 α = .05）", ylim=(0, 1),
           title="檢定力曲線：以目前觀察到的效果量推估需要的人數（點＝目前人數）")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "power_curve.png")


# ---------------------------------------------------------------- 圖表集
# (檔名, 標題, 這張圖呈現什麼, 重點)
BOOK = [
    ("A. 研究總覽", None, None, None),
    ("ucf_map", "論證路線圖（UCF）", "中心論點、六個分析步驟與三個研究問題的對應。",
     "全部圖表都服務同一個論點：CO₂ 升高使自覺疲勞上升，但反應速度不變；通風需約 3 倍換氣量。"),
    ("data_coverage", "各場次資料完整度", "每個場次有幾位學生、幾份問卷、哪臺感測器、幾顆手環與可用貼片。",
     "09-17 沒有 CO₂；09-30 下午、10-01、10-02 只有一臺感測器；可用的貼片數明顯少於收錄數。"),
    ("B. 環境暴露", None, None, None),
    ("co2_timeline", "CO₂ 時間序列", "每節課兩臺感測器的 CO₂，灰底為 Stroop 施測時段。",
     "下午與開冷氣的節次 CO₂ 持續上升，多數節次在施測時已超過 1000 ppm。"),
    ("co2_exposure", "CO₂ 濃度分級時間比例", "每節課的收錄時間中，落在各濃度區間的比例。",
     "09-21 下午幾乎全程 > 2000 ppm；09-22、09-23 上午幾乎全程 < 1000 ppm。"),
    ("sensor_agreement", "兩臺感測器一致性", "同一時間點兩臺讀值的散布圖與 Bland–Altman 圖。",
     "兩臺高度相關，但有系統性差距，且 09-24 起方向反轉，需並排校正。"),
    ("env_collinearity", "環境變項共線性", "各節課的 CO₂ 與溫度、濕度，依上午／下午與冷氣註記標記。",
     "CO₂ 高的節次同時較涼、較乾：CO₂、溫度、濕度屬於同一個「開冷氣關窗」狀態，是本研究最主要的限制。"),
    ("ac_inference", "冷氣狀態判定", "每節課前段與後段的 CO₂ 上升速度與最高濃度，用來推定沒有註記的節次是否開冷氣。",
     "09-21 下午、09-22 下午、09-23 下午推定為冷氣，09-21 上午推定為後半冷氣，09-22、09-23 上午為無冷氣。"),
    ("C. 測量品質", None, None, None),
    ("ecg_quality", "心電貼片品質", "每顆貼片每節課的品質判定與異常 RR 比例。",
     "只有約一半的貼片節次可用；E2605-02、E2605-05 幾乎每次都是 poor。"),
    ("hrv_conversion", "每秒心率能否換算 HRV", "貼片逐拍 RR 的真 RMSSD 與兩種每秒心率換算值的比較。",
     "手環的平滑心率換算出的 RMSSD 約低估 80%，所以 HRV 只能用貼片的逐拍 RR。"),
    ("stroop_conditions", "Stroop 測驗效度", "中性、一致、不一致三種情境的反應時間。",
     "每位學生的不一致題比中性題慢（配對差值中位數 27 ms、平均 16 ms），但未達顯著（Friedman p = 0.076；Wilcoxon p = 0.18）。每種情境只有 8 題，"
     "干擾效應本身不明顯，這也是干擾分數信度低、不宜作為主要結果的原因；建議增加題數。"),
    ("D. 檢定選擇與相關", None, None, None),
    ("assumption_qq", "常態性檢查", "每個結果變項的「高 − 低 CO₂」配對差值 Q-Q 圖。",
     "8 個結果變項中 7 個必須用無母數，另一個因 n < 15 建議用無母數。"),
    ("correlation_heatmap", "變項相關矩陣", "整體與個人內 Spearman 相關。",
     "個人內 CO₂ 與疲勞 ρ = +0.39；CO₂ 與溫度 ρ = −0.87。"),
    ("E. 主要結果", None, None, None),
    ("fatigue_vs_co2", "CO₂ 與自覺疲勞（場次層級）", "各場次的平均疲勞與 FSS（負對照）。",
     "疲勞隨 CO₂ 上升；問「過去 24 小時」的 FSS 則沒有跟著變。"),
    ("nonparametric", "無母數分析", "個人內 Spearman ρ 與高／低 CO₂ 的配對比較，每點或每線為一位學生。",
     "疲勞的方向一致（11 人中 8 人上升），反應時間沒有一致方向。"),
    ("effect_size_forest", "效果量（Hedges' g）", "各結果變項高 vs 低 CO₂ 的 g 與 r_rb，含學生層級 bootstrap 95% CI。",
     "疲勞 g = +0.53、正確率 g = −0.69（中等）；反應時間小且 CI 含 0；負對照接近 0。"),
    ("stroop_vs_co2", "CO₂ 與 Stroop 反應時間（場次層級）", "各場次平均反應時間對 Wa1、Wa2 的 CO₂。",
     "場次平均反應時間與 CO₂ 沒有一致的關係。"),
    ("temp_rh_adjustment", "溫濕度調整前後的 CO₂ 係數", "各結果變項在不調整、加溫度、加溫度與濕度時的 CO₂ 係數。",
     "疲勞的係數始終為正，但加入濕度後信賴區間變寬並碰到 0。"),
    ("hr_vs_co2", "心率與 CO₂ 的時間變化", "每 5 分鐘的平均心率與 CO₂。",
     "下午 CO₂ 上升時心率常同時下降，但這也可能是坐定後的自然下降，控制時間後效應不穩定。"),
    ("hrv_vs_co2", "CO₂ 與心率變異度（HRV）", "配戴者內 CO₂ 與 log RMSSD 的關係，以及三種篩選門檻下的 LMM 效果。",
     "CO₂ 較高時 RMSSD 略低；主要分析未達顯著，最嚴格門檻下達顯著。"),
    ("model_comparison", "感測器比較（ΔAIC）", "用 Wa1、Wa2、兩臺平均分別建模，加入 CO₂ 後的 AIC 改善。",
     "三種暴露來源之間的差距都小於 2，分不出哪一臺比較好。"),
    ("seating_simulation", "座位表是否有幫助", "假設學生吸到的是最近那臺的濃度，模擬座位表能否改善模型。",
     "在目前的效果量下，座位表幾乎不會改變結論。"),
    ("F. 機制與預測", None, None, None),
    ("path_diagram", "徑路分析", "piecewise SEM，模型 A（CO₂→疲勞→Stroop）與模型 B（加入心率）。",
     "CO₂→疲勞成立（β = +0.36）；疲勞→反應時間不成立，中介鏈不成立；模型 B 樣本太小不解讀。"),
    ("ventilation_fits", "質量平衡擬合", "12 段 CO₂ 上升曲線的實測值與擬合曲線。",
     "擬合都很好（R² ≥ 0.94），換氣率中位數約 1.2 次／小時，作為情境模擬的依據。"),
    ("prediction_simulation", "通風情境模擬", "三種換氣率下 50 分鐘課的 CO₂ 軌跡與下課時的疲勞增量。",
     "換氣率提高到約 3.5 次／小時可把 CO₂ 壓在 1000 ppm，疲勞增量由 +0.49 降到 +0.08 分。"),
    ("power_curve", "檢定力曲線", "以目前效果量推估不同人數下的檢定力。",
     "疲勞要達 80% 檢定力需約 24 人（目前 11 人，檢定力約 45%）；反應時間的效果太小，需 66 人。"),
]


def export_numbered():
    """依研究流程把圖片依序編號，複製到 results/圖表/（PNG 300 dpi 與 SVG）。說明文字另見 圖表說明.md。"""
    import shutil
    out = R / "圖表"
    (out / "svg").mkdir(parents=True, exist_ok=True)
    # 只刪舊圖檔、保留資料夾：資料夾若被檔案總管或 OneDrive 開著，整個刪除會失敗
    for old in [*out.glob("*.png"), *(out / "svg").glob("*.svg")]:
        old.unlink()
    n = 0
    for key, title, _, _ in BOOK:
        if title is None:
            continue
        n += 1
        name = f"{n:02d}_{title.replace('／', '_').replace('/', '_')}"
        shutil.copy(FIG / f"{key}.png", out / f"{name}.png")
        shutil.copy(FIG / "svg" / f"{key}.svg", out / "svg" / f"{name}.svg")
    return out


def main():
    AC.build()
    for f in (data_coverage, co2_exposure, ac_inference, ecg_quality_map, stroop_conditions, env_collinearity,
              temp_rh_adjustment, ventilation_fits, power_curve):
        f()
        print("完成", f.__name__)
    print(export_numbered())


if __name__ == "__main__":
    main()
