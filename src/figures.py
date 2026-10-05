"""圖表：只畫區塊／節次層級的平均，不畫逐人資料點（圖會推上公開 repo）。"""
from __future__ import annotations

import matplotlib
import matplotlib.dates
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as D  # noqa: E402

FIG = D.ROOT / "results" / "figures"
COL = {"Wa1": "#2a78d6", "Wa2": "#eb6834", "Avg": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": ["Microsoft JhengHei", "Microsoft YaHei", "sans-serif"], "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10, "axes.axisbelow": True,
    "svg.fonttype": "none",  # SVG 裡的字保留成文字，可在 Illustrator／Inkscape／Word 直接編輯
})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")  # 300 dpi：放大或印刷都清楚
    (FIG / "svg").mkdir(exist_ok=True)
    fig.savefig(FIG / "svg" / name.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)


def co2_timeline(co2, trials):
    allc = pd.concat(co2.values())
    sessions = sorted(set(D._session_id(allc[allc.time >= "2026-09-21"].time)))
    ncol = 4
    nrow = int(np.ceil(len(sessions) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), sharey=True, squeeze=False)
    for ax in axes.flat[len(sessions):]:
        ax.set_visible(False)
    for ax, sess in zip(axes.flat, sessions):
        for s, c in co2.items():
            g = c[D._session_id(c.time) == sess]
            if len(g):
                ax.plot(g.time, g.co2, color=COL[s], label=s)
        tb = trials[trials.block == sess]
        if len(tb):
            ax.axvspan(tb.time.min(), tb.time.max(), color=MUTED, alpha=0.15, lw=0, label="Stroop 施測")
        for th, lab in [(1000, "1000"), (2000, "2000（計畫書中止門檻）")]:
            ax.axhline(th, color=MUTED, ls="--", lw=1)
        ax.set_title(sess, color=INK)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", rotation=0)
    for row in axes:
        row[0].set_ylabel("CO₂ (ppm)")
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.suptitle("兩台感測器的 CO₂ 時間序列（灰底 = Stroop 施測時段；虛線 = 1000 / 2000 ppm）", y=1.06, color=INK)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.025), ncol=3, frameon=False)
    fig.tight_layout()
    _save(fig, "co2_timeline.png")


def sensor_agreement(pair):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    a.scatter(pair.co2_Wa2, pair.co2_Wa1, s=10, color=COL["Wa1"], alpha=0.5, edgecolors="none")
    lim = [400, max(pair.co2_Wa1.max(), pair.co2_Wa2.max()) + 100]
    a.plot(lim, lim, color=MUTED, lw=1, ls="--")
    a.set(xlabel="Wa2 CO₂ (ppm)", ylabel="Wa1 CO₂ (ppm)", xlim=lim, ylim=lim, title="同一時間點的讀值（虛線 = 完全一致）")
    m = (pair.co2_Wa1 + pair.co2_Wa2) / 2
    d = pair.co2_Wa1 - pair.co2_Wa2
    b.scatter(m, d, s=10, color=COL["Wa1"], alpha=0.5, edgecolors="none")
    for v, ls in [(d.mean(), "-"), (d.mean() - 1.96 * d.std(), "--"), (d.mean() + 1.96 * d.std(), "--")]:
        b.axhline(v, color=MUTED, lw=1, ls=ls)
        b.text(m.max(), v, f" {v:.0f}", va="center", color=MUTED)
    b.set(xlabel="兩台平均 (ppm)", ylabel="Wa1 − Wa2 (ppm)", title="Bland–Altman：平均差與 95% 一致界限")
    fig.tight_layout()
    _save(fig, "sensor_agreement.png")


def stroop_vs_co2(blk):
    b = blk.dropna(subset=["co2_Wa1"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, s in zip(axes[:2], ["Wa1", "Wa2"]):
        bb = b.dropna(subset=[f"co2_{s}"])
        se = bb.rt_sd / np.sqrt(bb.n_students)
        ax.errorbar(bb[f"co2_{s}"], bb.rt_mean, yerr=se, fmt="o", ms=8, color=COL[s], capsize=3)
        for _, r in bb.iterrows():
            ax.annotate(r.block, (r[f"co2_{s}"], r.rt_mean), textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color=MUTED)
        ax.set(xlabel=f"{s} 施測時 CO₂ (ppm)", ylabel="平均反應時間 (ms)", title=f"{s}：各場次平均 ± SE")
    ax = axes[2]
    ax.plot(blk.mean_test_no, blk.rt_mean, "o-", color=INK, ms=8)
    for _, r in blk.iterrows():
        ax.annotate(r.block, (r.mean_test_no, r.rt_mean), textcoords="offset points", xytext=(6, 6), fontsize=8,
                    color=MUTED)
    ax.set(xlabel="平均第幾次做 Stroop", ylabel="平均反應時間 (ms)", title="練習效應：做越多次越快")
    fig.tight_layout()
    _save(fig, "stroop_vs_co2.png")


def model_comparison(res):
    keep = ["Stroop 平均反應時間 (ms)", "Stroop 干擾分數 (ms)", "自覺疲勞（現在，1–7）", "心率 (bpm)，控制上課經過時間",
            "HRV log(RMSSD)，控制上課經過時間"]
    r = res[res.outcome.isin(keep)]
    fig, axes = plt.subplots(1, len(keep), figsize=(4 * len(keep), 4))
    for ax, o in zip(axes, keep):
        g = r[r.outcome == o].set_index("exposure").reindex(["Wa1", "Wa2", "Avg"])
        ax.bar(g.index, g.delta_aic_vs_base, color=[COL[e] for e in g.index], width=0.6)
        ax.axhline(0, color=MUTED, lw=1)
        ax.axhline(-2, color=MUTED, lw=1, ls="--")
        for i, (e, v) in enumerate(g.delta_aic_vs_base.items()):
            ax.text(i, v, f"{v:.1f}", ha="center", va="top" if v < 0 else "bottom", color=INK, fontsize=9)
        ax.set_title(o, fontsize=9.5, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("加入 CO₂ 後的 ΔAIC（越負越好；虛線 −2）")
    fig.tight_layout()
    _save(fig, "model_comparison.png")


def hr_vs_co2(hw):
    sessions = sorted(hw.dropna(subset=["co2_Avg"]).session.unique())
    fig, axes = plt.subplots(2, len(sessions), figsize=(3.2 * len(sessions), 6), sharey="row", sharex="col")
    for j, sess in enumerate(sessions):
        g = hw[hw.session == sess].groupby("win").agg(hr=("hr", "mean"), co2=("co2_Avg", "mean")).dropna()
        axes[0, j].plot(g.index, g.hr, color=INK)
        axes[1, j].plot(g.index, g.co2, color=COL["Avg"])
        axes[0, j].set_title(sess, fontsize=10)
        axes[1, j].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        axes[1, j].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    axes[0, 0].set_ylabel("所有配戴者平均心率 (bpm)")
    axes[1, 0].set_ylabel("CO₂ 兩台平均 (ppm)")
    fig.suptitle("每 5 分鐘：心率（上）與 CO₂（下）", y=1.02, color=INK)
    fig.tight_layout()
    _save(fig, "hr_vs_co2.png")


def fatigue_vs_co2(blk):
    b = blk.dropna(subset=["co2_Wa1", "fatigue_now"])
    fig, (a, c) = plt.subplots(1, 2, figsize=(11, 4.4))
    for _, r in b.iterrows():
        x = np.nanmean([r.co2_Wa1, r.co2_Wa2])
        pm = r.block.endswith("PM")
        a.scatter(x, r.fatigue_now, s=70, color=COL["Avg"] if pm else "#fcfcfb", edgecolors=COL["Avg"], lw=2,
                  zorder=3)
        a.annotate(r.block, (x, r.fatigue_now), textcoords="offset points", xytext=(6, 6), fontsize=8, color=MUTED)
        c.scatter(x, r.fss, s=70, color=MUTED if pm else "#fcfcfb", edgecolors=MUTED, lw=2, zorder=3)
    a.set(xlabel="填寫前 3 分鐘 CO₂（兩台平均；09-21 AM 只有 Wa1）", ylabel="平均自覺疲勞（1–7）",
          title="現在的疲勞程度（實心 = 下午，空心 = 上午）")
    c.set(xlabel="填寫前 3 分鐘 CO₂", ylabel="平均 FSS（1–7）", title="負對照：FSS 問「過去 24 小時」", ylim=a.get_ylim())
    fig.tight_layout()
    _save(fig, "fatigue_vs_co2.png")


def seating_simulation(sim):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (o, g) in zip(axes, sim.groupby("outcome", sort=False)):
        x = np.arange(len(g))
        ax.bar(x - 0.18, g.pct_seating_wins_by_2, width=0.34, color=COL["Wa2"], label="用座位表明顯較好（ΔAIC>2）")
        ax.bar(x + 0.18, g.pct_avg_detects_effect, width=0.34, color=COL["Avg"], label="不用座位表也測得到效應（p<.05）")
        ax.set_xticks(x, [f"{v:g}" for v in g.true_beta_per_100ppm])
        unit = "ms" if o == "rt_mean" else "分"
        ax.set(xlabel=f"模擬的真實效應（每 100 ppm，{unit}）", ylabel="模擬中的比例 (%)", ylim=(0, 100),
               title="反應時間" if o == "rt_mean" else "自覺疲勞")
        ax.grid(axis="x", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    fig.suptitle("假設「學生吸到的是最近那台」為真：座位表能幫上忙的機率", y=1.03, color=INK)
    fig.tight_layout()
    _save(fig, "seating_simulation.png")


def nonparametric(plot: dict, summary: pd.DataFrame):
    """左：每位學生的個人內 Spearman ρ（一點一人，不標代碼）；右：高／低 CO2 配對。"""
    rho_keys = [k for k in plot if k.endswith("（兩台平均）")]
    pair_keys = ["自覺疲勞（兩台平均）", "Stroop 反應時間（兩台平均）"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    ax = axes[0]
    for i, k in enumerate(rho_keys):
        v = plot[k]["rho"]
        jit = np.random.default_rng(i).uniform(-0.12, 0.12, len(v))
        ax.scatter(np.full(len(v), i) + jit, v, s=40, color=COL["Avg"], alpha=0.8, edgecolors="#fcfcfb", zorder=3)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=2.5, zorder=4)
        p = summary[(summary.outcome == k) & summary.method.str.startswith("個人內")].p.iloc[0]
        ax.text(i, 1.08, f"p = {p:.2f}", ha="center", fontsize=8.5, color=MUTED)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(rho_keys)), [k.replace("（兩台平均）", "") for k in rho_keys], fontsize=8.5,
                  rotation=20, ha="right")
    ax.set(ylim=(-1.1, 1.2), ylabel="個人內 Spearman ρ（CO₂ vs 結果）",
           title="每點 = 一位學生；橫線 = 中位數（Wilcoxon 檢定中位數 ≠ 0）")
    ax.grid(axis="x", visible=False)
    for ax, k in zip(axes[1:], pair_keys):
        m = plot[k]["pairs"]  # 欄位：[低, 高]
        for lo, hi in m:
            ax.plot([0, 1], [lo, hi], color=COL["Avg"] if hi > lo else COL["Wa1"], alpha=0.7, marker="o", ms=5)
        ax.plot([0, 1], np.median(m, axis=0), color=INK, lw=3, marker="o", ms=8, zorder=5)
        row = summary[(summary.outcome == k) & summary.method.str.startswith("高／低")].iloc[0]
        ax.set_xticks([0, 1], ["CO₂ < 1000", "CO₂ ≥ 1000"])
        ax.set_xlim(-0.3, 1.3)
        name = k.replace("（兩台平均）", "")
        ax.set_title(f"{name}：每線一人\nHL 中位差 {row.hl_diff_high_minus_low:+.2f}，p = {row.p:.3f}",
                     fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[1].set_ylabel("學生平均（1–7 分）")
    axes[2].set_ylabel("學生平均反應時間 (ms)")
    fig.tight_layout()
    _save(fig, "nonparametric.png")


PATH_POS = {
    "A": {"co2h": (0, 4), "temp": (0, 2.9), "rh": (0, 1.8), "sleep_h": (0, 0.7), "sleep_q": (0, -0.4),
          "fatigue_now": (1.45, 1.1), "test_no": (2.9, 2.0), "rt_mean": (2.9, 3.4), "interference": (2.9, 0.6)},
    "B": {"co2h": (0, 4), "temp": (0, 2.9), "rh": (0, 1.8), "sleep_h": (0, 0.7), "sleep_q": (0, -0.4),
          "hr_pre": (1.45, 3.5), "fatigue_now": (1.45, 1.0), "test_no": (2.9, 3.6), "rt_mean": (2.9, 2.0)},
}
# 研究假設的路徑：一律畫出；其他控制路徑只有在 bootstrap 顯著時才畫（完整數值見 path_coefficients.csv）
HYPOTHESIZED = {"A": {("co2h", "fatigue_now"), ("fatigue_now", "rt_mean"), ("fatigue_now", "interference"),
                      ("co2h", "rt_mean"), ("co2h", "interference")},
                "B": {("co2h", "hr_pre"), ("hr_pre", "fatigue_now"), ("co2h", "fatigue_now"),
                      ("fatigue_now", "rt_mean"), ("hr_pre", "rt_mean"), ("co2h", "rt_mean")}}


def path_diagram(paths: pd.DataFrame, fit: pd.DataFrame, labels: dict, exo_corr: dict | None = None):
    """徑路圖：數字為標準化係數 β；粗黑實線 = 以學生為單位 bootstrap 95% CI 不含 0，灰虛線 = 含 0。"""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, (name, pos) in zip(axes, PATH_POS.items()):
        f = fit[fit.model == name].iloc[0]
        ax.set_xlim(-1.25, 3.4)
        ax.set_ylim(-0.9, 5.1)
        ax.axis("off")
        pm = paths[paths.model == name]
        for k, (x, y) in pos.items():
            key = k in ("fatigue_now", "hr_pre")
            ax.add_patch(FancyBboxPatch((x - 0.38, y - 0.22), 0.76, 0.44,
                                        boxstyle="round,pad=0.02,rounding_size=0.06", fc="#fcfcfb",
                                        ec=INK if key else MUTED, lw=2 if key else 1.3, zorder=3))
            ax.text(x, y, labels[k], ha="center", va="center", fontsize=10.5, color=INK, zorder=4,
                    fontweight="bold" if key else "normal")
        hidden = 0
        for _, r in pm.iterrows():
            sig = r.boot_ci_low > 0 or r.boot_ci_high < 0
            if (r["from"], r["to"]) not in HYPOTHESIZED[name] and not sig:
                hidden += 1
                continue
            (x0, y0), (x1, y1) = pos[r["from"]], pos[r["to"]]
            rad = 0.0
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                         connectionstyle=f"arc3,rad={rad}", shrinkA=40, shrinkB=40,
                                         color=INK if sig else "#a9a7a0", lw=2.4 if sig else 1.2,
                                         ls="-" if sig else (0, (5, 3)), zorder=2))
            lx, ly = (x0 + x1) / 2, (y0 + y1) / 2
            if abs(x1 - x0) < 0.01:  # 垂直的線，標籤放旁邊
                lx += 0.33
            ax.text(lx, ly, f"β = {r.beta_std:+.2f}", fontsize=9.5, ha="center", va="center",
                    color=INK if sig else MUTED, fontweight="bold" if sig else "normal",
                    bbox=dict(fc="#fcfcfb", ec="none", pad=1.2), zorder=5)
        if exo_corr and name in exo_corr:  # 外生變項之間的相關（雙箭頭）
            for (u, v), rv in exo_corr[name].items():
                (x0, y0), (x1, y1) = pos[u], pos[v]
                ax.add_patch(FancyArrowPatch((x0 - 0.38, y0), (x1 - 0.38, y1), arrowstyle="<|-|>",
                                             mutation_scale=11, connectionstyle="arc3,rad=0.55",
                                             color=COL["Wa1"], lw=1.2, zorder=1))
                ax.text(x0 - 0.38 - 0.55 * abs(y1 - y0) / 2 - 0.04, (y0 + y1) / 2, f"r = {rv:+.2f}",
                        fontsize=9, color=COL["Wa1"], ha="right", va="center",
                        bbox=dict(fc="#fcfcfb", ec="none", pad=0.5))
        warn = "" if name == "A" else "\n⚠ 樣本太小，標準化 β > 1 代表共線性，係數不可解讀"
        title = ("模型 A：CO₂ → 自覺疲勞 → Stroop（控制溫濕度、睡眠、第幾次施測）" if name == "A"
                 else "模型 B：CO₂ → 施測前心率 → 疲勞 → 反應時間（子樣本）")
        ax.set_title(f"{title}\nn = {int(f.n)} 人次、{int(f.students)} 人；Fisher's C = {f.fisher_C:.1f}"
                     f"（df = {int(f.df)}，p = {f.p:.2f}）{warn}", fontsize=11, color=INK)
        ax.text(1.45, -0.85, f"另有 {hidden} 條控制路徑不顯著、未畫出（見 path_coefficients.csv）",
                ha="center", fontsize=9, color=MUTED)
    fig.text(0.5, 0.005, "β = 標準化係數。粗黑實線：以學生為單位 bootstrap 95% CI 不含 0；灰虛線：含 0。"
             "藍色雙箭頭：外生變項間的相關。Fisher's C 的 p > .05 表示未畫的路徑與資料不衝突。",
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "path_diagram.png")


def make_all(co2, pair, blk, hw, res, trials, seat_sim, np_plot=None, np_sum=None):
    if np_plot is not None:
        nonparametric(np_plot, np_sum)
    fatigue_vs_co2(blk)
    seating_simulation(seat_sim)
    co2_timeline(co2, trials)
    sensor_agreement(pair)
    stroop_vs_co2(blk)
    model_comparison(res)
    hr_vs_co2(hw)


def correlation_heatmap(mats: dict, labels: list[str], n: int, students: int):
    """發散色階：藍 = 負相關、紅 = 正相關、中點淺灰；格內數字為 ρ，* 為 p < .05。"""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#eeede9", "#e34948"])
    k = len(labels)
    fig, axes = plt.subplots(1, len(mats), figsize=(8.2 * len(mats), 7.4))
    for ax, (title, (r, p)) in zip(np.atleast_1d(axes), mats.items()):
        rv, pv = r.values.astype(float), p.values.astype(float)
        mask = np.triu(np.ones_like(rv, bool), 1)  # 只畫下三角，避免重複
        shown = np.where(mask, np.nan, rv)
        im = ax.imshow(shown, cmap=cmap, vmin=-1, vmax=1)
        for i in range(k):
            for j in range(i + 1):
                if i == j:
                    continue
                star = "*" if pv[i, j] < 0.05 else ""
                ax.text(j, i, f"{rv[i, j]:.2f}{star}", ha="center", va="center", fontsize=8.5,
                        color="#fcfcfb" if abs(rv[i, j]) > 0.55 else INK)
        ax.set_xticks(range(k), labels, rotation=40, ha="right", fontsize=9)
        ax.set_yticks(range(k), labels, fontsize=9)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=11, color=INK)
    fig.colorbar(im, ax=axes, shrink=0.7, label="Spearman ρ")
    fig.suptitle(f"變項間相關（n = {n} 人次、{students} 人；* p < .05，未校正多重比較）", color=INK, y=0.98)
    _save(fig, "correlation_heatmap.png")


def prediction_simulation(traj: dict, sc: pd.DataFrame, fits: pd.DataFrame):
    """左：實測擬合出的換氣率分布；中：三種通風情境的 50 分鐘 CO2 軌跡；右：下課時預測的疲勞變化（95% bootstrap 區間）。"""
    cols = [COL["Wa1"], COL["Avg"], COL["Wa2"]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [0.8, 1.3, 1]})
    ax = axes[0]
    jit = np.random.default_rng(0).uniform(-0.1, 0.1, len(fits))
    ax.scatter(jit, fits.ach_per_h, s=50, color=COL["Avg"], edgecolors="#fcfcfb", zorder=3)
    ax.hlines(fits.ach_per_h.median(), -0.3, 0.3, color=INK, lw=2.5)
    ax.set(xlim=(-0.6, 0.6), xticks=[], ylabel="換氣率 λ（次／小時）",
           title=f"實測擬合（{len(fits)} 節×感測器）\n中位數 {fits.ach_per_h.median():.2f} 次／小時")
    ax = axes[1]
    for (name, (t, c)), col in zip(traj.items(), cols):
        ax.plot(t, c, color=col, lw=2.4, label=f"{name}（λ = {sc.set_index('scenario').loc[name, 'ach_per_h']:.1f}）")
    ax.axhline(1000, color=MUTED, ls="--", lw=1)
    ax.text(1, 1015, "1000 ppm（室內空氣品質標準）", fontsize=8.5, color=MUTED, va="bottom")
    ax.set(xlabel="上課經過時間（分鐘）", ylabel="CO₂ (ppm)", title="一堂 50 分鐘課的 CO₂ 模擬（關窗開冷氣）")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax = axes[2]
    y = np.arange(len(sc))
    for i, (_, r) in enumerate(sc.iterrows()):
        ax.plot([r.d_fatigue_ci_low, r.d_fatigue_ci_high], [i, i], color=cols[i], lw=3, solid_capstyle="round")
        ax.scatter(r.d_fatigue_end, i, s=80, color=cols[i], edgecolors="#fcfcfb", zorder=3)
        ax.text(r.d_fatigue_ci_high + 0.03, i, f"{r.d_fatigue_end:+.2f}", va="center", fontsize=9.5, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, sc.scenario)
    ax.invert_yaxis()
    ax.set(xlabel="下課時自覺疲勞變化（1–7 分）", title="預測疲勞增加量（95% bootstrap 區間）")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "prediction_simulation.png")


def assumption_qq(qq: dict, checks: pd.DataFrame):
    """每個結果變項的「高 − 低 CO2」配對差值 Q-Q 圖；標題列出 Shapiro p 與檢定選擇。"""
    keys = list(qq)
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.9 * nrow), squeeze=False)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    for ax, k in zip(axes.flat, keys):
        v = np.sort(np.asarray(qq[k], float))
        c = checks[checks.outcome == k].iloc[0]
        if len(v) >= 3:
            th = stats_norm_ppf((np.arange(1, len(v) + 1) - 0.375) / (len(v) + 0.25))
            ax.scatter(th, v, s=42, color=COL["Avg"], edgecolors="#fcfcfb", zorder=3)
            sl, ic = np.polyfit(th, v, 1)
            xx = np.array([th.min(), th.max()])
            ax.plot(xx, ic + sl * xx, color=MUTED, lw=1.2, ls="--")
        must = c.decision.startswith("必須")
        ax.set_title(f"{k}\nn = {int(c.n_students)}；SW p = {c.shapiro_diff_p:.2f} → {c.decision}",
                     fontsize=9.5, color=COL["Wa2"] if must else INK)
        ax.set_xlabel("常態理論分位數", fontsize=9)
        ax.set_ylabel("高 − 低 CO₂ 差值", fontsize=9)
    fig.suptitle("步驟 1：配對差值的常態性檢查（點偏離虛線＝偏離常態；橘色標題＝必須用無母數）", color=INK, y=1.01)
    fig.tight_layout()
    _save(fig, "assumption_qq.png")


def stats_norm_ppf(p):
    from scipy.stats import norm
    return norm.ppf(p)


def effect_size_forest(eff: pd.DataFrame):
    """左：Hedges' g_av（Cohen's d 的小樣本校正）；右：配對等級二系列相關 r_rb。誤差線＝學生層級 bootstrap 95% CI。"""
    e = eff[eff.n_students >= 5].dropna(subset=["g_av"]).reset_index(drop=True)  # n < 5 的 CI 不可信，不畫
    fam_col = {"主要": INK, "敏感度": COL["Wa1"], "負對照": MUTED, "探索": COL["Wa2"]}
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 0.62 * len(e) + 2.2), sharey=True,
                             gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(e))
    for ax, (k, lim, lab) in zip(axes, [("g_av", 2.2, "Hedges' g_av（高 − 低 CO₂）"),
                                        ("r_rb", 1.05, "配對等級二系列相關 r_rb")]):
        if k == "g_av":
            for lo, hi, a in [(0.2, 0.5, 0.03), (0.5, 0.8, 0.06), (0.8, lim, 0.09)]:
                for sgn in (1, -1):
                    ax.axvspan(sgn * lo, sgn * hi, color=COL["Avg"], alpha=a, lw=0)
            for v, t in [(0.2, "小"), (0.5, "中"), (0.8, "大")]:
                ax.text(v + 0.03, len(e) - 0.45, t, fontsize=8.5, color=MUTED)
        ax.axvline(0, color=MUTED, lw=1)
        for i, r in e.iterrows():
            col = fam_col.get(r.family, INK)
            ax.plot([r[f"{k}_ci_low"], r[f"{k}_ci_high"]], [i, i], color=col, lw=2.6, solid_capstyle="round")
            ax.scatter(r[k], i, s=70, color=col, edgecolors="#fcfcfb", zorder=3)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    labels = [f"{r.outcome}（n = {int(r.n_students)}）" for _, r in e.iterrows()]
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    for i, r in e.iterrows():
        holm = f"，Holm p = {r.wilcoxon_p_holm:.3f}" if pd.notna(r.get("wilcoxon_p_holm")) else ""
        axes[1].text(1.08, i, f"g = {r.g_av:+.2f}（{r.magnitude_g_av}）  Wilcoxon p = {r.wilcoxon_p:.3f}{holm}",
                     transform=axes[1].get_yaxis_transform(), va="center", fontsize=9, color=INK)
    handles = [plt.Line2D([], [], color=c, lw=3, label=f) for f, c in fam_col.items() if f in set(e.family)]
    axes[0].legend(handles=handles, frameon=False, fontsize=9, loc="lower left", ncol=4,
                   bbox_to_anchor=(0, -0.32 if len(e) < 6 else -0.2))
    fig.suptitle("步驟 3：效果量（每人比較 CO₂ ≥ 1000 與 < 1000 ppm；誤差線＝以學生為單位 bootstrap 95% CI）",
                 color=INK, y=1.0)
    fig.tight_layout()
    _save(fig, "effect_size_forest.png")



def ucf_map(rows: dict):
    """UCF 論證路線圖：上＝中心論點（Unity）；中＝分析步驟，前一步的輸出是下一步的輸入（Flow）；
    下＝研究問題，連線表示哪一步提供論據（Coherence）。"""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(18, 7.4))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    def box(x, y, w, h, title, body, ec=MUTED, fc="#fcfcfb", lw=1.4, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec,
                                    lw=lw, zorder=2))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10.5, fontweight="bold",
                color=tc, zorder=3)
        ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=8.8, color=INK, zorder=3,
                linespacing=1.45)

    box(0.4, 6.05, 17.2, 1.3, "Unity｜中心論點", rows["thesis"], ec=INK, lw=2)
    xs = [0.4 + i * 2.93 for i in range(6)]
    for i, (x, (t, b)) in enumerate(zip(xs, rows["steps"])):
        box(x, 3.3, 2.6, 2.05, t, b, ec=COL["Wa1"], lw=1.6)
        if i < 5:
            ax.add_patch(FancyArrowPatch((x + 2.6, 4.32), (xs[i + 1], 4.32), arrowstyle="-|>", mutation_scale=16,
                                         color=COL["Wa1"], lw=2, zorder=1))
    ax.text(9, 5.62, "Flow｜每一步的輸出就是下一步的輸入", ha="center", fontsize=10, color=COL["Wa1"])
    qx = [0.4, 6.27, 12.13]
    for x, (t, b, verdict) in zip(qx, rows["questions"]):
        col = {"支持": COL["Avg"], "部分": COL["Wa2"], "未能回答": MUTED}[verdict]
        box(x, 0.1, 5.47, 1.85, t, b, ec=col, lw=2.2, tc=col)
    ax.text(0.4, 2.62, "Coherence｜連線＝該步驟為此研究問題提供論據", ha="left", fontsize=10, color=MUTED)
    for si, qi in rows["links"]:
        x0 = xs[si] + 1.3
        x1 = qx[qi] + 2.735
        ax.add_patch(FancyArrowPatch((x0, 3.3), (x1, 1.95), arrowstyle="-|>", mutation_scale=11,
                                     color="#b5b3ad", lw=1.0, zorder=1))
    fig.tight_layout()
    _save(fig, "ucf_map.png")


def hrv_conversion(m: pd.DataFrame, out: dict, polar_rmssd: np.ndarray):
    """左：貼片同一 5 分鐘窗的真 RMSSD vs 由每秒心率換算；右：三種來源的 RMSSD 分布（手環只有平滑過的每秒心率）。"""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax.scatter(m.rmssd, m.rmssd_from_hr, s=30, color=COL["Wa1"], alpha=0.7, edgecolors="#fcfcfb")
    lim = [4, max(m.rmssd.max(), m.rmssd_from_hr.max()) * 1.2]
    ax.plot(lim, lim, color=MUTED, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=lim, ylim=lim, xlabel="真 RMSSD：貼片逐拍 RR (ms)", ylabel="由貼片每秒心率換算的 RMSSD (ms)")
    ax.set_title(f"貼片：{out['windows']} 窗、{out['wearers']} 個配戴節次\nSpearman ρ = {out['spearman_rho']:.2f}，"
                 f"換算值 ≈ 真值 × {out['median_ratio_derived_over_true']:.2f}", fontsize=10.5)
    groups = [("貼片\n逐拍 RR（真）", m.rmssd.values, COL["Avg"]),
              ("貼片\n每秒心率換算", m.rmssd_from_hr.values, COL["Wa1"]),
              ("手環\n每秒心率換算", polar_rmssd, COL["Wa2"])]
    for i, (lab, v, c) in enumerate(groups):
        jit = np.random.default_rng(i).uniform(-0.16, 0.16, len(v))
        bx.scatter(np.full(len(v), i) + jit, v, s=10, color=c, alpha=0.35, edgecolors="none")
        bx.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=2.5)
        bx.text(i, np.median(v) * 1.25, f"{np.median(v):.1f}", ha="center", fontsize=10, color=INK,
                fontweight="bold")
    bx.set_yscale("log")
    bx.set_xticks(range(3), [g[0] for g in groups])
    bx.set(ylabel="5 分鐘 RMSSD (ms，對數尺度)", title="手環的平滑心率把心跳間的變化抹掉了（橫線＝中位數）")
    bx.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "hrv_conversion.png")
