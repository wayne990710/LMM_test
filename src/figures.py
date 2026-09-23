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
})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def co2_timeline(co2, trials):
    sessions = ["09-21 AM", "09-21 PM", "09-22 AM", "09-22 PM", "09-23 AM", "09-23 PM"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharey=True)
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
    axes[0, 0].set_ylabel("CO₂ (ppm)")
    axes[1, 0].set_ylabel("CO₂ (ppm)")
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


def make_all(co2, pair, blk, hw, res, trials, seat_sim):
    fatigue_vs_co2(blk)
    seating_simulation(seat_sim)
    co2_timeline(co2, trials)
    sensor_agreement(pair)
    stroop_vs_co2(blk)
    model_comparison(res)
    hr_vs_co2(hw)
