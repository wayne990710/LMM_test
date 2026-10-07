"""主要 LMM 分析的圖（讀 results/LMM/ 的結果表）。PNG 300 dpi + SVG。"""
from __future__ import annotations

import numpy as np
import pandas as pd

import figures as F  # 共用字型與配色設定
from figures import COL, INK, MUTED, plt

R = F.D.ROOT / "results" / "LMM"
IV_COL = {"CO₂（每 100 ppm）": COL["Avg"], "溫度（°C）": COL["Wa2"], "濕度（%）": COL["Wa1"]}


def _save(fig, name):
    (R / "svg").mkdir(parents=True, exist_ok=True)
    fig.savefig(R / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(R / "svg" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def sample_by_session():
    """各節次各依變量有幾位學生的資料：說明樣本數為何不同。"""
    n = pd.read_csv(R / "lmm_n_by_session.csv", index_col=0)
    v = n.drop(columns=["冷氣", "現場人數"])
    fig, ax = plt.subplots(figsize=(11, 6.2))
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("n", ["#f4f3ef", "#a8d8bf", "#1baf7a"])
    ax.imshow(v.T.values, cmap=cmap, aspect="auto", vmin=0, vmax=12)
    for i, col in enumerate(v.columns):
        for j, val in enumerate(v[col]):
            ax.text(j, i, "—" if val == 0 else str(val), ha="center", va="center", fontsize=10,
                    color=MUTED if val == 0 else INK)
    ax.set_yticks(range(v.shape[1]), [c.replace(" log(RMSSD)", "") for c in v.columns])
    ax.set_xticks(range(v.shape[0]), [f"{b}\n冷氣{a}" for b, a in zip(v.index, n["冷氣"])], fontsize=8.5)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("各節次有資料的學生數（共 12 位學生；— ＝ 該節沒有資料）", color=INK, fontsize=12)
    fig.text(0.5, -0.04, "Stroop 與疲勞量表在下課後施測，下課時間較趕時部分學生沒有測到；"
             "心率／HRV 只有能從研究代碼對照表對應到學生的節次（09-24 起）。09-17 沒有 CO₂ 資料，不納入模型。",
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    _save(fig, "01_各節次樣本數")


def coefficient_forest():
    """每個依變量一格：7 個預測變項的標準化係數與 95% CI（自變量上色，控制變因灰色）。"""
    c = pd.read_csv(R / "lmm_coefficients.csv").dropna(subset=["係數"])
    c["scale"] = c["標準化係數"] / c["係數"]
    c["lo"], c["hi"] = c["95% CI 下限"] * c.scale, c["95% CI 上限"] * c.scale
    lo, hi = np.minimum(c.lo, c.hi), np.maximum(c.lo, c.hi)
    c["lo"], c["hi"] = lo, hi
    dvs = list(dict.fromkeys(c["依變量"]))
    order = list(dict.fromkeys(c["變項"]))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), sharex=True)
    for ax, dv in zip(axes.flat, dvs):
        g = c[c["依變量"] == dv].set_index("變項").reindex(order)
        for i, (v, r) in enumerate(g.iterrows()):
            if pd.isna(r["係數"]):
                continue
            col = IV_COL.get(v, MUTED)
            sig = r.p < 0.05
            ax.plot([r.lo, r.hi], [i, i], color=col, lw=2.4, solid_capstyle="round")
            ax.scatter(r["標準化係數"], i, s=60, zorder=3, color=col if sig else "#fcfcfb", edgecolors=col, lw=2)
            ax.text(1.02, i, f"p = {r.p:.3f}" if r.p >= 0.001 else "p < .001", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=8.5, color=INK if sig else MUTED, fontweight="bold" if sig else "normal")
        ax.axvline(0, color=MUTED, lw=1)
        ax.axhline(2.5, color="#e4e3df", lw=1)
        ax.set_yticks(range(len(order)), order, fontsize=9)
        ax.invert_yaxis()
        meta = c[c["依變量"] == dv].iloc[0]
        ax.set_title(f"{dv}（{meta['單位']}）", fontsize=11, color=INK)
        ax.grid(axis="y", visible=False)
    for ax in axes[-1]:
        ax.set_xlabel("標準化係數（預測變項 +1 SD 時，依變量變化幾個 SD）", fontsize=9)
    fig.suptitle("LMM 結果：三個自變量（上方，彩色）與四個控制變因（下方，灰色）同時放入模型\n"
                 "線＝95% 信賴區間；實心點＝p < .05", color=INK, fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, "02_LMM 係數")


def spearman_heatmap():
    """兩兩個人內 Spearman ρ。上方三列：環境 × 依變量；下方：依變量 × 依變量（下三角）。"""
    s = pd.read_csv(R / "spearman_pairs.csv")
    dvs = ["疲勞程度", "反應時間", "正確率", "干擾分數", "瞬時心率", "心率變異度"]
    rows = ["CO₂", "溫度", "濕度"] + dvs[1:]
    look = {}
    for _, r in s.iterrows():
        look[(r["變項 1"], r["變項 2"])] = look[(r["變項 2"], r["變項 1"])] = r
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("div", ["#2a78d6", "#eeede9", "#e34948"])
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    M = np.full((len(rows), len(dvs)), np.nan)
    for i, a in enumerate(rows):
        for j, b in enumerate(dvs):
            r = look.get((a, b))
            if r is None or (a in dvs and dvs.index(a) <= j):
                continue
            if pd.notna(r.get("ρ")):
                M[i, j] = r["ρ"]
    ax.imshow(M, cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")
    for i, a in enumerate(rows):
        for j, b in enumerate(dvs):
            r = look.get((a, b))
            if r is None or (a in dvs and dvs.index(a) <= j):
                continue
            if pd.isna(r.get("ρ")):
                ax.text(j, i, f"樣本不足\n{int(r['筆數'])} 筆、{int(r['學生'])} 人", ha="center", va="center",
                        fontsize=8, color=MUTED)
                continue
            star = "*" if r["FDR 校正 p"] < 0.05 else ""
            ax.text(j, i - 0.12, f"{r['ρ']:+.2f}{star}", ha="center", va="center", fontsize=11, color=INK,
                    fontweight="bold" if star else "normal")
            ax.text(j, i + 0.22, f"n = {int(r['筆數'])}，p = {r.p:.3f}", ha="center", va="center", fontsize=7.5,
                    color=MUTED)
    ax.axhline(2.5, color=INK, lw=1.5)
    ax.set_xticks(range(len(dvs)), dvs)
    ax.set_yticks(range(len(rows)), rows)
    ax.xaxis.tick_top()
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("個人內 Spearman 相關（每位學生先減去自己的平均）\n粗體＊＝FDR 校正後 p < .05；"
                 "p 為學生內置換檢定", color=INK, fontsize=11, pad=34)
    fig.tight_layout()
    _save(fig, "03_兩兩相關")


def main():
    sample_by_session()
    coefficient_forest()


if __name__ == "__main__":
    main()
