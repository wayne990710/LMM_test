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


def path_diagrams():
    """四個中介模型的徑路圖。環境 → 中介、中介 → 結果都畫；環境 → 結果的直接效果只畫顯著的。"""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    c = pd.read_csv(R / "path_coefficients.csv")
    c = c[c["角色"] == "路徑"]
    info = pd.read_csv(R / "path_models.csv").set_index("模型")
    ind = pd.read_csv(R / "path_indirect.csv")
    models = list(dict.fromkeys(c["模型"]))
    fig, axes = plt.subplots(2, 2, figsize=(17, 12.5))
    for ax, name in zip(axes.flat, models):
        g = c[c["模型"] == name]
        meds = (["瞬時心率", "疲勞程度"] if "序列" in name else ["瞬時心率"] if "心率" in name else ["疲勞程度"])
        tos = list(dict.fromkeys(g["到"]))
        outs = [v for v in tos if v not in meds]
        pos = {"CO₂": (0, 3.2), "溫度": (0, 2.0), "濕度": (0, 0.8)}
        if len(meds) == 1:
            pos[meds[0]] = (1.5, 2.0)
        else:
            pos[meds[0]], pos[meds[1]] = (1.25, 3.3), (1.75, 0.7)
        for k, o in enumerate(outs):
            pos[o] = (3.0, 2.0 if len(outs) == 1 else 3.3 - k * 1.3)
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(-0.3, 4.1)
        ax.axis("off")
        for k, (x, y) in pos.items():
            key = k in meds
            ax.add_patch(FancyBboxPatch((x - 0.36, y - 0.2), 0.72, 0.4, boxstyle="round,pad=0.02,rounding_size=0.06",
                                        fc="#fcfcfb", ec=INK if key else MUTED, lw=2 if key else 1.2, zorder=3))
            ax.text(x, y, k, ha="center", va="center", fontsize=11, zorder=4, fontweight="bold" if key else "normal")
        hidden = 0
        for _, r in g.iterrows():
            a, b = r["從"], r["到"]
            sig = r["bootstrap 下限"] > 0 or r["bootstrap 上限"] < 0
            direct = a in ("CO₂", "溫度", "濕度") and b in outs
            if direct and not sig:
                hidden += 1
                continue
            (x0, y0), (x1, y1) = pos[a], pos[b]
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, shrinkA=34,
                                         shrinkB=34, color=INK if sig else "#b5b3ac", lw=2.2 if sig else 1.0,
                                         ls="-" if sig else (0, (4, 3)), zorder=2,
                                         connectionstyle="arc3,rad=0.12" if direct else "arc3,rad=0"))
            t = 0.42 if not direct else 0.5
            ax.text(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, f"{r['β']:+.2f}", fontsize=9, ha="center", va="center",
                    color=INK if sig else MUTED, fontweight="bold" if sig else "normal",
                    bbox=dict(fc="#fcfcfb", ec="none", pad=0.8), zorder=5)
        n_sig = int(ind[ind["模型"] == name]["顯著"].sum())
        i = info.loc[name]
        ax.set_title(f"{name}\n{int(i['人次'])} 人次、{int(i['學生'])} 人；顯著的間接效果：{n_sig} 條", fontsize=11.5,
                     color=INK)
        ax.text(1.5, -0.25, f"另有 {hidden} 條不顯著的直接效果（環境 → 結果）未畫出", ha="center", fontsize=8.5,
                color=MUTED)
    fig.text(0.5, 0.005, "數字為標準化 β。黑色實線：以學生為單位 bootstrap 95% CI 不含 0；灰色虛線：含 0。"
             "每條方程式都控制人數、冷氣；疲勞的方程式另控制睡眠時長與品質。", ha="center", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, "04_徑路分析")


def main():
    sample_by_session()
    coefficient_forest()


if __name__ == "__main__":
    main()
