# LMM_test：教室 CO₂ 對學生認知與生理表現的初步 LMM 分析

專題研究「高中教室二氧化碳濃度對學生認知表現影響之預測模型」的初步資料檢核。

**主報告（UCF 整合、無母數、Cohen's d）：[UCF_REPORT.md](UCF_REPORT.md)**

技術細節：[REPORT.md](REPORT.md)（變項類別、LMM、無母數、徑路分析、模擬預測）。數值總表：[FACTS.md](FACTS.md)。

圖表：`results/figures/`（PNG）與 `results/figures/svg/`（SVG，文字可編輯）。**給指導教授的圖表集（24 張，依研究流程排列並附說明）：[results/圖表集.pdf](results/圖表集.pdf)**

## 資料來源
| 資料 | 位置 | 是否在本 repo |
|---|---|---|
| CO₂／溫濕度（Wa1、Wa2 兩台 SCD30） | `data/co2/` | ✅（環境資料，不含個資） |
| Stroop 逐題紀錄 | `../cognitive_performance_test/output/` | ❌ 含學生班級座號，依 REC 規定不可上傳 |
| 手環／心電貼片心率、RR | `../ECG/data/` | ❌ 研究參與者生理資料 |
| 自覺疲勞量表（Google 表單匯出） | `data/private/fatigue_responses.csv` | ❌ 含學生代碼 |
| 研究代碼對照表（裝置、冷氣、人數） | `data/private/code_table.csv`（字母代號對照：`patch_letters.csv`） | ❌ 可對應到個人 |

兩個姊妹資料夾的路徑可用環境變數 `STROOP_DIR`、`ECG_DIR` 覆寫。
執行時產生的逐人資料表放在 `data/private/`（已 gitignore）。

## 執行
```bash
pip install -r requirements.txt
python src/run_analysis.py
```

## 程式
- `src/data.py`：讀取與對齊（Stroop 40 秒區間 ±60 秒取 CO₂ 平均；心率／HRV 5 分鐘窗）
- `src/models.py`：隨機截距 LMM、Nakagawa R²、LOSO 與 5 折交叉驗證
- `src/run_analysis.py`：Wa1 / Wa2 / 兩台平均 各跑一次並比較
- `src/seating_test.py`：座位表是否有用（窮舉所有座位安排 + 模擬）
- `src/nonparametric.py`：無母數分析（個人內 Spearman＋Wilcoxon、高／低 CO₂ 配對、節次置換檢定、Friedman／Page）
- `src/path_analysis.py`：徑路分析（piecewise SEM，LMM＋以學生為單位的 cluster bootstrap、Fisher's C）
- `src/correlations.py`：整體與個人內 Spearman 相關矩陣
- `src/prediction.py`：CO₂ 質量平衡擬合、通風情境模擬、疲勞預測（bootstrap 不確定性）、留一節次交叉驗證
- `src/effect_sizes.py`：檢定選擇（尺度、常態、離群值、天花板）與效果量（Hedges' g、r_rb、LMM d₁₀₀₀，學生層級 bootstrap CI，Holm 校正）
- `src/hrv_check.py`：檢驗每秒心率能否換算成 HRV（貼片 vs 手環）
- `src/figure_book.py`：補充圖表（資料完整度、暴露分級、貼片品質、Stroop 效度、練習效應、共線性、溫濕度調整、質量平衡擬合、檢定力）並輸出 PDF 圖表集
- `src/ucf_map.py`：UCF 論證路線圖
- `src/figures.py`：圖表（PNG＋SVG；不畫可辨識個人的資料）

`data/private/patch_letters.csv`（貼片字母 → 編號，範本見 `patch_letters_TEMPLATE.csv`）存在時，
會自動加跑逐人心率模型與「CO₂ → 施測前心率 → 反應時間」中介檢驗。

## 輸出（`results/`）
- `model_comparison.csv`：每個結果變項 × 每個 CO₂ 來源的係數、ΔAIC、R²、交叉驗證 RMSE
- `sensor_agreement.csv`：兩台感測器一致性（Bland–Altman）
- `co2_exposure_by_session.csv`：各節次超過 1000/1500/2000 ppm 的時間比例
- `seating_exhaustive.csv`、`seating_simulation.csv`：座位表檢驗
- `session_summary.csv`、`confounding_check.csv`
- `figures/*.png`
