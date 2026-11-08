# 從「圖資料庫」到「時序圖分析平台」

> 對象：專案擁有者 ｜ 日期：2026-11-08 ｜ 依據：`origin/main` @ `b131a45`（26 crates / 約 7.5 萬行 Rust / 570+ 單元測試）

---

## 一、先把問題說清楚：為什麼現在的價值被低估

不是因為功能不夠，而是因為**買方無法用一句話判斷這是什麼、值多少錢**。三個可驗證的原因：

1. **自我描述是「資料庫」，但產品不是資料庫。** 「Graph-Temporal-Vector DataBase」「消除雙資料庫陷阱」是架構語言，不是需求語言。客戶不會早上醒來想「我今天需要消除雙資料庫陷阱」。
2. **品類一旦落在「圖資料庫」，就自動進入最差的賽道。** Neo4j / Kuzu / Memgraph / FalkorDB 已經把評估清單寫好了：Cypher、ACID、HA、生態、雲服務。單人專案在別人的清單上打分，永遠是低分。
3. **26 個 crate 對買方是負分。** 功能清單越長，「這是誰在維護、明年還在嗎」的疑慮越重。`README` 的 12 條 key features 沒有一條回答「所以我能拿它做什麼別人做不到的事」。

**價值公式**：感知價值 = 品類可信度 × 護城河具體度 × 取證時間倒數。功能數量不在公式裡，它只通過「維護風險」這一項**扣分**。

---

## 二、新定位

**一句話**：

> gtvdb 是嵌入式時序圖分析引擎：在同一個 Arrow 表上，回答「`T` 時刻這張圖長什麼樣」與「這個狀態是怎麼一步步變成現在的」。

**電梯陳述（Moore 模板）**：

> 對**需要「時間點正確」與「可解釋來源」的資料/ML 工程師**（point-in-time 特徵、agent 記憶、量子與混合實驗的除錯審計）而言，gtvdb 是一個**時序圖分析引擎**，它把「邊的有效區間」與「狀態的來源路徑」當成表的一等欄位，用 SQL 查詢，並嵌入你的進程而非另開一個服務。與 Neo4j（邊沒有原生時間區間）、kdb+（有時間沒有圖）、Zep/Graphiti（在別人的資料庫上做服務）不同，**時間與來源不是外掛層，而是儲存格式本身**。

### 三個支柱（每個都要能當場證明）

| 支柱 | 主張 | 可當場驗證的東西 |
|:--|:--|:--|
| **時間是一等欄位** | 邊帶半開區間 `[valid_from, valid_to)`；`as-of T` 是索引操作而非全表掃描 | `csr.rs` 的 zone map（因 `valid_to` 非單調，用 `max valid_to` 剪枝 + run 內二分）；`bitemporal.rs` 的業務時間 vs 系統時間 |
| **來源可解釋** | 每個狀態都能回溯到產生它的那條路徑與機率，代價 O(depth) 而非 O(狀態數) | `gtv-core/src/provenance.rs` + `gtv-core/src/quantum_state.rs` + `QUANTUM_STATE_*` SQL（2026-11-08 已完成） |
| **嵌入式分析，不是又一個伺服器** | 進程內、Arrow 零拷貝、DataFusion SQL、Parquet 落地，沒有 cluster 要運維 | `GtvContext`；`gtv` REPL；`cargo run -p gtv-cli` |

### G / T / V 重新定義

不要再說「三種資料庫合一」。改說：**一張表，三個維度**。

- **G**：邊（誰指向誰）
- **T**：邊的有效區間（何時成立）
- **V**：節點/邊上的向量（語意相近）

這三者在同一個 `RecordBatch` 裡，所以「時間切片後再做向量檢索再遍歷」不需要跨系統搬資料。這才是護城河的**一句話版本**，也是唯一競爭對手無法快速補上的部分——它不是功能，是資料佈局。

---

## 三、對位表（給買方看的，不是給自己看的）

| | 原生時間區間邊 | 圖遍歷 | 欄式/嵌入式 | 狀態來源回溯 | 向量檢索 |
|:--|:--:|:--:|:--:|:--:|:--:|
| **gtvdb** | ✅ 半開區間 + 雙時間 | ✅ 時序 CSR、k-hop、direction-optimizing | ✅ Arrow/Parquet，進程內 | ✅ 路徑 + 機率 | ✅ HNSW/IVF + filter-aware |
| Neo4j | ❌（靠節點屬性/APOC 模擬） | ✅ | ❌（服務） | △（需自建） | △（外掛） |
| kdb+/q | △（時間序列，非區間圖） | ❌ | ✅ | ❌ | ❌ |
| DuckDB | ❌ | ❌（recursive CTE 可做但無索引） | ✅ | ❌ | ✅（vss） |
| Zep / Graphiti | △（時序知識圖，服務層） | ✅ | ❌ | △ | ✅ |
| Kuzu | ❌ | ✅ | ✅ | ❌ | △ |

**不要寫**：`kdb+ replacement`。把自己最強的對手寫進 tagline，等於告訴買方「請用 kdb+ 的標準評估我」。

---

## 四、價值階梯：技術價值 → 感知價值 → 收入

| 層 | 產出 | 誰付錢 | 為什麼這一層能收費 |
|:--|:--|:--|:--|
| **L1 核心（開源）** | `gtv-core` + `gtv-index` 發佈到 crates.io，穩定 API | 免費用戶、開發者 | 佔領「時序圖」這個詞；被嵌入即分發 |
| **L2 平台（開源 + 商業支援）** | `gtv` REPL / SQL / workload packs | 團隊、顧問 | 從「能用」到「敢用在生產」的距離就是付費點 |
| **L3 產業包（商業）** | 銀行監管（IRRBB / MA(BS)28 / CRM / AML）＋ 驗證腳本＋審計包 | 金融機構、顧問 | 條文對照、可審計、責任轉移——這三件事只有人能賣 |

**單人專案唯一能規模化的模式是「被嵌入」，不是「被部署」。** 所以 L1 必須先成立，L3 才有毛利。目前順序是反的：L3 的程式碼最多，L1 的門面是 73 bytes。

---

## 五、量子狀態圖：為什麼它是第一個 workload pack，而不是第 27 個 crate

使用者要的 schema（`state_id / parent_state / gate / timestamp / probability`）本質上不是「量子功能」，而是**一張帶時間與機率的來源森林**。它同時證明三件事：

1. **平台的通用性**：同一個 kernel，換一個領域只要約 600 行（`provenance.rs` + `state.rs`），不需要新引擎。
2. **護城河的實用性**：狀態重建是 O(depth) 的路徑乘積，不是掃描全部狀態；這正是時序 CSR 的長處。
3. **一個別人沒佔的位**：量子團隊現在的除錯方式是 dump statevector（2ⁿ 振幅）或 pickle 整個結果。**存「分支結構 + 閘序列 + 機率」是 O(gates)**，而且可以直接回答「這條 0.003 的分支是哪些閘造成的」。這在量子 SDK（Qiskit/qoqo/PennyLane）裡沒有對應品類。

**誠實邊界（一定要寫在文件與 README 裡）**：我們存的是**分支結構與機率**，不是振幅，因此**不能**從這個 log 計算干涉項或重建完整 statevector。要算振幅，只能存 blob 引用。把邊界寫清楚，比誇大更能建立信任，也避免第一個認真的用戶發現落差後離開。

商業化路徑：`quantum-state` 作為 L1 的 showcase + 一篇文章（「用時序圖除錯變分電路」），目標是 qoqo/Qiskit 社群的理解，不是營收。

---

## 六、對外文案的具體改動（可直接執行）

1. **GitHub repo description**（現在是一個字 `gtvdb`）：
   ```bash
   gh repo edit jacalwu/gtvdb \
     --description "Embedded temporal graph analytics: point-in-time graph queries + state provenance, in Arrow/DataFusion, in-process." \
     --add-topic temporal-graph --add-topic graph-analytics --add-topic arrow \
     --add-topic datafusion --add-topic provenance --add-topic quantum-computing
   ```
2. **根 `README.md`**：換成 landing page（已重寫）。第一屏必須有：一句話、30 秒可跑、架構圖、alpha 聲明。
3. **`README.en.md` 標題**：把 "high-performance, unified single-engine in-memory database" 改成 "temporal graph analytics engine"，並把 12 條 features 折成 3 組（時間 / 來源 / 向量）。
4. **`ROADMAP.md`**：改成三欄——**主線**（kernel、SQL、state graph）、**參考應用**（banking、hft）、**封存**（PQC、QAOA、distributed、q frontend）。現在還寫著 11 個 crate 與 P1–P5 全勾，這是「已完成」的錯覺。
5. **`v0.1.0-alpha` release**：有 release 才有「這是可評估的東西」的訊號。
6. **刪掉 tagline 裡的 "kdb+"、"dual-database trap"、"everything"**。

---

## 七、停止清單（比新增更重要）

| 停止 | 理由 |
|:--|:--|
| 新增 crate | 26 個已超過單人可維護上限；每個新 crate 都稀釋「這是什麼」 |
| 分散式 | 16 個 open issue 有 9 個是分散式，自 9/2 未動。目前「分散式」只是遠端執行 SQL，沒有 shard/planner/worker——承諾會摧毀可信度 |
| 量子計算（QAOA / QuEST 電路求解） | 與儲存平台的敘事無關，卻是 CI 唯一全綠的部分，製造「一切正常」的錯覺 |
| PQC / CBOM | 有價值但屬於獨立產品；留在主 repo 會讓「時序圖平台」這句話變弱 |
| 用 benchmark 當賣點 | 效能是門檻不是價值；TC4 的 `recall@10 = 0%` 被當 PASS 是自傷 |

> 這些不是刪除，是**移出主敘事**：搬到 `labs/` 或獨立 repo，README 只留一行連結。

---

## 八、90 天里程碑

| 週 | 產出 | 驗收 |
|:--|:--|:--|
| 1–2 | README landing page、repo description/topics、`v0.1.0-alpha`、CI 覆蓋 core/quantum | 陌生人 5 分鐘內能跑出第一個 `as-of` 查詢與一次狀態重建 |
| 3–4 | `gtv-core` + `gtv-index` 發佈 crates.io；API 文件（docs.rs） | `cargo add gtv-core` 可用 |
| 5–8 | 兩個 canonical demo：① point-in-time 特徵 join ② 量子 run 除錯 | 各一頁 notebook + 一張對比表（含與 DuckDB/Neo4j 的實測） |
| 9–12 | Python 綁定（pyo3，只暴露 6 個函式）+ 一篇技術文章 | `pip install` 後 10 行內完成時序圖查詢 |

**唯一要追的指標**：從「看到 repo」到「跑出第一個時間切片查詢」的完成率。不是 star。

---

## 九、風險與反面意見

- **「時序圖分析」是不是太小眾？** 是，但小眾 ≠ 沒錢：point-in-time 特徵、agent 記憶、審計回溯都是預算項目。相反「通用圖資料庫」是大市場但我們拿不到。
- **量子狀態圖會不會被當成噱頭？** 只要文件寫清楚「不存振幅」，它就是一個乾淨的能力展示，而不是誇大。若寫成「量子資料庫」，第一輪技術盡調就會崩。
- **銀行包怎麼辦？** 不刪。它是 L3 的現金流來源，只是不該和引擎共用同一個門面。等 L1 有可信度後，銀行反而是最容易轉化的客戶——他們買的就是「可審計的來源」。
