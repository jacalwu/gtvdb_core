# gtvdb 現狀評估與可行發展方向

> 評估基準：`origin/main` @ `b131a45`（2026-10-04，144 commits，CI 綠）
> 本地工作區：`C:/ai/gtvdb/gtvdb` @ `1e24d9e`（2026-08-30）——**落後 115 個 commit**
> 規模：24 個 crate / 約 74,800 行 Rust / 約 812 KB Markdown / 約 570 個單元測試
> 對外狀態：1 star、0 fork、0 watcher、16 個 open issue（全部建立於 2026-09-02 且未動）

---

## 0. 一句話結論

**這不是一個「庫」，而是一個在 38 天內長成的平台**：核心引擎（Arrow + DataFusion + Temporal-CSR + ANN）只佔約 22% 的程式碼，其餘是銀行監管分析面、PQC／量子、kdb+/q 前端、Web UI。工程質感在「高品質技術演示」之上、「可被外人採用」之下——真正的瓶頸不是技術能力，是**範圍（scope）與對外接口**。

---

## 1. 分層現狀

| 層 | Crate | LOC | 成熟度 | 判斷 |
|---|---|---:|---|---|
| **L0 Kernel** | `gtv-core` 3.1k、`gtv-array` 6.5k、`gtv-index` 4.7k、`gtv-pattern` 0.9k、`gtv-storage` 1.4k | ~16.6k | ★★★★☆ | 唯一達到「可對外發佈」水準的部分 |
| **L1 Engine / Surface** | `gtv-engine` **19.0k**、`gtv-catalog` 3.1k、`gtv-enterprise-sql` 4.3k、`gtv-q` 2.5k、`gtv-ingest` 2.4k、`gtv-server` 1.0k、`gtv-proto` 0.1k、`gtv-shm` 0.3k | ~32.7k | ★★☆☆☆ | 功能齊、耦合重；engine 已成巨石 |
| **L2 銀行領域** | `gtv-governance` 4.7k、`gtv-scenario` 4.5k、`gtv-largeexposure` 3.7k、`gtv-refdata` 1.0k | ~13.9k | ★★☆☆☆ | 規則實作完整，缺真實資料與外部驗證 |
| **L3 前緣／離題** | `gtv-crypto` 1.0k、`gtv-cbom` 1.2k、`gtv-quantum` 1.3k | ~3.5k | ★★★☆☆ | 本身寫得不錯，但與資料庫核心無關 |
| **L4 客戶端** | `gtv-cli` 6.6k、`webui/` | ~6.6k | ★★☆☆☆ | 展示用；demo 資料仍硬編碼 |

---

## 2. 真正稀缺、值得保護的資產

這四項是別人不容易複製的部分，也是所有發展方向都應該圍繞的東西。

### 2.1 Arrow 上的時序 CSR（`crates/gtv-core/src/csr.rs`）

不是「有 temporal 欄位的 CSR」，而是三層剪枝都做了的正確版本：

1. **半開區間語義** `valid_from <= T < valid_to`，貫穿 CSR、SQL、pattern、delta。
2. **run 內二分搜尋 `valid_from`**：只掃「可能已經開始」的邊。
3. **per-chunk zone map（chunk=64，記錄 max `valid_to`）**：整塊過期的直接跳過（`valid_to` 在 run 內非單調，所以不能用第二次二分——這個判斷是對的）。
4. 低出度（< 16）退回線性掃描——沒有為了「漂亮」而犧牲常數因子。

加上 `transpose()`（反向索引）、direction-optimizing BFS（pull/push 切換）、`BudgetTracker` / `CancelToken`（有界走訪與可取消），這是**論文附錄級別的實作**，不是玩具。

### 2.2 向量索引層（`crates/gtv-index`）

- 統一的 **`Metric` 契約**（L2 / Cosine / IP，全部轉成「越小越近」，並在文件裡明說 IP 不滿足三角不等式、對 HNSW/IVF 的影響）——這種細節意識少見。
- Flat（連續列主存 + AVX2/FMA 四累加器 + bounded top-K fold/reduce）、IVF（k-means coarse，f32 精確探測，無量化）、HNSW（連續記憶體佈局）。
- `plan_ann` / `execute_ann` 做 **filter-aware 路由 + exact rerank + recall 估計 + telemetry**，以及 `.gtvidx` 持久化容器。

這是目前 repo 裡**最接近「可直接商用元件」**的東西。

### 2.3 記憶體牆工程能力（`testcase/hft/TC1_TUNING.md`）

這份文件比程式碼更值錢：TC1 從 71 ms 優化到 2.5 ms，中間 v1→v4（branchless、prefetch、NT store、payload 解耦）**全部無效**，最後靠「熔接（fusion）把輸出 16 MB 砍到 8 MB」才突破，並用 CUDA merge-join 把同一原則搬到 GPU（3.5 ms，對主機負載最不敏感）。結論「突破記憶體牆只能削減移動的位元組，而非執行的指令」是有實測支撐的正確洞察。

### 2.4 工程紀律的跡證

- `gtv-storage/src/atomic.rs`：temp → fsync → rename → fsync(dir) 的原子發布。
- `hdb.rs`：kdb 風格的 `date/table/symbol.parquet` 分區 + 全域 symbol 字典 + 分區剪枝。
- CBOM drift check 進 CI（`cargo run -p gtv-cbom -- --check doc/cbom.md`）。
- 約 570 個 `#[test]`。

---

## 3. 八個可驗證的短板

### 3.1 CI 只照到外層，核心 55% 沒有 CI

`.github/workflows/ci.yml` 只有 4 個 job，跑的是：`gtv-cbom`、`gtv-crypto`、`gtv-server`（security / pqc_tls）、`gtv-quantum`、`gtv-shm`。

**`gtv-core`、`gtv-array`、`gtv-engine` 主路徑、`gtv-index`、`gtv-pattern`、`gtv-storage`、`gtv-catalog` 完全不在 CI 裡。** 結果是：CI 綠燈給人「專案很健康」的錯覺，而真正的引擎沒有任何回歸保護。這是目前**最高優先級的技術債**。

### 3.2 `gtv-engine` 變成 19k 行的巨石，且違反自己定的邊界

`prod_p4.md` 明寫「`gtv-core` / `gtv-engine` / `gtv-index` / `gtv-pattern` **不可以**依賴企業批 crate，analysis engine 主路徑零污染」。

實際上 `gtv-engine/Cargo.toml` 依賴 `gtv-quantum`、`gtv-catalog`；`src/` 下有 `crm.rs`、`hft.rs`、`indicator.rs`、`quant.rs`、`quantum.rs`、`tickdata.rs`、`yahoo.rs`、`market/`。**「DataFusion 整合層」變成了什麼都往裡塞的抽屜。**

### 3.3 沒有 WAL；CSR 不落地

持久化只到「Parquet 分區 + 原子 rename」。`docs/storage_design.md` 把 WAL、mmap、disk format 寫得很完整，但那是**設計文件**：`gtv-storage` 沒有 WAL，`TemporalCSR` 也沒有任何序列化格式（`grep` 不到 storage 依賴 csr）。每次啟動都要從邊表重建索引，10 億邊等級不可行。

### 3.4 「分散式」其實是遠端執行 SQL

`QueryService.Execute` 收一條 SQL、在單節點跑完、把整個結果用**一個 Arrow IPC payload** 回傳。沒有 shard、沒有 planner 切分、沒有 worker、沒有 Flight `DoGet` 串流。16 個 issue 裡 9 個是分散式（shard rebalance、shuffle/spill、follower reads、distributed knn…），**全部未開工**。

### 3.5 開源分發的第一格是斷的

- 根 `README.md`：**73 bytes**，只有一行語言切換連結。GitHub 首頁看到的就是這個。
- repo 描述：`gtvdb`；topics：空；無 release、無 tag、無 crates.io 發佈、無 docs.rs、無 `examples/` 入口、無 quickstart 能在 60 秒內跑出東西。
- 1 star / 0 fork。不是社群不感興趣，是**沒有任何入口**。

### 3.6 文件 812 KB 平鋪在根目錄

30+ 個 `.md` 在 repo 根（`forcast.md`、`forcast1.md`、`function2.md`、`function3.md`、`prod_p1..p4`、`hft_tc*.md`、`tuning3.txt`、`tuning5.md`…），`docs/` 又有 40+ 份。沒有 `index`、沒有 `status: draft/active/archive`、沒有與程式碼的對應關係。`ROADMAP.md` 仍寫「11 個 crate、P1–P5 全勾」，與 24 crate 的現實脫節——**新人（包括三個月後的你）無法判斷哪些是核心**。

> **已修（本次整理）**：30 份筆記全部移入 `docs/archive/`，並在 `docs/archive/README.md`
> 建立分類索引（架構史 / 產品規劃 / 預測規格 / 壓測紀錄 / 手冊），逐條標注「何時被什麼取代」。
> repo 根現在只剩 `README.md`、`README.zh-CN.md`、`README.en.md`、`ROADMAP.md`、`LICENSE.md`。
> 上面引用的檔名現已位於 `docs/archive/` 下。

### 3.7 驗收門檻有洞：latency 過了，recall 是 0%

`hft_tc2-tc4_results.md` 記錄 TC4 用 IVF 達到 `< 8 ms`（100k 2.5 ms、1M 4.6 ms），但同表的 `recall@10` 是 **0% / 40%**。文件解釋是「均勻隨機 512 維資料沒有聚類結構」——這個解釋合理，**但把這種查詢判定為 PASS 是門檻設計的問題**：延遲門檻必須與 recall 門檻成對，否則「快」可以靠「亂答」達成。

### 3.8 本地與遠端脫節

本地 checkout 停在 8/30，未提交的 `asof.rs`、`window.rs`、`window_simd/` 還在 working tree。這意味著有一段時間，實測數據與推送內容不是同一棵樹。

---

## 4. 三條可行的發展方向

三個身份目前互相拉扯：**嵌入式引擎**、**銀行監管分析產品**、**量化／HFT 工具**。任何一條都能成立，但同時做三條等於三條都不成立。

### A. 嵌入式「時序圖 + 向量」引擎（library-first）

| 項目 | 內容 |
|---|---|
| 一句話 | 給 AI agent / GraphRAG 用的、帶時間語義的嵌入式圖＋向量索引（「SQLite of temporal knowledge graphs」） |
| 客戶 | AI 應用開發者、做 agent 長期記憶的團隊、需要「as-of 某日期的關係」的分析系統 |
| 護城河 | 時間切片 CSR + 過濾感知 ANN 在同一個 Arrow runtime；這正是 Neo4j / LanceDB / kuzu 都缺的交集 |
| 要補 | 穩定 API 邊界、Python 綁定（PyO3）、磁碟格式與 mmap、增量更新（目前 CSR 全量重建）、範例 |
| 致命風險 | 市場教育成本；Zep/Graphiti（Neo4j 上的 temporal KG）已佔位 |
| 12 個月可行性 | **高**。單人可完成，不需團隊、不需牌照 |

### B. 銀行監管分析產品（domain-first）

| 項目 | 內容 |
|---|---|
| 一句話 | 面向 HKMA / Basel 的 IRRBB、MA(BS)28 大額風險、CRM 資本分配、AML 可解釋性分析平台 |
| 客戶 | 香港／區域銀行、合規與風險部門 |
| 護城河 | 已經寫好的規則引擎（`le.md`、`irrbb_verification.md` 對照 SPM IR-1 / BCBS d368+d578）＋ 可重演／可審計的設計 |
| 要補 | 真實銀行資料驗證、獨立驗證報告、審計軌跡、部署與運維、**信任**（這一項最貴） |
| 致命風險 | 單人 repo 無法通過銀行的 vendor risk assessment；銷售週期 12–24 個月 |
| 12 個月可行性 | **低（作為產品）／高（作為顧問、求職、合作提案的材料）** |

### C. HFT / kdb+ 替代品（perf-first）

| 項目 | 內容 |
|---|---|
| 一句話 | Rust 寫的 tick 資料引擎：asof join + 滾動窗口 + 點時間快照，價格是 kdb+ 的 1/10 |
| 客戶 | prop shop、量化團隊、想擺脫 kdb+ 授權費的買方 |
| 護城河 | 實測的 fusion / 記憶體牆 / CUDA merge-join 能力，業界少見 |
| 要補 | 完整 tick 攝取（Kafka/交易所格式）、q 語言相容度、穩定的 sub-ms 全鏈路、支援 |
| 致命風險 | 買方極度保守，不會採用 1 star 專案；ArcticDB / QuestDB / kdb+ 已佔位 |
| 12 個月可行性 | **中低**。技術可行，商業不可行（除非有業內人脈背書） |

### 決策矩陣

| | 誰會付錢 | 多快能驗證 | 與現有資產匹配 | 單人可完成 |
|---|---|---|---|---|
| A 嵌入式引擎 | 開發者（付費轉化慢但有） | 3 個月（有 demo 就能判斷） | ★★★★★ | ✅ |
| B 銀行產品 | 銀行（金額大） | 12–24 個月 | ★★★☆☆（規則在，信任不在） | ❌ |
| C HFT 引擎 | 交易團隊 | 6–12 個月 | ★★★★☆ | △（需人脈） |

---

## 5. 建議：選 A，把 B 降級為「參考應用」

### 為什麼

1. **你 80% 的技術價值在 kernel，但 kernel 目前沒有對外接口**。先把它變成別人 `cargo add` 得進來的東西，價值才會被市場定價。
2. **單人專案唯一能規模化的商業模式是「被嵌入」，不是「被部署」**。被部署需要 support、SLA、合規；被嵌入只需要文件、範例、穩定 API。
3. **B 的產出不會浪費**：把它變成 `examples/banking-lab/` 或獨立 repo，是很好的差異化材料（能對著 HKMA 文件逐條驗證，這種能力在面試與顧問提案裡非常值錢），但不該與引擎共用同一個 repo 與敘事。
4. **C 的技術洞察已經寫成文件了**，這是資產；但「賣給 prop shop」這一步不該是現在的重心。

### 建議的產品形狀

```
gtv (workspace)
├── gtv-core       # Arrow schema + TemporalCSR + traversal     ← 對外，穩定
├── gtv-vector     # Metric + Flat/IVF/HNSW + filter routing    ← 對外，穩定
├── gtv-array      # asof / mavg / msum / deltas / fusion       ← 對外，穩定
├── gtv-store      # 磁碟格式 + mmap + WAL（唯一要新寫的）      ← 對外
├── bindings/py    # PyO3：Graph + Vector 兩個 class            ← 對外
└── examples/      # agent-memory、fraud-ring、banking-lab      ← 不對外承諾
```

其餘 15 個 crate：archive 到 `archive/` 分支或獨立 repo。**不是刪掉，是移出主線。**

---

## 6. 90 天行動計劃（若選 A）

### 第 1–2 週：止血（不寫新功能）

- [ ] `git pull`，把本地同步到 `b131a45`；清掉未提交的 `asof.rs` / `window.rs` / `window_simd/`。
- [ ] 根 `README.md` 換成：一句話定位 + 30 秒可跑的 quickstart + 一張架構圖 + 現況聲明（alpha）。加 topics、description、release `v0.1.0-alpha`。
- [ ] CI 補上 `cargo build --workspace` 與 `cargo test -p gtv-core -p gtv-array -p gtv-index -p gtv-storage -p gtv-pattern`，再加 `clippy -- -D warnings`。
- [ ] `ROADMAP.md` 重寫成三欄：**主線（kernel）／參考應用（banking lab）／封存**。

### 第 3–6 週：收斂成一個可安裝的庫

- [ ] 從 24 crate 抽出 4 個對外 crate（見上圖），其餘移到 `archive/`。
- [ ] 凍結公開 API：
      `TemporalGraph::open(path)` / `neighbors(src, at)` / `khop(seeds, k, at)` /
      `VectorIndex::search(query, k, filter)` / `asof_join(left, right)`。
- [ ] 寫 `examples/agent_memory.rs`：10 萬條帶時間的邊 + 1024 維 embedding，查「as-of T、2-hop 內、語義最近」——**這是整個專案的門面 demo**。
- [ ] CSR 增量更新（delta buffer 併入而非全量重建）——目前 `LsmStore::compact_now` 是 O(E log E) 全量重建。

### 第 7–10 週：分發

- [ ] `maturin` + PyO3 綁定（AI 生態在 Python，這一步沒做等於沒發佈）。
- [ ] crates.io `0.1.0`、docs.rs、CHANGELOG、`#![deny(missing_docs)]`。
- [ ] 一組**可重現的對照 benchmark**：同一份資料 vs Neo4j（圖＋時間）／LanceDB（向量＋過濾）／DuckDB（asof）。放進 `benches/`，用 criterion。

### 第 11–13 週：敘事

- [ ] 技術文章一：〈Arrow 上的時間切片 CSR：為什麼 `valid_to` 不能二分〉（zone map 的推導）。
- [ ] 技術文章二：〈熔接與記憶體牆：asof join 從 71 ms 到 2.5 ms〉（已有材料，改寫即可）。
- [ ] 把 banking 面整理成 `banking-lab` 獨立 repo，README 明確寫「reference implementation, not a production risk system」。

---

## 7. 應該停止做的事（kill list）

1. **停止新增 crate。** 24 個已超過單人可維護上限，且每個新 crate 都會讓「這是什麼專案」更難回答。
2. **停止在 DB repo 裡發展 quantum / PQC / CBOM。** 它們本身寫得不錯，但與資料庫核心無關，而且**它們是 CI 唯一照到的地方**——造成「CI 綠＝專案健康」的錯覺。移到獨立 repo。
3. **停止在根目錄堆 `.md`。** 全部進 `docs/`，加 `docs/README.md` 索引與 `status` 標頭。
4. **停止用「kdb+ / KDB.AI parity」當主要賣點。** 玩具規模的 parity 不能證明可替代性，還把最強的對手寫進自己的 tagline。
5. **停止做分散式。** 16 個 issue 全在這裡；單機都還沒產品化，分散式是純負債。
6. **停止把「延遲達標」當作效能驗收。** 每個延遲門檻都要配一個 recall／正確性門檻。

---

## 8. 如果最後選 B 或 C，各自要先補什麼

**選 B（銀行）**：先找一個願意做 pilot 的風險部門（哪怕免費），拿到真實資料與一條可以放進論文的驗證曲線；把 `gtv-largeexposure` / `gtv-scenario` 的規則與 HKMA 條文做成可勾稽的對照表；其餘全部砍掉。**沒有 pilot 客戶就不要寫第 5 萬行業務規則。**

**選 C（HFT）**：先把「單機、單程序、從 tick 到信號 sub-ms」這條鏈打通（目前 asof 是 2.5 ms／1M，但攝取、索引、查詢三段是分開的）；q 語言前端做到能跑真實 `.q` 腳本；找一位業內使用者每天用。**沒有每日使用者就不要做分散式與 UI。**

---

## 9. 最後一句

這個 repo 現在最大的問題不是「做得不夠多」，而是**做得太多而沒有出口**。你已經證明了三件很難的事：能在 Arrow 上把時序圖索引做對、能把記憶體牆撞開、能在五週內把一個監管領域的規則寫到可驗證。接下來該證明的第四件事，是**克制**：把 74,800 行砍到 20,000 行，讓陌生人十分鐘內跑起來。
