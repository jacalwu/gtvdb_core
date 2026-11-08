# gtvdb_core — 核心路线圖

嵌入式時序圖分析平台核心：在一個 Arrow 表上回答「`T` 時刻這張圖長什麼樣」與「這個狀態是怎麼變成的」。
本倉只承諾**內核主線**；領域工作負載仍留在原完整 workspace（`jacalwu/gtvdb` 的 `gtv_ext_lib/`），不進本路線圖。定位理由見 [docs/positioning.md](docs/positioning.md)。

---

## 主線（Mainline）

- [x] **M1 時序核心**：Temporal-CSR（半開區間邊 + zone map 剪枝 + run 內二分）、k-hop、direction-optimizing 走訪
- [x] **M2 SQL 表面**：DataFusion REPL、`as_of` / `neighbors` / `ann` table functions、`mavg`/`msum` WindowUDF
- [x] **M3 向量**：Flat（AVX2+FMA）/ IVF / HNSW，統一 `Metric` 契約、filter-aware 路由 + exact rerank
- [x] **M4 落地**：Parquet 分區 + atomic rename、SnapshotStore（多版本時間旅行）
- [x] **M5 來源圖（state provenance）**：`gtv-core::provenance` + `gtv-core::quantum_state` + `QUANTUM_STATE_*` SQL
- [x] **M6-P0 讀路徑閉環**：GTVSeg v1 定寬可映射格式 + `Col<T>` mmap 後端 + `register_segment()` / `gtv_segments()` / `gtv_segment_edges()` + `bench_segment` 基線。詳 `docs/storage-mmap-design.md` §8
- [x] **M6-P1 寫路徑閉環**：WAL（定寬 42B/邊 + 幀校驗和）+ MemSegment 原生 Vec delta + manifest / flush / crash recovery / 段級復用 / `graph_at(T)` + `register_segment_store()`。寫路徑零編碼零壓縮、不產生也不提交 CSV。詳 `docs/storage-mmap-design.md` §9
- [x] **M11 擴展機制**：`gtv_engine::ext::GtvExtension` + `ExtensionRegistry` + `gtv_extensions()`；領域工作負載移出核心倉
- [x] **M13 核心倉拆分**：從完整 workspace 拆出獨立的 `gtvdb_core`（乾淨歷史、零擴展依賴、可獨立 `cargo test --workspace`），並以 CI 守衛固定邊界
- [ ] **M6 持久化補完（P2–P3）**：窄化 16B / 分層 Skip Index / SIMD 謂詞下推 → compaction / 孤兒段 GC / 冷熱淘汰 / `.gtvidx` mmap 化 / L2 Parquet 歸檔。熱層不壓縮；P1 已吸收「寫路徑與壓縮解耦」「段級復用」，但只用半套 LSM（LSM 寫、CSR 讀）。並發寫入與真實 kill -9 進程級測試仍未做
- [ ] **M7 增量 frontier**：狀態圖的 live branch set 隨 append 維護，取代每次查詢重算
- [ ] **M8 分發**：`gtv-core` / `gtv-index` 發佈 crates.io，`v0.1.0-alpha` release
- [ ] **M9 綁定**：pyo3（只暴露 6 個函式）
- [ ] **M10 canonical demo**：point-in-time 特徵 join、量子 run 除錯各一
- [ ] **M12 動態載入器**：把 dylib 的 C-stable vtable 包成 `GtvExtension`（由擴展側交付）

## 明確不做（核心倉）

| 項目 | 理由 |
|:--|:--|
| 銀行監管 / HFT / 量子求解 / PQC / q 前端 | 工作負載，留在原完整 workspace；核心只保留註冊介面 |
| 分散式（shard / planner / worker） | 目前只是遠端執行 SQL；無真實多機工作負載前不承諾 |
| 提交任何生成資料集 | 歷史教訓：兩個 >50MB CSV blob 曾拖慢每次 clone；改為生成器 + 校驗和 |
| 熱層壓縮 | mmap 後必須能零解碼直接算，見 `docs/oceanbase-storage-lessons.md` |

## Workspace 邊界

```
crates/                    # 平台（本倉全部內容）
├── gtv-core / gtv-array / gtv-index / gtv-index-store / gtv-storage / gtv-delta
├── gtv-pattern / gtv-catalog / gtv-ingest / gtv-udf / gtv-engine / gtv-proto
└── gtv-cli                # 精簡 REPL，零擴展依賴（bin: gtv）
```

`cargo check -p gtv-engine` 不拉進任何領域 crate；CI `core-boundary` job 同時檢查目錄樹、manifest 依賴與 `cargo tree`。
