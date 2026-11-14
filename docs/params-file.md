# gtVdB Parameters File

gtVdB 用一個 Oracle `init.ora` 風格的扁平參數檔控制其行為（記憶體、GPU、預設方言、部署路徑）。檔案由新 crate `crates/gtv-params` 解析；該 crate **不依賴任何 GPU 庫**，只透過 `sysinfo` 探測系統實體記憶體，因此無 GPU 的 CI 也能全綠編譯與測試。

> 倉庫根目錄的 `gtvdb.params.example` 是一份完整、可複製使用的樣本。

## 檔案發現順序

啟動時依序尋找（第一個存在的檔案生效；都不存在則只用工內預設，不報錯）：

1. 環境變數 `GTV_PARAMS_FILE` 指向的路徑（若存在）
2. `./gtvdb.params`（工作目錄）
3. 執行檔所在目錄的 `gtvdb.params`
4. Unix：`/etc/gtvdb/gtvdb.params`；Windows：`C:\ProgramData\gtvdb\gtvdb.params`

## 語法

- 扁平 `KEY=VALUE`；`=` 或空白分隔皆可（`MEMORY_TARGET=16000M` 或 `MEMORY_TARGET 16000M`）。
- 鍵名**全部大寫**、大小寫不敏感（`memory_target` 等同 `MEMORY_TARGET`）。
- `#` 或 `--` 開頭為註解；空行忽略。
- 未知鍵**只記警告、不崩潰**。
- 環境變數 `GTV_PARAM_<KEY>` 優先級最高；`GTV_PARAMS_FILE` 選擇檔案。

### 單位（二進制，大小寫不敏感）

`K` / `M` / `G` / `T`，可尾隨 `B`（`4096M` = 4096 MiB = 4294967296 bytes）。無單位視為 bytes。

### 分數參考

| 寫法 | 意義 |
|:--|:--|
| `SYSTEM:0.5` | 系統實體記憶體的一半 |
| `TARGET:0.5` | `MEMORY_TARGET` 的一半 |
| `DEVICE:0.8` | CUDA 裝置 VRAM 總量的 80% |

`AUTO` / `SYSTEM` 亦可用於部分純量鍵。

## 參數一覽

| 鍵 | 預設 | 說明 |
|:--|:--|:--|
| `MEMORY_TARGET` | `SYSTEM:0.5`（系統記憶體一半） | gtVdB 可分配的總記憶體。**僅當顯式設定時**才在引擎掛載 DataFusion `GreedyMemoryPool`（避免小 CI 機被 auto-half 卡死）。 |
| `MEMORY_OBJ_AREA` | `TARGET:0.5`（`MEMORY_TARGET` 一半） | 可常住記憶體（resident-in-memory）object 的記憶體區域。必須 `≤ MEMORY_TARGET`，否則報錯 `ObjAreaExceedsTarget`。 |
| `CUDA_MODE` | `AUTO` | `OFF` 永不開裝置；`ON` 無裝置則啟動報錯；`AUTO` 有裝置才開。引擎在**無裝置時會把 `ON` 降級為 `OFF`** 以保證 GPU-less 機器仍能開機。 |
| `CUDA_VRAM` | `DEVICE:0.8`（裝置 VRAM 80%） | CUDA 啟用時的顯存預算。`CUDA_MODE=OFF` 時為 0 / 無裝置。 |
| `CUDA_DEVICE` | `0` | 選用的 CUDA 裝置序號。 |
| `ACCEL_CACHE` | `ON` | 裝置常駐快取：`ON` / `OFF` / `SHADOW`（只記帳不碰 GPU，用於買卡前以真實負載標定）。 |
| `ACCEL_SYNC_THRESHOLD` | `CUDA_VRAM` 的 25% | 待同步的 host→device 增量達此量即批次上傳；`AUTO` 為預設值，也可寫具體大小如 `512M`。 |
| `ACCEL_SYNC_INTERVAL` | `5` | 距離上次上傳超過此秒數即套用增量（即便未達門檻）；`0` 表示每個批次邊界都上傳。 |
| `ACCEL_HEAT_HALF_LIFE` | `30` | 常駐區塊「熱度」的半衰期（秒）。熱度＝經衰減的存取頻率，用於准入與淘汰：一次性的大掃描不會把熱門 corpus 擠下卡，被拒絕的 corpus 仍可在幾次查詢後爭取到位置。`0` 關閉熱度、回到純 LRU。 |
| `DEFAULT_MODE` | `SQL` | 預設方言：`SQL` 或 `Q`。`Q` 在 quantum/q 擴展未安裝時回退為完整 SQL 並提示。 |
| `LOG_LEVEL` | `INFO` | `ERROR`/`WARN`/`INFO`/`DEBUG`/`TRACE`，對應 `RUST_LOG` 過濾詞彙。CLI 在 `RUST_LOG` 未設時據此設定。 |
| `SPILL_DIR` | `<tmp>/gtv-spill` | 落盤溢寫目錄。環境變數 `GTV_SPILL_DIR` 優先級高於檔案。 |
| `WORKER_THREADS` | `AUTO` | 顯式整數或 `AUTO`（由 `GTV_TARGET_PARTITIONS` / CPU 數決定）。 |
| `DATA_DIR` | `./data` | 資料目錄。 |
| `CATALOG_DIR` | `./catalog` | 目錄 / 中繼資料目錄。 |
| `TELEMETRY` | `FALSE` | 遙測開關（`TRUE`/`FALSE`/`1`/`0`/`yes`/`no`/`on`/`off`）。 |

> 既有環境變數約定保留並補充：`GTV_TARGET_PARTITIONS`、`GTV_BATCH_SIZE`、`GTV_SPILL_DIR` 仍生效，`GTV_SPILL_DIR` 優先級高於參數檔 `SPILL_DIR`。

## 樣本（來自使用者需求）

```ini
# gtVdB parameters — Oracle-style flat file
MEMORY_TARGET=16000M      # 總可分配記憶體
MEMORY_OBJ_AREA=8000M     # 常住記憶體 object 區域（= target 一半）
CUDA_MODE=TRUE            # 啟用 GPU 加速
CUDA_VRAM=6000M           # GPU 顯存預算（80% 裝置 VRAM 時可寫 DEVICE:0.8）
```

## 三段式解析

1. **`GtvParams`** — 檔案原始值 + 預設。
2. **`GtvParams::resolve(device_total: Option<u64>)`** — 給定可選的 CUDA 裝置 VRAM 總量（由呼叫方注入，使本 crate 不含 GPU 依賴），算出具體位元組數與選定模式，並做 `obj_area ≤ target`、CUDA 裝置可用性校驗。
3. **`ResolvedParams`** — 引擎實際消費的具體值；提供 `summary_lines()` 供 CLI `.params` 指令列印穩定格式摘要。

## 引擎 / CLI 接線

- 引擎：`GtvContext::with_params(params)` 與 `GtvContext::with_extensions_and_params(ext, params)`。`ctx.params()` / `ctx.resolved()` / `ctx.default_mode()` 可存取。建構時：顯式 `MEMORY_TARGET` 才掛記憶體池；套用 `WORKER_THREADS` 並以 `GTV_SPILL_DIR` 覆寫溢寫目錄；`CUDA_MODE=ON` 無裝置時降級為 `OFF` 續跑。
- CLI（`gtv`）：啟動即 `GtvParams::load()`，列印參數警告；`RUST_LOG` 未設時依 `LOG_LEVEL` 設定並初始化 `tracing` 訂閱者；以 `with_extensions_and_params` 建上下文；`DEFAULT_MODE` 決定初始 REPL 方言；新增 `.params` 指令列印 `resolved().summary_lines()`。

## 與 GPU 加速的關係（重要）

核心引擎已掛載 `gtv-accelerator`（`--features cuda` 啟用裝置後端；預設建置只走宿主 kernel）。引擎在解析參數時呼叫 `device_total_bytes()` 取得裝置 VRAM，`CUDA_VRAM=DEVICE:0.8` 才會換算成具體位元組；無 GPU 時該函式回傳 `None`，`ON` 降級為 `OFF`，啟動不會失敗。詳 `docs/acceleration.md`。

`CUDA_VRAM` 是**上限而非承諾**：裝有裝置時，准入前會以 `cudaMemGetInfo` 讀取實際空閒顯存（每 250 ms 至多一次），把可用容量壓到 `min(CUDA_VRAM, 實測空閒 + 已佔用)`。這避免 Windows WDDM 把配置悄悄落到 host-backed sysmem —— 那種配置不會報錯，但會讓 kernel 頻寬崩掉。`.accel` 會顯示實測值與被壓縮的次數。

`.params` 印出解析後的鍵值，`.accel` 印出裝置、熱度、准入拒絕、上傳與淘汰的即時帳本。
