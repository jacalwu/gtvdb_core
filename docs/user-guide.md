# gtvdb 用户手册

> **适用基线**：`main` @ `58013c3`（2026-11-15）
> **本手册的每一条命令都在本机实跑过**，贴的输出是真实终端结果（不是示意）。跑不通的命令不写进来。
> 想先知道「这是什么产品」，看 [`product-design.md`](product-design.md)；想知道「内部怎么实现」，看 [`design-review-2026-11.md`](design-review-2026-11.md)。

---

## 0. 菜单：我想…… → 看哪节

| 我想…… | 去哪 |
|:--|:--|
| 5 分钟内跑出第一条查询 | §2 |
| 知道该用 REPL 还是写 Rust | §1 |
| 用 Python 跑（`pip install gtvdb` 装的是驱动，不含引擎） | §12 |
| 按时间点查「这个节点当时连到谁」 | §3 |
| 找环 / 找 pattern（AML 洗钱环、担保链） | §3 |
| 做向量检索、按标签过滤 | §4 |
| 打开一个大文件但不解压、不导入 | §5 |
| 查数据质量、新鲜度、有无重复 | §6 |
| 回答「这个状态是怎么变成的」 | §7 |
| 查某条命令的语法 | §8 |
| 调内存 / 开 GPU / 省显存 | §9 |
| 报错了不知道啥意思 | §10 |
| 想避免踩坑（数据、基准、边界） | §11 |

---

## 1. 先选轨道：三条用法，能力不同

gtvdb 是**嵌入式**引擎，不是一个等你连的服务。用之前先认清自己在哪条轨道上——很多人「功能找不到」其实是选错了轨道。

| 轨道 | 怎么用 | 能做什么 | 不能做什么 |
|:--|:--|:--|:--|
| **A. REPL**（零代码） | `cargo run -p gtv-cli --bin gtv` | 段 mmap 冷开、CSV/Parquet 直读、向量索引冷开、量子状态图、DQ/健康检查、参数与加速层遥测 | **图遍历**（`neighbors`/`khop`/`motif`）、双时间、asof join、embedding 检索 |
| **B. Rust 嵌入** | `GtvContext::new()` + `register_*` + `ctx.sql(...)` | 上面全部 + 图遍历 + 双时间 + asof join + 治理集合 | 没有常驻服务端；不能多进程并发写 |
| **C. 扩展仓** | `gtvdb_ext`（另一仓） | 银行监管 / HFT / 量子求解等领域功能 | 不属于本仓，不在本手册范围 |

**为什么图遍历不在 REPL 里**：`neighbors()` 需要一个已经在内存里建好索引的 `TemporalCSR`，而这个索引必须由宿主用 `register_neighbors(&csr)` 挂上去；核心 REPL 目前没有「从 CSV 建图」的指令。这不是 bug，是设计边界——见 §3.1 的绕法。

> **Python 属于哪条轨道？** 轨道 A。本仓没有 Python 绑定，Python 只能当客户端去驱动 `gtv` 进程（`examples/python/gtv_repl.py`），所以它的能力与限制**和 REPL 完全一样**——包括"图遍历不可达"。详见 §12。

---

## 2. 五分钟跑起来

### 2.1 前置

- Rust 稳定版（edition 2021）；Windows 需 MSVC 工具链
- 首次构建会编译 DataFusion + Arrow，约 1–3 分钟

```bash
git clone https://github.com/jacalwu/gtvdb_core && cd gtvdb_core
cargo build -p gtv-cli          # 产物：target/debug/gtv
```

### 2.2 第一条查询（仓库自带数据，不用准备任何文件）

```bash
cargo run -p gtv-cli --bin gtv
```

```text
gtv> SELECT * FROM quantum_state_summary('examples/quantum-state/ghz3.jsonl');
+--------+-------------+-------+--------+-----------+--------------------+-------------+
| states | transitions | roots | leaves | max_depth | mean_branching     | semantics   |
+--------+-------------+-------+--------+-----------+--------------------+-------------+
| 10     | 9           | 1     | 3      | 6         | 1.2857142857142858 | Conditional |
+--------+-------------+-------+--------+-----------+--------------------+-------------+
```

这就是「一个 GHZ-3 电路在中间加了一次测量之后，状态长成了一棵树而不是一条链」。§7 会解释怎么顺着这棵树往回追。

### 2.3 先看环境

```text
gtv> .params
  MEMORY_TARGET    = 16094 MiB    (auto: system/2)
  MEMORY_OBJ_AREA  = 8047 MiB     (auto: target/2)
  CUDA_MODE        = OFF
  CUDA_VRAM        = 0 MiB        (no device)
  ACCEL_CACHE      = OFF          (sync: 0 MiB / 5s)
  ACCEL_HEAT      = 30s half-life
  ACCEL_PREFETCH   = OFF
  ACCEL_GPUDIRECT  = OFF
  DEFAULT_MODE     = SQL
  LOG_LEVEL        = INFO
  SPILL_DIR        = C:\Users\...\AppData\Local\Temp\gtv-spill
  WORKER_THREADS   = auto
  TELEMETRY        = false

gtv> .accel
  ACCEL_CACHE      = OFF
  DEVICE           = cpu (cpu)
  VRAM_TOTAL       = 0 MiB
  BUDGET           = 0 MiB / 0 MiB (0.0% used)
  REQUESTS         = 0 (0 device, 0 host, 0 fallback, 0 cache-declined)
  RESIDENT         = 0 blocks, hit ratio 0.0% (0 hits / 0 misses)
  UPLOADS          = 0 (0 MiB transferred)
  ...
```

> `CUDA_MODE=OFF` 不代表你配错了：参数档默认是 `AUTO`，**探不到设备就降级为 OFF**，启动永不依赖 GPU。

### 2.4 `.help` 全文

```text
Core commands
  .help                     show this help
  .quit | .exit             leave the shell
  .tables                   list registered DataFusion tables
  .extensions               SELECT name, version, objects FROM gtv_extensions()
  .segment   <name> <file>  mmap a GTVSeg v1 file as <name> (O(1), no decode)
  .store     <prefix> <dir> <node_count>
                            register every segment published by a P1 SegmentStore
  .csv       <name> <file>  register a CSV as table <name>
  .parquet   <name> <path>  register a Parquet file / directory as <name>
  .knn-index <name> <file> [labels.txt]
                            cold-open a vector index from <file> for knn(name,query,k[,label])
  .ann-index <name> <file> cold-open a vector index from <file> for ann(name,query,k[,metric])
  .knn-save  <name> <store-dir>
                            persist the in-memory knn collection <name> into an IndexStore
  .params                   show the effective parameter file
  .accel                    show device acceleration: VRAM budget, resident cache, ...
```

### 2.5 路径怎么写（Windows 用户第一个坑）

REPL 是**原生 Windows 程序**，它不认 Git Bash / MSYS 的 `/tmp/...`。用原生路径：

```text
✅ .csv edges C:/Users/me/AppData/Local/Temp/edges.csv
❌ .csv edges /tmp/edges.csv          → io error: The system cannot find the path specified
```

正斜杠 `/` 或反斜杠 `\\` 都行，但**不要**用 `/c/...`、`/tmp`、`~`。

---

## 3. 图：时间切片、遍历、找环

### 3.1 关键事实（先看这段，否则会卡住）

`neighbors()` / `khop()` / `motif()` 都需要先由宿主把图挂上来：

```rust
ctx.register_neighbors(&csr);   // neighbors(), khop()
ctx.register_motif(&csr);       // motif(), motif_from()
ctx.register_batch_graph(&csr, &seeds, valid_at, k);  // gtv_batch_neighbors / gtv_batch_khop
```

**在裸 REPL 里直接查会报错**：

```text
gtv> SELECT * FROM neighbors(0, 16000000000);
error: Error during planning: table function 'neighbors' not found
```

这不是缺功能，是缺「把图放进内存」这一步。下面 3.2 的示例把它做完。

### 3.2 跑官方示例：一张 6 条边的小图

```bash
cargo run -p gtv-engine --example hello_graph
```

源文件 [`crates/gtv-engine/examples/hello_graph.rs`](../crates/gtv-engine/examples/hello_graph.rs)，图是这样定义的（`valid_from`/`valid_to` 单位是**纳秒**，区间**半开** `[from, to)`）：

| src | dst | type | 有效区间（秒） |
|--:|--:|--:|:--|
| 0 | 1 | 1 转账 | [10, 20) |
| 1 | 2 | 1 转账 | [11, 21) |
| 2 | 0 | 1 转账 | [12, 22) | ← 与上两条构成 3 环
| 3 | 0 | 2 担保 | [10, 20) |
| 0 | 4 | 1 转账 | [15, 25) |
| 4 | 5 | 1 转账 | [16, 26) |

真实输出：

```text
=== ① 时间切片：16s 时节点 0 的出边（2 条都在有效期内） ===
SELECT src, dst, edge_type, valid_from / 1000000000 AS from_s, valid_to / 1000000000 AS to_s
FROM neighbors(0, 16000000000)
+-----+-----+-----------+--------+------+
| src | dst | edge_type | from_s | to_s |
+-----+-----+-----------+--------+------+
| 0   | 1   | 1         | 10     | 20   |
| 0   | 4   | 1         | 15     | 25   |
+-----+-----+-----------+--------+------+

=== ② 半开区间：20s 时 [10,20) 那条已失效，只剩 [15,25) ===
+-----+-----+--------+------+
| src | dst | from_s | to_s |
+-----+-----+--------+------+
| 0   | 4   | 15     | 25   |
+-----+-----+--------+------+

=== ③ 两跳可达：16s 时从 0 出发走 2 步 ===
+-----+-----+
| hop | dst |
+-----+-----+
| 1   | 1   |
| 1   | 4   |
| 2   | 2   |
| 2   | 5   |
+-----+-----+

=== ④ 3 环检测：16s 时全图的 ring3（0->1->2->0） ===
+----------+----+----+----+-------+-------+
| match_id | n0 | n1 | n2 | t_min | t_max |
+----------+----+----+----+-------+-------+
| 0        | 0  | 1  | 2  | 10    | 12    |
+----------+----+----+----+-------+-------+

=== ⑤ 批量表：注册时给定种子 {0,3}，一次规划出全部结果 ===
SELECT seed, hop, dst FROM gtv_batch_khop ORDER BY seed, hop, dst
+------+-----+-----+
| seed | hop | dst |
+------+-----+-----+
| 0    | 1   | 1   |
| 0    | 1   | 4   |
| 0    | 2   | 2   |
| 0    | 2   | 5   |
| 3    | 1   | 0   |
| 3    | 2   | 1   |
| 3    | 2   | 4   |
+------+-----+-----+
```

**四件事要记住**：

1. **时间单位是纳秒**（`i64`）。SQL 里传 `16000000000` 就是 16 秒。想按秒看，除以 `1000000000`。
2. **区间半开**：`[10,20)` 在 `t = 20s` 已经不成立。②就是这条的证明。
3. **`motif()` 的返回列随形状变**：`ring3` 给 `n0,n1,n2`；`ring4` 给 `n0..n3`；`path3` 给 `n0,n1,n2`。列数 = 形状变量数 + `match_id` + `event_min/max`。
4. **批量表的种子集是注册时绑定的**，不是 SQL 参数。想换种子要重新 `register_batch_graph`；换时间窗也一样。

### 3.3 只想看「某个环节的证据边」：`gtv_motif_edges`

`gtv_motif_matches` 一行一个 match，`gtv_motif_edges` 一行一条**匹配到的边**，用来回答「这条环具体走了哪几条转账」：

```sql
SELECT * FROM gtv_motif_edges WHERE match_id = 0 ORDER BY edge_pos;
```

列：`match_id`、`edge_pos`（路径内第几条，0 起）、`edge_idx`（CSR 槽位，用它回查原始记录）、`src`、`dst`、`edge_type`、`valid_from`、`valid_to`。

> `edge_pos` 与 `edge_idx` 是**两个不同的数**：前者是路径位置，后者是 CSR 数组下标。曾经有个版本把前者当后者发布过，现在分开了。

### 3.4 图函数速查

| 函数 | 参数 | 返回列 | 备注 |
|:--|:--|:--|:--|
| `neighbors(src, valid_at)` | 字面量 | `src, dst, edge_type, valid_from, valid_to` | 出边 |
| `khop(src, k, valid_at)` | 字面量 | `hop, dst` | `hop` 从 1 开始 |
| `motif(shape, valid_at [, limit [, window]])` | 字面量 | `match_id, n0.., event_min, event_max` | `shape` ∈ `ring3 ring4 path2 path3 path4 diamond` |
| `motif_from(shape, seed, valid_at [, limit [, window]])` | 字面量 | 同上 | 单种子锚定 |
| `gtv_batch_neighbors` | 注册时绑定 | `src, dst, edge_type, valid_from, valid_to` | 可 join / 过滤 |
| `gtv_batch_khop` | 注册时绑定 | `seed, hop, dst` | 同上 |

**只接受字面量**是刻意的：把 `src` 换成列名会直接报错，因为逐行重规划比图内核慢几个数量级。要按一批种子查，用批量表或 Rust 循环。

---

## 4. 向量检索

### 4.1 跑官方示例

```bash
cargo run -p gtv-cli --example hello_vector
```

源文件 [`crates/gtv-cli/examples/hello_vector.rs`](../crates/gtv-cli/examples/hello_vector.rs)。语料 4 条二维向量，标签 `jazz`/`jazz`/`rock`/`rock`：

```text
=== ① 最近邻：query 是逗号分隔的字符串，不是数组字面量。最近的是 rock 的 id2 ===
SELECT * FROM knn('songs', '0.1,0.9', 2)
+----+---------------------+
| id | distance            |
+----+---------------------+
| 2  | 0.14142137441313676 |
| 0  | 0.9055384769532424  |
+----+---------------------+

=== ② 标签过滤：只在 jazz 里找（第 4 个参数）—— id2 被排除，结果完全不同 ===
SELECT * FROM knn('songs', '0.1,0.9', 2, 'jazz')
+----+--------------------+
| id | distance           |
+----+--------------------+
| 0  | 0.9055384769532424 |
| 1  | 1.2727921611791464 |
+----+--------------------+

=== ③ 换距离：同一份语料再用 cosine 注册一份，排序和 L2 不同 ===
SELECT * FROM knn('songs_cosine', '0.9,0.1', 2)
+----+----------------------+
| id | distance             |
+----+----------------------+
| 1  | 0.006116271018981934 |
| 3  | 0.2191312313079834   |
+----+----------------------+
```

**三个坑**：

1. **query 是字符串** `'0.1,0.9'`，不是 `[0.1, 0.9]`，也不是 `ARRAY[...]`。写成数组会报 `expected a string literal`。
2. **metric 在注册时定**，不是查询参数。想要 cosine 就 `register_knn_metric(..., Metric::Cosine)` 另注册一份。
3. **`distance` 的口径**：L2 报**欧氏距离**（已经开过平方根）；cosine 报 `1 - cos`；内积报 `-dot`。

### 4.2 落盘 → REPL 冷开（同一份语料，不重解码）

示例最后会打印一个 `.gtvidx` 路径。拿它进 REPL：

```text
gtv> .knn-index songs C:/Users/me/AppData/Local/Temp/gtv-hello-vector/<uuid>/v2/index.gtvidx
cold-opened `...` as knn collection `songs` (mmap-backed, no re-decode)

gtv> SELECT * FROM knn('songs', '0.1,0.9', 2);
+----+---------------------+
| id | distance            |
+----+---------------------+
| 2  | 0.14142137441313676 |
| 0  | 0.9055384769532424  |
+----+---------------------+
```

同一个文件也能用 `ann()` 表面打开：

```text
gtv> .ann-index songs2 <同一个文件>
cold-opened `...` as ann index `songs2` (mmap-backed, no re-decode)

gtv> SELECT * FROM ann('songs2', '0.1,0.9', 2);
```

**冷开是 O(1)**：1e6 × 128 的语料，`885 ms`（重解码）→ `0.85 ms`（mmap），常驻内存从 1487.7 MiB 降到 495.8 MiB。

### 4.3 三个向量表面怎么选

| 表面 | 保证 | 过滤方式 | 什么时候用 |
|:--|:--|:--|:--|
| `knn(name, query, k [, label])` | **精确**（Flat 全扫 / IVF 按 `nprobe` 探测） | 标签字符串 | 要精确答案、要标签过滤、语料能进显存 |
| `ann(name, query, k [, metric])` | 近似 + exact rerank | id allow-list（`'1,2,3'`） | HNSW、大语料、能接受召回率换延迟 |
| `embedding_search(name, query, k [, tenant [, as_of]])` | 精确（治理集合） | 租户 + 时间窗 | 要 provenance、租户隔离、过期语义 |

`knn()` **不支持 HNSW**（会明确让你改用 `ann()`）。`ann_explain` / `cbo_explain` 可以看它选了哪条路。

### 4.4 带标签冷开

标签不进索引文件，用侧车文件**按原始输入序**一行一个标签：

```text
gtv> .knn-index songs <索引文件> labels.txt
```

> 顺序是**原始输入序**，不是 IVF 的 cell 序——`orig_pos` 随 `GIVFCMP` 持久化，引擎会自动处理，你只要保证与当初喂给 `register_knn` 的顺序一致。

---

## 5. 段与文件：不导入、不解压，直接映射

### 5.1 打开一个段

```text
gtv> .segment hello C:/Users/me/AppData/Local/Temp/gtv-hello-graph.seg
mmap'd GTVSeg `...` as `hello` — no decode, no copy

gtv> SELECT name, backend, node_count, edge_count, file_bytes FROM gtv_segments();
+-------+---------+------------+------------+------------+
| name  | backend | node_count | edge_count | file_bytes |
+-------+---------+------------+------------+------------+
| hello | mmap    | 6          | 6          | 4544       |
+-------+---------+------------+------------+------------+

gtv> SELECT src, dst, edge_type, valid_from / 1000000000 AS from_s
     FROM gtv_segment_edges('hello') ORDER BY src, valid_from;
+-----+-----+-----------+--------+
| src | dst | edge_type | from_s |
+-----+-----+-----------+--------+
| 0   | 1   | 1         | 10     |
| 0   | 4   | 1         | 15     |
| 1   | 2   | 1         | 11     |
| 2   | 0   | 1         | 12     |
| 3   | 0   | 2         | 10     |
| 4   | 5   | 1         | 16     |
+-----+-----+-----------+--------+
```

`gtv_segments()` 全列：`name, path, backend, node_count, edge_count, chunk_count, chunk_size, file_bytes, all_active_from, all_active_to, valid_from_min, valid_to_max, node_id_min, node_id_max, src_id_min, src_id_max, prunable`。

> `gtv_segment_edges` 是**物化**路径（把边变成 Arrow 表给你看/导出），不是零拷贝遍历。真要遍历请走 §3 的 `neighbors()`。

### 5.2 一整个 store 的段

```text
gtv> .store g /path/to/store 100000
```

把 P1 `SegmentStore` 发布过的每个段注册成 `<prefix>_<段号>`，文件布局 `<dir>/...`。这一步同时会同步 L2 聚合视图（`gtv_range_agg` 依赖它）。

### 5.3 CSV / Parquet

```text
gtv> .csv edges C:/Users/me/AppData/Local/Temp/edges.csv
registered CSV `...` as table `edges`

gtv> .tables
edges
```

也可以不注册，一次性读：

```sql
SELECT count(*) FROM read_csv('C:/.../edges.csv');
SELECT * FROM read_parquet('C:/.../data.parquet');
```

⚠️ **`.csv` 只给你一张 DataFusion 表，不等于「有了图」**。`neighbors()` 仍然不可用（见 §3.1）。CSV 表的用途是：直接用 SQL 过滤 / 聚合 / join，或者作为 `dq_report` 这类治理函数的输入。

---

## 6. 元数据与数据质量

### 6.1 DQ 报表（要求表里有 `ts` 列）

```text
gtv> .csv tx C:/Users/me/AppData/Local/Temp/tx.csv
gtv> SELECT * FROM dq_report('tx');
+-------+------+-----------+---------+------------+---------------+--------------+--------------+
| table | rows | n_symbols | nan_inf | dup_ts_sym | non_monotonic | min_rows_sym | max_rows_sym |
+-------+------+-----------+---------+------------+---------------+--------------+--------------+
| tx    | 3    | 1         | 0       | 0          | 0             | 3            | 3            |
+-------+------+-----------+---------+------------+---------------+--------------+--------------+

gtv> SELECT * FROM health_check('tx');
+---------------+--------+-----------------------------------+
| check         | status | detail                            |
+---------------+--------+-----------------------------------+
| freshness     | warn   | last bar 20772 days ago (limit 7) |
| row_count     | warn   | 3 rows (min 50)                   |
| no_nan        | ok     | 0 non-finite float values         |
| no_duplicates | ok     | 0 duplicate (ts,symbol)           |
| ts_sorted     | ok     | 0 backwards jumps                 |
+---------------+--------+-----------------------------------+

gtv> SELECT * FROM dq_check('tx');
+-------+-------+
| issue | count |
+-------+-------+
```

三个函数的分工：`dq_report` 看**形状**，`health_check` 看**五个体检项**（新鲜度 / 行数 / NaN / 重复 / 时序），`dq_check` 只列**问题**。`ts` 是纳秒整数——`ts=1000` 就是 1970 年，所以上面 freshness 报 warn 是正常的。

发布门禁是另一个函数 `dq_gate('table', '<rules_json>')`，失败会返回具体规则、列、观测值、阈值。

### 6.2 目录与提交日志

这些函数读的是**磁盘上的 catalog / manifest**，需要宿主先挂上：

| 函数 | 需要 | 裸 REPL 表现 |
|:--|:--|:--|
| `gtv_tables()` / `gtv_snapshots()` / `gtv_files()` | `attach_catalog(store)` | `error: gtv_tables(): no catalog attached` |
| `gtv_invalidations()` / `gtv_tags()` | 同上 | 返回空 |
| `gtv_active_edge_count()` / `gtv_range_agg()` | `register_segment_store()` 建立聚合视图 | 报无视图 |

也就是说：**这三个 L1 函数在 REPL 里查不到东西是正常的**，不是数据坏了。

---

## 7. 量子状态图：回答「这个状态是怎么来的」

一条状态日志是 JSONL，每行 `state_id / parent_state / gate / timestamp / probability`（`probability` 是**条件**概率，绝对概率 = 沿根路径连乘）。仓库自带一份 GHZ-3：

### 7.1 先看森林形状

```text
gtv> SELECT * FROM quantum_state_summary('examples/quantum-state/ghz3.jsonl');
+--------+-------------+-------+--------+-----------+--------------------+-------------+
| states | transitions | roots | leaves | max_depth | mean_branching     | semantics   |
+--------+-------------+-------+--------+-----------+--------------------+-------------+
| 10     | 9           | 1     | 3      | 6         | 1.2857142857142858 | Conditional |
+--------+-------------+-------+--------+-----------+--------------------+-------------+
```

### 7.2 某一时刻还活着哪些分支

```text
gtv> SELECT * FROM quantum_state_frontier('examples/quantum-state/ghz3.jsonl', 6, 0.05)
     ORDER BY probability DESC;
+----------+--------------+-----------+-------+-------------+--------------------------+----------------+
| state_id | gate         | timestamp | depth | probability | probability_renormalized | explained_mass |
+----------+--------------+-----------+-------+-------------+--------------------------+----------------+
| 9        | x q[2]       | 5         | 5     | 0.5         | 0.5                      | 1.0            |
| 7        | measure q[1] | 6         | 6     | 0.4         | 0.4                      | 1.0            |
| 8        | measure q[1] | 6         | 6     | 0.1         | 0.1                      | 1.0            |
+----------+--------------+-----------+-------+-------------+--------------------------+----------------+
```

参数：`(log_path, at [, min_prob])`。`at` 是**时间步**（不是秒），`min_prob` 过滤小概率分支。

### 7.3 某条稀有分支到底经过了哪些门

想象「为什么会有 0.1 的分支」这种问题——不 dump statevector，直接查：

```text
gtv> SELECT step, gate, prefix_probability
     FROM quantum_state_reconstruct('examples/quantum-state/ghz3.jsonl', 8);
+------+-------------------+--------------------+
| step | gate              | prefix_probability |
+------+-------------------+--------------------+
| 0    | |000>             | 1.0                |
| 1    | h q[0]            | 1.0                |
| 2    | cx q[0],q[1]      | 1.0                |
| 3    | cx q[0],q[2]      | 1.0                |
| 4    | measure q[0]      | 0.5                |
| 5    | rx(0.785398) q[1] | 0.5                |
| 6    | measure q[1]      | 0.1                |
+------+-------------------+--------------------+
```

一眼看出：0.1 是「先以 0.5 概率测到某个结果，再在 0.8 的分支上被 `measure q[1]` 选中」乘出来的。

### 7.4 边界（别用错）

**存的是分支结构与概率，不是振幅。** 因此：
- ✅ 能回答「这条分支经过哪些门、概率多少」
- ❌ **不能**算干涉项、**不能**重建完整 statevector

要振幅只能自己存 blob 并把引用放在这里。把这条写进文档是刻意的——写成「量子数据库」第一轮技术尽调就会崩。

---

## 8. REPL 指令参考

| 指令 | 参数 | 作用 | 备注 |
|:--|:--|:--|:--|
| `.help` / `?` | — | 帮助 | |
| `.quit` / `.exit` | — | 退出 | Ctrl-D 同效 |
| `.tables` | — | 列已注册表 | 只含 DataFusion 表，不含段/向量集合 |
| `.extensions` | — | 已装载扩展 | 核心构建下为空表 |
| `.segment` | `<name> <file>` | mmap 一个 GTVSeg | O(1)，不解码；同名重复注册会覆盖 |
| `.store` | `<prefix> <dir> <node_count>` | 注册整仓段 + 同步聚合视图 | 段名 `<prefix>_<id>` |
| `.csv` | `<name> <file>` | 注册 CSV | 只是表，不是图 |
| `.parquet` | `<name> <path>` | 注册 Parquet 文件或目录 | |
| `.knn-index` | `<name> <file> [labels.txt]` | 冷开向量索引到 `knn()` | Flat / IVF；HNSW 用 `.ann-index` |
| `.ann-index` | `<name> <file>` | 冷开索引到 `ann()` | Flat / IVF / HNSW |
| `.knn-save` | `<name> <store-dir>` | 把内存集合存成 `.gtvidx` | 需该集合已在宿主里注册 |
| `.params` | — | 打印生效参数 | 含 auto 推导值 |
| `.accel` | — | 加速层遥测 | 无 GPU 时显示 `cpu (cpu)` |

其余输入一律当 SQL 执行。

---

## 9. 参数与调优

参数来自三处（后者覆盖前者）：`gtvdb.params.example` 复制成 `gtv.params` → 环境变量 `GTV_PARAM_<KEY>` → `GtvContext::with_params`。

### 9.1 常见场景怎么配

| 场景 | 建议 |
|:--|:--|
| 小内存机器 / 容器 | `MEMORY_TARGET=2G`、`MEMORY_OBJ_AREA=TARGET:0.5`。**只有显式写了才强制**，否则用「系统一半」但不强制 |
| 想用 GPU | `CUDA_MODE=ON`、`CUDA_VRAM=DEVICE:0.8`（拿 80% 显存）；`ACCEL_CACHE=ON` |
| 有 GPU 但想先看命中率 | `CUDA_MODE=ON` + `ACCEL_CACHE=SHADOW`（影子账本，不真占显存） |
| 反复查同一批语料 | `ACCEL_PREFETCH=ON`、`ACCEL_PREFETCH_BLOCKS=1`（每轮 1 块） |
| 明确不要 GPU | `CUDA_MODE=OFF`（同时否决 `ACCEL_GPUDIRECT`，没有设备的直传不是一件事） |
| 大排序容易 OOM | 设 `SPILL_DIR` 到空间大的盘（或 `GTV_SPILL_DIR` 环境变量） |

### 9.2 `.accel` 逐行怎么读

| 行 | 含义 |
|:--|:--|
| `DEVICE = cpu (cpu)` | 后端；`cpu` 表示走主机路径，不是报错 |
| `BUDGET = 0 MiB / 0 MiB` | 已用 / 总额度；弹性预算**只缩不胀** |
| `REQUESTS = 0 (0 device, 0 host, 0 fallback, 0 cache-declined)` | 括号里依次是：设备执行、主机执行、失败回退、**被准入拒绝** |
| `RESIDENT = 0 blocks, hit ratio` | 常驻块数与命中率；`ACCEL_CACHE=SHADOW` 时也会累计 |
| `UPLOADS` / `EVICTIONS` | 传了多少 MiB、淘汰了多少 |
| `HEAT` | 最热块的衰减热度、因热度不足被推迟的块数（输出用 `/` 分隔，形如 `hottest 0.00 / 0 heat-deferred / 0 promoted`） |
| `PREFETCH` | 页面预热提示数；heat pump 每轮成功 / 被拒的块数（`pump N ok / M refused`） |
| `GPUDIRECT = n/a (host backend)` | 直传计数；**当前构建永远报 `n/a` 或 0 direct**，因为没编入 `cufile` |

---

## 10. 报错速查

| 报错原文 | 原因 | 怎么修 |
|:--|:--|:--|
| `table function 'neighbors' not found` | 没注册 CSR | §3.1：跑 `hello_graph`，或在宿主里 `register_neighbors` |
| `arrow error: Io error: The system cannot find the path specified. (os error 3)` | 用了 MSYS 路径 `/tmp/...` | 改成 `C:/...` 原生路径（§2.5） |
| `table has no Int64 ts column` | DQ / health 需要 `ts` 整数列 | 加一列 `ts`（纳秒），或换个函数 |
| `gtv_tables(): no catalog attached (call GtvContext::attach_catalog first)` | 没挂 catalog | 裸 REPL 查不到是正常的（§6.2） |
| `knn: query dim 3 != index dim 2` | query 维度与语料不符 | 数一下逗号分隔的分量个数 |
| `knn: bad query component 'x'` | query 字符串里有非数字 | 只能是 `0.1,0.2` 这种 |
| `unknown command '.xxx' — type .help` | 指令拼错 | §8 表 |
| `table 'datafusion.public.xxx' not found` | 表没注册 | 先 `.csv` / `.parquet` / `.segment` |
| `Hnsw` 相关错误（`knn()` 上） | `knn` 不支持 HNSW | 用 `.ann-index` + `ann()` |
| 启动时 `parameter resolution warning: ...; continuing with CUDA disabled` | 参数解析失败但不致命 | `.params` 看实际生效值 |

---

## 11. 三条使用纪律

1. **不要往仓库里放数据**。`*.csv` / `*.seg` / `*.parquet` / `*.pdf` 一律不入库；示例都写在系统临时目录。`git status` 干净是仓库的硬规则，CI 有 `generated-data` job 守着。
2. **基准数字只用于同机 A/B**。手册和文档里的毫秒数是在 RTX 4060 Laptop / Windows 上量的，**不能当对外性能承诺**，跨平台只比趋势。
3. **边界要一起看**。`knn` 不支持 HNSW、量子图不存振幅、GPUDirect 当前恒不可用、REPL 没有图遍历——这些不是「待修 bug」，是已知边界，写在 [`product-design.md`](product-design.md) §7。

---

## 12. 用 Python 跑同一套查询

**能，而且能 `pip install gtvdb`——但装的是驱动，不是引擎。** 本仓没有任何 PyO3 / maturin 代码（`grep -rn pyo3 --include=Cargo.toml .` 为空），ROADMAP §4 把 M9（pyo3）判为 🗑 下架，所以 Python 侧永远不编译扩展：`gtvdb` 是纯 Python 的 DB-API 2.0 客户端，引擎仍在别处跑。今天有两条可用的路：**Python 当客户端驱动 `gtv` 子进程**（本地、零依赖），或**走 gRPC 协议 v1 连一个跑着的服务**（跨机器、带回执）。

| 方式 | 今天能用 | 能力 | 代价 |
|:--|:--|:--|:--|
| **A. 客户端驱动 REPL**（§12.1–12.5） | ✅ | REPL 的全部能力（SQL + 12 条 `.` 指令） | 继承 REPL 的边界（§12.5）；每进程 335 ms 启动 |
| **B. gRPC 协议 v1**（§12.6） | ✅ | DB-API 2.0（`pip install gtvdb`），远程 SQL、`as_of` / `snapshot_id` 固定、`QueryReceipt`（含 tri-state 与 freshness） | 需先起服务；服务端目前只注册演示表；只读、无流式、无服务端分页 |
| C. 原生绑定（pyo3 / cdylib） | ❌ 需立项 | 可绕开 REPL 边界（图遍历、建索引） | 新 crate + 跨平台 wheel + 错误模型映射；M9 已下架 |

### 12.1 三十秒上手

```python
import sys; sys.path.insert(0, "examples/python")
from gtv_repl import Gtv

with Gtv() as db:                                  # 起一个 gtv 进程
    db.dot(".csv tx C:/data/tx.csv")               # 注册表
    rows = db.query("SELECT * FROM dq_report('tx');")   # 跑 SQL，拿 dict 列表
    print(rows[0]["rows"])
```

前置只有一个：先 `cargo build -p gtv-cli`。客户端按 `$GTV_BIN` → `target/release/gtv` → `target/debug/gtv` 的顺序找二进制。

### 12.2 完整示例的真实输出

`python examples/python/hello_python.py`（8 段，覆盖注册、DQ、量子状态、参数遥测、向量冷开、错误处理）：

```text
① 启动：一个 gtv 子进程，状态在整个会话里保持
binary   C:\ai\gtvdb\gtvdb_core\target\debug\gtv.exe
startup  335 ms (含构建产物定位)

③ 用 Python 写 CSV，再交给 gtv 注册
wrote C:/Users/JACAL_~1/AppData/Local/Temp/gtv-py-hw9csjaf/tx.csv
registered CSV `.../tx.csv` as table `tx`
edges
tx

④ 数据质量：先把报表取回来，再在 Python 里判断
dq_report  rows=3 dup_ts_sym=0 non_monotonic=0
  freshness      warn  last bar 20772 days ago (limit 7)
  row_count      warn  3 rows (min 50)
  no_nan         ok    0 non-finite float values
  no_duplicates  ok    0 duplicate (ts,symbol)
  ts_sorted      ok    0 backwards jumps
→ 2 项 warn，3 项 ok（ts=1000 是 1970 年，所以 freshness 报 warn 是正常的）

⑤ 量子状态图：森林形状 / 存活分支 / 稀有分支的门序列
states=10 transitions=9 depth=6 semantics=Conditional
  state 9   x q[2]         p=0.5
  state 7   measure q[1]   p=0.4
  state 8   measure q[1]   p=0.1
→ 最稀有分支 state 8 (p=0.1)，追溯它的门序列：
  [0] |000>                p=1.0
  ...
  [6] measure q[1]         p=0.1

⑧ 出错时 Python 拿到的是异常，不是被吞掉的输出
ok=False  .csv → error: Execution error: arrow error: Io error: The system cannot find the file specified. (os error 2)
ok=False  SELECT → error: Error during planning: table function 'neighbors' not found
raised GtvError: error: Error during planning: table function 'neighbors' not found

总耗时 408 ms（含一次进程启动）
```

注意 §12.2 里 `dq_report` 的数值是**在 Python 里判断**的（`warns = [c for c in checks if c["status"] == "warn"]`），不是靠打印——这是用 Python 而不是 REPL 的唯一实质理由：**结果可以进程序**。

### 12.3 客户端 API

| 方法 | 返回 | 用途 |
|:--|:--|:--|
| `Gtv(binary=None, cwd=None, timeout=120, env=None)` | — | 定位二进制；构造不起进程，`start()` / `with` 才起 |
| `sql(stmt, check=True)` | `str` 输出原文 | 跑 SQL 或 `.` 指令；有 `error:` 行就抛 `GtvError` |
| `try_sql(stmt)` | `(ok, text)` | 不抛异常，自己判断（示例 §12.2 第 ⑧ 段） |
| `query(stmt)` | `list[dict[str, str]]` | `sql` + 解析最后一张表 |
| `table(text)` | `list[dict[str, str]]` | 只解析表（自己拿到输出后再处理时用） |
| `dot(cmd)` | `str` | `.csv` / `.segment` / `.accel` 等 |
| `banner` | `str` | 启动横幅（版本与"零扩展"声明） |

### 12.4 实现里三个不显然的点（自己写客户端会踩）

1. **REPL 没有"结果结束"标记**。提示符在管道模式下不打印（rustyline 只在 tty 上写），所以客户端在每条语句后面追加 `SELECT '<uuid>' AS __gtv_eof;`，读到这个 uuid 就算结束。uuid 每次不同，语句自己的输出里出现同名字符串也不会误判。**收尾边框要一起读掉**，否则它会变成下一次调用的第一行（第一版就栽在这，每段前面多一行 `+---+`）。
2. **`stderr` 要并进 `stdout`**。REPL 用 `eprintln!` 报错（`error: ...`），分开读的话先到的错误会看起来"没输出"，而且没人排空 stderr 队列时还可能死锁。
3. **别按 `|` 切单元格**。量子门的名字就是 `|000>`，值里天然带竖线；中文又是双宽字符。所以 `table()` 用边框 `+---+` 的加号位置定列边界，再按显示宽度切片。按 `|` 切的第一版把 `|000>` 解析成了空 gate + `000>`。

### 12.5 边界：Python 继承 REPL 的一切限制

Python 走 REPL，就等于站在手册 §1 的轨道 A 上，**不能**做的事一模一样：

| 想做 | 为什么不行 | 出路 |
|:--|:--|:--|
| `neighbors` / `khop` / `motif` | 需要宿主 `register_neighbors(&csr)` 挂一个 `TemporalCSR`，REPL 没有建图指令 | 写 Rust（`examples/hello_graph.rs`），或等 `.graph` 指令立项 |
| 建向量索引、写 `.gtvidx` | `.knn-index` 只能**读**已有索引；构建走 `IndexStore::build` | 先跑 `hello_vector`（Rust）建一次，Python 侧冷开（§12.2 第 ⑦ 段就是这么做的） |
| `gtv_tables()` / `dq_gate` 的目录侧 | 需要 `attach_catalog` / `register_segment_store` | Rust 嵌入 |
| 每请求一个进程 | 一次启动 335 ms（debug 构建；release 更快但不适合这个用途） | 长会话复用一个 `Gtv` 实例；需要并发就每个线程一个实例 |

一句话：**Python 是"操作已配置好的引擎"，不是"配置引擎"。**

### 12.6 远程会话：gRPC 协议 v1

跨机器、或想让答案带"证据"时，走 gRPC，不走子进程。协议定义在 `crates/gtv-proto/proto/gtvquery.proto`；Python 侧是一个**可安装的驱动** `python/`（PyPI 包名 `gtvdb`），对外是 DB-API 2.0，和 `psycopg2` / `pymysql` 一个用法。

```bash
pip install gtvdb             # 尚未发布时：pip install ./python（在 gtvdb_core 根下）
python -c "import gtvdb; print(gtvdb.apilevel, gtvdb.paramstyle)"
```

注意装的是**驱动**：`import gtvdb` 不会带起引擎，`connect()` 指向一个已在跑的服务。发版流程（TestPyPI 预演、trusted publishing、版本号不可重用）见 `python/PUBLISHING.md`。

纯 Python 实现（只有 `grpcio` + `pyarrow` 两个依赖），没有编译产物、没有平台 wheel、用户机器上不需要 `protoc`。三层结构，按需要往下取：

| 层 | 对象 | 多了什么 |
|:--|:--|:--|
| DB-API 2.0 | `Connection` / `Cursor` | `execute` / `fetchall` / `description`，`pandas.read_sql` |
| 会话 | `Session` / `Result` | `as_of` 固定、`QueryReceipt`、`tri_state` |
| 线协议 | `gtvdb._stubs` | `gtvquery.QueryService` v1 |

**① 起一个服务**。开发/联调用内核自带的示例服务（协议 v1 全套，只有两张演示表）：

```bash
cargo run -p gtv-engine --example serve_demo          # 监听 127.0.0.1:50051
GTV_ADDR=0.0.0.0:50051 cargo run -p gtv-engine --example serve_demo
```

生产部署用 `gtvdb_ext` 的 `gtv-server`（带鉴权、TLS、真实表）：它的 core pin 已升到协议 v1，`ExecuteQuery` / `Negotiate` / `Execute`（v0 兼容）三个 RPC 都在，`as_of` 会一路传到引擎，回执同时出现在 `receipt_json` 与 Arrow schema metadata 里。超时字段也被真正实现——`timeout_ms` 到点返回 `DEADLINE_EXCEEDED`，而不是悄悄截断。

**② 连上去查**——和 PostgreSQL 一样的三行：

```python
import gtvdb

con = gtvdb.connect("gtv://127.0.0.1:50051")      # 或 host=..., port=...
cur = con.cursor()
cur.execute("SELECT * FROM numbers WHERE n <= ?", (3,))
cur.fetchall()          # [(1, 1.0), (2, 4.0), (3, 9.0)]
cur.description[0][1]   # gtvdb.INTEGER
con.close()
```

`with gtvdb.connect(...) as con:` 也可以；`con.execute(sql)` 是 `con.cursor().execute(sql)` 的快捷方式。pandas 直接可用：`pd.read_sql("SELECT * FROM numbers", con)`（pandas 会提示"只支持 SQLAlchemy 连接"，这是它的保守告警，实际能跑）。

**③ 需要证据时，往下走一层**（DB-API 没有词汇表达"这个空结果什么也没证明"）：

```python
from gtvdb import Session

with Session.connect("127.0.0.1:50051") as session:   # 建 channel + 版本握手
    result = session.query("SELECT * FROM numbers WHERE n <= 3")
    result.table.to_pydict()                          # {'n': [1, 2, 3], 'square': [1.0, 4.0, 9.0]}
    result.tri_state                                  # 'found'
    result.freshness                                  # {'as_known_at': 1700000000000000000, ...}
```

**④ "开一个会话"在这里是什么意思**。服务端是无状态的：没有 `CREATE SESSION`、没有事务句柄、也没有需要 `close` 的游标。会话 = 客户端把**必须固定下来的东西**打成一个包：

| 固定什么 | 为什么 |
|:--|:--|
| channel（一条 TCP 连接） | 每条查询复用，别一次一连接 |
| 协商出的协议版本 | v1 有 as_of 与回执；v0 只有 SQL 进出 |
| bearer token | 每次调用带 `authorization: Bearer <token>` |
| `as_of`（业务时间） | 不固定的话，连着的两条查询可能看到两个快照 |
| `snapshot_id` | 客户端认为自己在读的快照；不一致时服务端**报错**而不是换个快照糊过去 |

DB-API 层把 `as_of` 放在**连接**上（等价于"在 T 时刻打开一个会话"），并提供 `pin()` 换时间而不换连接：

```python
con = gtvdb.connect("gtv://host:50051", as_of=1_700_000_000_000_000_000)
con.pin(as_of=1_700_000_086_400_000_000)   # 同一条 channel，新时间
```

```python
pinned = session.at(as_of=1_700_000_000_000_000_000)   # 子会话，共享 channel
a = pinned.query("SELECT count(*) FROM prices")        # 两条查询同业务时间
b = pinned.query("SELECT * FROM numbers")
```

**④ 结果里比行更重要的是回执**。`Result.receipt` 就是 `QueryReceipt` 的 dict：

```text
空结果 + tri_state="no_match"     → 证否：确实没有
空结果 + tri_state="incomplete"   → 什么也没证明（预算/超时打断了）
```

`Result.is_conclusive` 只在前者（或 `found`）为 `True`。`result.receipt["snapshot_id"]`、`["as_of"]`、`["freshness"]["as_known_at"]` 让"这批数据有多新"变成可传递的事实，而不是一句口头承诺。

**⑤ 四个自己写客户端会踩的点**：

1. **v0 的 `tri_state` 是假的**。旧服务端不算 tri-state，字段到线上一律是 proto3 默认值 `FOUND`；照读就会把"不知道"读成"有结果"。驱动对 v0 一律忽略该字段，只信回执。
2. **回执在 Arrow 里也有一份**（`gtv.receipt.json` 这个 schema metadata）。所以 v1 服务端写给 v0 客户端的数据仍然带着出处，落盘成 `.arrow` 文件后回执也还在。
3. **`?` 是客户端渲染的，不是服务端绑定**。协议 v1 里 SQL 就是一个字符串，没有绑定槽位（`QueryRequest` 没有参数数组）。所以 `_params.py` 在**字符串字面量/标识符/注释之外**逐个替换 `?`，值按类型渲染成 `'O''Brien'`、`TIMESTAMP '...'`、`X'0102'`、`NULL`——强于"`str()` 拼进 SQL"，弱于真正的 bind protocol（参数不能用在只能是字面量的位置，且每次都要重新 plan）。`paramstyle="qmark"` 说的就是这件事。
4. **stub 不要手改**。`python/gtvdb/_stubs/` 是 `python/gen_stubs.py` 的产物；改完 `.proto` 跑 `python python/gen_stubs.py`，CI 用 `--check` 保证不漂移。

**⑥ 目前的边界**（诚实清单）：

| 想 | 现在 | 为什么 / 出路 |
|:--|:--|:--|
| 写数据（INSERT/UPDATE） | ❌ 引擎拒绝 | 写入按既定分工走 RDBMS/Redis，再定时 refresh 进 gtvdb；不会假装支持 |
| `commit()` / `rollback()` | no-op | 引擎没有事务；留着只是为了满足 PEP 249 和 pandas 的调用习惯 |
| 服务端绑定参数 | ❌ | 协议 v2 的事；现在是客户端渲染 |
| 流式拉取 / 游标分页 | ❌ 一次往返拿全量 | 大结果集先 `LIMIT`，或用 `read_table()` 走 Arrow 零拷贝 |
| SQLAlchemy | ❌ 未提供 dialect | `pd.read_sql` 能跑（会 warn）；正式支持要做 `gtvdb+grpc://` dialect，属下一步 |
| Oracle/PostgreSQL 那种 `fetchmany(size)` 服务端分页 | ❌ 客户端切片 | 同上，需要服务端 portal |

测试：`python -m unittest discover -s python/tests`（30 个用例，纯 Python 假服务端，不需要 Rust 进程），覆盖 DB-API 契约、参数渲染、v0 回退、`as_of` 上线、错误码映射、pandas。示例脚本 `python examples/python/hello_grpc.py` 打真实服务端。

---

## 13. 下一步

| 你想 | 去哪 |
|:--|:--|
| 接自己的扩展 | `crates/gtv-engine/src/ext.rs` 的 `GtvExtension`，示例见 `README.md` §5 |
| 看参数全集 | [`params-file.md`](params-file.md)、`gtvdb.params.example` |
| 看段格式 | [`storage-mmap-design.md`](storage-mmap-design.md) |
| 看 GPU 边界 | [`acceleration.md`](acceleration.md) |
| 看性能基线怎么量 | `crates/gtv-engine/examples/bench_m6_gate.rs` 等基准（`--release`） |
| 用 Python 调 | 本节 §12：本地子进程 `examples/python/gtv_repl.py`，远程 `pip install gtvdb` + `gtvdb.connect()`（§12.6） |
