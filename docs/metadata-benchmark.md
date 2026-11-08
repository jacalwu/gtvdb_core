# L1 / L2 元数据：实现与基准（MongoDB 样本数据）

> 2026-11-09｜对应设计文档 [metadata-design-comparison.md](metadata-design-comparison.md) §7 的**阶段 A**。
> 数据来源 [`erelbi/mongodb_sample_data`](https://github.com/erelbi/mongodb_sample_data)（MongoDB 官方 sample 数据集的 mongodump 转存）。
> 结论：**L2 的时间盒在区间/点事件负载上削掉 97.8%–99.7% 的扫描量，在静态图和开放式 as-of 图上如预期地不削（或不谎报）**；L1 把「有哪些表 / 有哪些版本 / 版本由哪些文件组成」变成 SQL 可查。
> 状态：代码已落地、测试已跑通（421 passed / 0 failed / 1 ignored）。**未提交、未推送。**

---

## 1. 这次到底做了什么

### 1.1 L2：段时间盒（`gtv-storage`）

| 交付 | 位置 | 说明 |
| :-- | :-- | :-- |
| `SegmentMeta` + 判定式 | `crates/gtv-storage/src/segment_meta.rs` | 四个时间标量 + 节点/edge_type 包围盒 + `overlaps / may_contain / all_active_at / fully_covers / classify` |
| manifest 字段 | `crates/gtv-storage/src/store.rs` `SegmentEntry` | 6 个 `Option` 字段，flush 时一次线性扫描写入 |
| 剪枝选择器 | 同上 | `entries_at_business` / `entries_overlapping` / `entries_at` / `business_span` / `prune_summary` |
| SQL 视界 | `crates/gtv-engine/src/segment.rs` | `gtv_segments()` 新增 7 列（含 `all_active_from/to`、`valid_from_min/valid_to_max`、`node_id_min/max`、`prunable`） |

**四个标量不是两个**，这是本节最容易搞错的地方：

```text
交集（全命中窗口）: max_valid_from = max(valid_from), min_valid_to = min(valid_to)
并集（剪枝包围盒）: valid_from_min = min(valid_from), valid_to_max = max(valid_to)

跳段      : t < valid_from_min || t >= valid_to_max
区间重叠  : valid_from_min < T2 && valid_to_max > T1
全段短路  : max_valid_from <= t < min_valid_to  →  active_edge_count = edge_count
```

用交集剪枝会**漏数据**，用并集短路会**多算**。两者在同一结构里并存，各自的谓词都写明了读哪一组。

**旧 manifest 兼容**：6 个字段都是 `Option`，缺失时谓词一律**保守选中**（`overlaps → true`、`all_active_at → false`）。有测试模拟「删掉这 6 个 key 的 v1 manifest」，验证仍然 100% 返回数据。

### 1.2 L1：单一元数据出口（`gtv-engine`）

| 交付 | 说明 |
| :-- | :-- |
| `GtvContext::attach_catalog(root)` / `catalog()` | 把 `FsCatalog` 挂到会话，槽位与 UDTF 共享，SQL 立刻可见 |
| `gtv_tables()` | 每表一行：schema 版本、列数、分区规范、事件时间列、最新快照、快照数、文件数、行数、字节数、事件时间盒 |
| `gtv_snapshots()` | 全库版本日志：parent、op、schema/spec 版本、文件数、created_at、summary |
| `gtv_files('t' [, 'snap'])` | **精确**文件清单 + 每文件事件时间盒（可 pin 到任意历史版本） |
| `GtvContext::sync_catalog()` | 按「最新快照的文件清单」把表注册成 DataFusion 视图 —— 表能被查询**是因为 catalog 这么说**，而不是因为有人调过 `register_csv` |

`sync_catalog` 只读快照里列出的文件，不会因为上游目录里多了一个 parquet 就读进来（这是「目录扫描式注册」的经典错误结果）。

---

## 2. 基准数据

原始数据是 BSON 文档，不是图。**投影规则是显式的，不是猜的**：

| 数据集 | 原始集合 | 投影 | 边数 | 节点数 | 时间语义 |
| :-- | :-- | :-- | --: | --: | :-- |
| `trips.gtvedge` / `trips_x100` | `sample_training.trips`（Citi Bike 骑行） | 起点站 → 终点站 | 10,000 / 1,000,000 | 465 | **真实区间** `[start, stop)` |
| `comments_point` | `sample_mflix.comments` | 用户(email) → 电影 | 50,303 | 14,559 | 点事件 `[t, t+1)` |
| `comments_cumulative` | 同上 | 同上 | 50,303 | 14,559 | 持续态 `[t, i64::MAX)` |
| `routes` | `sample_training.routes` | 机场 → 机场 | 66,985 | 3,404 | **静态** `[0, i64::MAX)`（对照组） |

数据事实（脚本实测，非估算）：

- `comments` 日期在 **1970–2018 之间近乎均匀分布**（每年约 1,000 条），所以按月分桶得到 586 个都非空的段 —— 正好压测「多小段」。
- `trips` 全部落在 2016-01-01 ~ 2016-01-06 的 5 天里；`trips_x100` 是把这 10,000 条边复制到 **100 个连续的 30 天窗口**（**合成放大，报告里单独标注**）。
- `routes` 无边级时间信息，**任何时间窗口都无法剪枝**，作为「L2 必须在无能时保持沉默」的对照组。

导出脚本把 BSON 转成 34 字节定长的 `.gtvedge`（`src u64 | dst u64 | valid_from i64 | valid_to i64 | edge_type u16`）。**没有 CSV**：仓库禁止提交 CSV，且定长记录让准备成本在测量之外一次付清。

全部数据在 `C:/ai/gtvdb/_benchdata/`，**在三个仓库之外**。

---

## 3. 方法

每条被测语句都在 `crates/gtv-engine/examples/bench_metadata.rs` 里，release 构建，单机单进程，页缓存已预热。

**窗口**：以「最早边起点 → 最晚边起点」为数据轴（不用 `business_span()`，它会因为 `valid_to=i64::MAX` 把轴拉到 292 年后），在其中取
`1×30d`（最选择性）、`3×30d`、`12×30d`、`full`。

**每个窗口都做一次正确性断言**（这是这个基准最重要的部分）：

```rust
let full   = active_edges(store, t1, t2, None);          // 打开所有段、逐边判定
let pruned = active_edges(store, t1, t2, Some(&ids));    // 只打开 L2 选中的段
assert_eq!(pruned, full, "pruning lost rows");
```

一个只测速度的基准会愉快地奖励「剪得干净但漏数据」的剪枝器。这里每个窗口都比对了全量答案。

**指标含义**：

| 列 | 含义 |
| :-- | :-- |
| `cold open (manifest)` | `SegmentStore::open` 读 manifest 建索引，**一个 `.seg` 都没打开** |
| `open every .seg` | 没有 manifest 元数据时的替代做法：逐段 mmap + 校验 header |
| `decide` | `entries_overlapping` 本身耗时（min of 20），纯整数比较 |
| `selected / skip %` | 被选中段数 / 被跳过的比例 |
| `full_hit` | 命中「全段短路」的段数（`edge_count` 即答案，不解边） |
| `edge-bound` | 选中段的 `edge_count` 之和（上界） |
| `active` | 逐边判定的真实活跃边数 |
| `scan ... (all) -> ...` | 全段扫描耗时 → 只扫选中段的耗时 |

---

## 4. 结果

### 4.1 冷启动：L2 没有给 open 增加任何成本

| 数据集 | 段数 | manifest | cold open | 逐段打开（无 L2 时） | 倍差 |
| :-- | --: | --: | --: | --: | --: |
| trips_x100 | 100 | 53.2 KB | **2.52 ms** | 38.07 ms | 15× |
| comments_point（月） | 586 | 315.7 KB | **2.78 ms** | 452.13 ms | 163× |
| comments_point（年） | 49 | 26.4 KB | **2.90 ms** | 23.41 ms | 8× |
| comments_cumulative | 586 | 316.5 KB | **3.02 ms** | 477.68 ms | 158× |
| routes（对照） | 12 | 6.1 KB | **2.61 ms** | 5.03 ms | 2× |

段数从 49 涨到 586（12×），冷启动从 2.90 ms 变成 2.78 ms —— **基本不变**。原因很直接：时间盒是 manifest 里的标量，读 manifest 本来就是 O(S)，而打开段是 O(S × mmap + header 校验 + 页错误)。

### 4.2 剪枝决策：100 段 0.1 µs，586 段 0.7–1.8 µs

约 **1–3 ns / 段**。这个数字的意义不是「快」，而是**它便宜到可以在每次查询里做，不需要为它设计缓存**。

### 4.3 剪枝收益：区间与点事件 100×–529×

`trips_x100`（1M 条真实区间边，100 段，每段 10k 边）：

| 窗口 | 选中/总 | skip | full_hit | edge-bound | active | scan 全量 → 剪枝后 | 加速 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| 1×30d | 1/100 | **99.0%** | 0 | 10,000 | 10,000 | 44.11 → **0.43 ms** | **103×** |
| 3×30d | 3/100 | 97.0% | 0 | 30,000 | 30,000 | 46.05 → 1.35 ms | 34× |
| 12×30d | 12/100 | 88.0% | 0 | 120,000 | 120,000 | 45.52 → 5.27 ms | 8.6× |
| full | 100/100 | 0% | 0 | 1,000,000 | 1,000,000 | 47.62 → 44.98 ms | 1.06× |

`comments_point`（50,303 点事件，586 段）：

| 窗口 | 选中/总 | skip | scan 全量 → 剪枝后 | 加速 |
| :-- | --: | --: | --: | --: |
| 1×30d | 2/586 | **99.7%** | 518.59 → **0.98 ms** | **529×** |
| 3×30d | 4/586 | 99.3% | 490.67 → 2.04 ms | 240× |
| 12×30d | 13/586 | 97.8% | 536.43 → 13.44 ms | 40× |
| full | 586/586 | 0% | 534.82 → 527.53 ms | 1.01× |

全量窗口的 1.0× 是**预期**：窗口覆盖全部时间轴时，L2 本来就没有可剪的东西，它必须退化成全扫而不是假装省了时间。

### 4.4 对照组：无法剪枝时 L2 必须沉默

`routes`（静态图，12 段，每个窗口都覆盖全部边）：

| 窗口 | 选中/总 | skip | full_hit | scan 全量 → 剪枝后 |
| :-- | --: | --: | --: | --: |
| 1×30d | 12/12 | **0.0%** | 12 | 4.91 → 4.95 ms |
| 3×30d | 12/12 | 0.0% | 12 | 6.01 → 7.29 ms |
| full | 12/12 | 0.0% | 0 | 5.69 → 6.58 ms |

0% 剪枝、耗时无改善，但**也没有多算**：`full_hit` 命中说明「全段活跃」是真话（静态图确实每条边都活跃），所以 `edge_count` 直接就是答案，逐边测试被跳过。

### 4.5 开放式 as-of 图：剪枝只剩「砍尾巴」

`comments_cumulative`（`[t, MAX)`，586 段）：

| 窗口 | 选中/总 | skip | full_hit | edge-bound | active | scan 全量 → 剪枝后 | 加速 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| 1×30d | 149/586 | 74.6% | **147** | 12,983 | 12,982 | 444.04 → 129.39 ms | 3.4× |
| 3×30d | 299/586 | 49.0% | 295 | 25,973 | 25,968 | 408.58 → 270.66 ms | 1.5× |
| 12×30d | 72/586 | 87.7% | 59 | 6,261 | 6,180 | 462.15 → 55.86 ms | 8.3× |
| full | 586/586 | 0% | 0 | 50,303 | 50,303 | 482.21 → 546.31 ms | 0.9× |

这是**最诚实的反例**：开放式边的 `valid_to_max` 恒为 `MAX`，包围盒右端永远不闭合，所以只能靠「起点晚于窗口终点」来砍尾。收益从 529× 掉到 3.4×。

但这里暴露了 L2 真正值钱的地方：**147/149 个选中段被判定为 `FullHit`**——它们的所有边都在窗口开始前就已存在，所以答案就是 `edge_count`，一行边都不用碰。129 ms 里剩下的工作，基本就是那 2 个必须 `Scan` 的段。

`12×30d` 那行有个细节值得注意：`edge-bound 6,261` 而 `active 6,180`，差的 81 条正是「起点在窗口内、但晚于 T2」的边界边 —— 包围盒是上界，不是答案。

### 4.6 L1：50 表 × 20 快照 = 1,000 个版本

| 操作 | 耗时 | 备注 |
| :-- | --: | :-- |
| `FsCatalog::list_tables()` | **0.62 ms** | 50 张表的注册表 |
| 空 catalog 跑 `gtv_tables()`（基线） | **0.62 ms** | UDTF + planner 固定开销 |
| `gtv_tables()` | 121.53 ms | 50 行 ×（schema + latest + 20 条快照日志 + 20 条文件索引） |
| `gtv_snapshots()` | 25.03 ms | 1,000 行统一版本日志 |
| `gtv_files('t0000')` | 2.31 ms | 单表 20 个文件 |
| `sync_catalog()` | 329.40 ms | 50 张表按快照文件清单注册成视图（≈6.6 ms/表） |

`gtv_tables()` 的 121 ms 里只有 0.6 ms 是 SQL 开销，其余是**每表约 2.4 ms 的真实元数据读取**：一张表要读 `schemas.json`、`latest`、`snapshots.jsonl`(20)、`files.jsonl`(20)。这是「**不缓存、永远准确**」的代价，也是当前最明显的优化点（见 §6.2）。

---

## 5. 五个结论

**① L2 的时间盒确实能免掉大部分 I/O，而且免得很干净。**
区间/点事件负载 97.8%–99.7% 的段被跳过，扫描量降到 1/100–1/529，且每个窗口都与全量答案逐条比对通过。这不是「索引变快了」，是**根本不需要打开那些文件**。

**② 冷启动与段数几乎无关，这是设计目标的直接验证。**
49 段与 586 段的 `open` 都是 ~2.8 ms。`docs/storage-mmap-design.md` 里「O(1) cold open」的主张在多段场景下依然成立，因为段选择发生在打开文件之前。

**③ 剪枝决策便宜到可以不缓存（1–3 ns/段）。**
所以「元数据可能陈旧」这个风险类别在 L2 上是**不存在的**：每次查询现算，永远和 manifest 一致。这与 L1 的选择（也不缓存）相互印证。

**④ 开放式时间语义会让包围盒退化，此时收益靠 `FullHit` 而不是 `Skip`。**
as-of 图（`[t, MAX)`）只能砍尾，剪枝率 49%–88%，加速 1.5×–8.3×；但 147/149 的命中全是「整段全命中」，也就是**用 `edge_count` 代替解边**。这说明「交集窗口」不是冗余字段，它在开放式场景里比并集包围盒更有价值。

**⑤ 段粒度必须对着节点基数选，不能只对着边量选。**
这是本次最有行动价值的一条：

| comments_point 分桶 | 段数 | 每段边数 | 数据大小 | B/edge | flush 总耗时 | 全量扫描 |
| :-- | --: | --: | --: | --: | --: | --: |
| 按月（30 天） | 586 | 86 | **69.02 MB** | **1,439** | 9,975 ms | 535 ms |
| 按年（365 天） | 49 | 1,027 | **6.96 MB** | **145** | 857 ms | 26 ms |

同样的 50,303 条边，同样的 14,559 个节点，**按月分桶让存储膨胀 10 倍、flush 慢 12 倍、全扫慢 20 倍**。原因是 GTVSeg 是**稠密 CSR**：每段都要带 `node_count` 个 `offsets` 和 `chunk_index`（14,559 × 8 B ≈ 116 KB），哪怕这段只有 86 条边。586 × 116 KB ≈ 68 MB，正好是那 69 MB。

因此落地规则是：**每段边数应当 ≳ 节点数**，否则分桶收益被 CSR 的 O(V) 开销吃掉。真正的解法是 P2 的「段内节点区间窄化」（`node_id_min/max` 已经算好并存进 manifest，正好为此准备），或按 `(time_bucket, src_range)` 二维分片。

---

## 6. 已知限制与后续

### 6.1 测量口径的限制

- **Windows + 每个段 fsync**：`flush` 数字（1M 边 1.53 s；586 段 10 s）里含大量 `fsync`，Linux 上会明显更低。跨平台比较请只比相对值。
- **页缓存已预热**：`scan` 是热缓存口径。冷缓存下差距会更大（L2 的优势只会更明显，因为跳过的文件连页错误都不产生）。
- **`trips_x100` 是合成放大**：真实数据只有 10,000 条。放大只改时间偏移，不改拓扑，因此对「剪枝率」无影响，对「绝对耗时」有影响。
- **单进程单写者**：与 `FsCatalog` 现状一致（`store.rs:22-23`），并发写未测。

### 6.2 下一步（按收益排序）

1. **L1 的每表元数据索引**：`gtv_tables()` 2.4 ms/表主要花在重复解析 `snapshots.jsonl` / `files.jsonl`。加一个 `metadata/<table_id>/stats.idx`（commit 时原子更新）即可降到 O(1)/表，而**不引入缓存不一致**（它是提交协议的一部分，不是旁路缓存）。
2. **段内节点区间窄化（P2）**：`node_id_min/max` 已就位。写入时只覆盖 `[node_id_min, node_id_max]` 的 offsets 段，可把稠密 CSR 的 O(V) 开销变成 O(ΔV)。这是 §5⑤ 的根治手段。
3. **`[T1,T2)` 原生聚合**：用 `prune_summary` 的 `FullHit` 直接产出「每桶边数 = edge_count」，配合 `12x30d` 这类窗口可以完全不打开文件。
4. **compaction**：586 个 86 条边的段，合并成 49 个 1,027 条边的段，正是 §5⑤ 的在线版本。

---

## 7. 复现

```bash
# 1. 取数据（仓库外，约 30 MB）
mkdir -p /c/ai/gtvdb/_benchdata && cd /c/ai/gtvdb/_benchdata
curl -sSLO https://raw.githubusercontent.com/erelbi/mongodb_sample_data/main/sample_mflix/sample_mflix/comments.bson
curl -sSLO .../sample_training/sample_training/trips.bson
curl -sSLO .../sample_training/sample_training/routes.bson

# 2. BSON -> 定长边文件（无 CSV）
python export_edges.py

# 3. 跑分
cd /c/ai/gtvdb/gtvdb_core
cargo run -p gtv-engine --release --example bench_metadata -- /c/ai/gtvdb/_benchdata
```

测试：

```bash
cargo test -p gtv-storage --test l2_time_box            # 8 项：剪枝不漏、FullHit 真实性、旧 manifest 兼容
cargo test -p gtv-engine  --test l1_catalog_meta        # 6 项：L1 三个 UDTF + sync_catalog
cargo test -p gtv-engine  --test l2_segment_prune_sql   # 3 项：SQL 侧的两个时间盒与 NULL 语义
cargo test --workspace                                  # 421 passed / 0 failed / 1 ignored
```
