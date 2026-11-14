# 过去时点 Node / Edge 更新的处理设计

> 2026-11-09｜对照 `docs/delta-segment.md`、`docs/update-segment.md` 两份需求稿，以及 `gtvdb_core` 当前代码事实。
> 触发问题：在时序图上，如果用户要对**过去某一个时间点**的 Node 或 Edge 做 update，怎么处理？现有设计是否需要改动。
> 状态：设计评估 + 现状判定。**未改动任何代码，未提交、未推送。**

---

## 0. 结论先行

**需要改，但只需改三处，而且都不是两份文档里最重的那些。**

| 判定 | 项 | 理由 |
|:--|:--|:--|
| **必须改** | 引入 `Retract`（撤销）原语，目标用**物理位置** `(segment_id, row)` | 现有段不可变 + 只增并集，无法表达「旧事实从 S2 起不再成立」 |
| **必须改** | 读路径在**物化期**折叠 retract（`graph_at` / 段选择） | `graph_at` 现在是纯并集，更正后新旧版本会同时出现（§3.1） |
| **必须改** | WAL 与 delta 的 mutation 编码（WAL 升 v2） | WAL 只有 `OP_INSERT`，遇到未知 op 直接报错，无法向前兼容 |
| 不用改 | `TemporalCSR` 结构、`neighbors()` 热路径 | 系统时间在物化前折叠，CSR 永远不需要 system 列（§4.1 I2） |
| 不用改 | L2 六个判定式、`SegmentMeta` | 只增一个「段级 system 盒」，判定式原样复用（§5.2） |
| 不用改 | L3、图漫步 `khop_*` | L3 是派生缓存；`khop` 消费的是「已解析的 CSR + 业务时间」，天然正确（§5.3） |
| **要收缩** | 两份文档中的 u128 EdgeId/VersionId、Property Blob、Branch/Merge、system-time 改写、单版本 split | 见 §2 与 §7 |

一句话：**两份文档的「不可变段 + PUT after-image + RETRACT + 原子发布」是对的；但它们把一个三五百行的删除层，写成了三千行的可变图引擎。**

---

## 1. 需求归约

### 1.1 先把三条时间轴分开

「过去某一个时间点 update」这个说法里混了三条不同的轴，必须先切开，否则设计一定会错位：

| 轴 | 含义 | 谁写入 | 能否回写 |
|:--|:--|:--|:--|
| **业务时间** `valid_from/valid_to` | 事实在现实世界中何时为真 | 用户 | ✅ 可以更正（这正是本设计要解决的） |
| **系统时间** `system_from/system_to` | 数据库何时开始相信它 | 提交动作 | ❌ **永不回写**，这是审计保证 |
| **提交/摄取时间** | 这次写入落在哪个 txn | WAL / manifest | ❌ 单调递增 |

当前代码里第三条是 `SegmentEntry.system_ts` + `WAL.txn_id`（`store.rs:91`、`wal.rs:82`），第二条只在 `bitemporal.rs` 有类型定义、**没有接进存储**，第一条是 CSR 的两个列。

**因此「过去的系统时间」不是需求。** 用户想「让数据库以为我在上周就知道这件事」，那是篡改审计，只能作为 branch 或显式的 `REPLACE` 提供，不能是默认语义。所有「过去时点」都必须解释成 **business time 在过去**。这条一旦说清，一半的复杂度消失。

### 1.2 四种意图，一个判据

同样叫「在 Day5 更新」，实际有四种截然不同的动作。区分它们的判据只有一个：**新事实与已存在的版本，是否在 business 轴和 system 轴上都重叠**。

| 意图 | 例子 | 双轴重叠？ | 动作 |
|:--|:--|:--:|:--|
| **补录 / 迟到数据** | Day5 的边以前从没写过，现在补上 | 否（business 有交集但无旧事实） | 纯 append，`system_from = S2` |
| **更正 business 值** | 以前说 Day1–Day10 是 A，实际 Day5 起是 B | 是 | retract 旧版本 + 区间切分 + PUT |
| **删除（逻辑）** | Day4–Day7 这段不应该存在 | 是 | retract + 切分，修正区间不产生 PUT |
| **改写系统时间** | 「假装我上周就知道」 | — | **拒绝**（仅 branch / REPLACE） |

判据实现已经存在，就在 `crates/gtv-core/src/bitemporal.rs:120`：

```rust
pub fn overlaps(&self, other: &Self) -> bool {
    self.business_overlaps(other) && self.system_overlaps(other)
}
```

它当前是**死代码**（全仓只有 `lib.rs:22` 的 re-export，无调用方）。这个函数恰好就是「该 append 还是该 retract」的判定器——这是本设计最省力的一步：不新写语义，只把它接进写路径。

### 1.3 「过去时点」的两个子问题，难度差一个量级

- **过去时点的「点查询」**：`neighbors(src, business_t, system_t)`。只需在候选集里选对版本。
- **过去时点的「区间汇总」**：`[T1,T2)` 内活跃边数 / 活跃时长。这里有一个陷阱：区间切分会让**一条逻辑边变成三行**，行数不再等于边数（§4.8）。

两份文档只谈了前者，后者的语义在两份文档里完全没有定义。而上一轮的基准报告（`metadata-benchmark.md`）已经在为 `range_agg` 铺路——这两件事会撞上，必须先定语义。

---

## 2. 两份文档：三处硬伤、一处过度设计、两处必须保留

### 2.1 硬伤 A：`correct_interval` 只处理「一个旧版本 × 一个修正窗」

`update-segment.md` §8 的伪代码先取 `old.valid` 与 `[C,D)` 的交集，然后最多产生三段。问题在于它把「旧记录」当成**一条**。

反例（真实且常见）：某边的业务历史是

```
[D1, D5) = A
[D5, D10) = B        ← 两个不同版本
```

用户在 S2 修正 `[D3, D8)` 为 C。按文档伪代码，只能拿「一条 old」去切，结果要么漏掉 B 段的 `[D5,D8)`，要么产生与 B 重叠的新版本——后者直接违反 `update-segment.md` 自己的不变量三（同一 Snapshot 内同一实体有效区间不得重叠）。

**正确做法是对所有与修正窗重叠的版本做一次 sweep**，不是对一个版本做 split。修正窗跨越版本边界时，中间的多个版本要被一并撤销，修正值覆盖整个窗口。

### 2.2 硬伤 B：`old.with_system_to(commit_ts)` 是「可变记录」写法

同一段伪代码第 416 行 `versions.push(old.with_system_to(commit_ts))`，注释说「这里的关闭旧版本是逻辑语义，物理实现不是回写」。

但把它和 §4.2 的「RETRACT 表达关闭」放在一起看，就是两套模型混用：**如果既把旧版本作为一条新行 emit 出来（带 `system_to = S2`），又保留原行并 retract 它，物理上就有两条旧行**。一条 `system_to = S2`、一条 `system_to = ∞` 但被 retract 到 S2——语义等价，行数翻倍，且 retract 指向哪一条？这种歧义会在 compaction 时变成静默错误。

正确写法：**关闭系统区间不产生任何新行**，只是 retract 记录里一个时间戳。旧行原封不动。

### 2.3 硬伤 C：`delta-segment.md` §4.1 说「Patch 语义可以保留在 WAL」

原文：

> Patch 语义可以保留在 WAL 或 Commit Audit Log 中，但不进入查询热路径。

**这条不能照做。** 原因：WAL 重放时如果只有 patch，就必须知道「这个 patch 作用在哪个 base 版本上」，也就是必须能读到那个版本、且它必须还在。而 compaction 恰恰会把那个版本吃掉并重新编号。于是崩溃恢复的正确性依赖于「compaction 之后不再重放旧 WAL」——这是一个隐式的、没有 manifest 保证的耦合。

**WAL 里必须存 after-image + retract 目标，与 delta 文件同构。** 代价是 WAL 变大（一次更新写整行），换来的是重放不需要任何 base 状态、不需要二次派生、不会因为 compaction 而失效。更新量远小于摄取量，这个代价可以接受。

### 2.4 硬伤 D（补充）：修正窗内的空洞没定义

边的业务历史可能是断开的：`[D1,D5)` 存在，`[D5,D7)` 不存在，`[D7,D10)` 存在。用户修正 `[D3,D8)` 时，中间的 `[D5,D7)` 填不填？

两份文档都没提。这是**必须显式化的策略**，不能靠实现巧合：

- `AssertOverWindow`（默认）：修正即断言，`[D3,D8)` 全段都是新值，空洞被填上。
- `OnlyWhereExists`：只改已存在的部分，空洞保持空。

默认选前者，因为它符合「我说这段时间是 C」的自然语义；但必须写进 API。

### 2.5 过度设计：u128 `EdgeId` / `VersionId`

`delta-segment.md` §13.1 的**理由是对的**：

> 不能只写 `DELETE edge_id = E1`，因为一个 EdgeId 可能有多个历史版本。必须准确指向 `target_version_id`。

但它的**实现选错了**：合成 128-bit VersionId 不是唯一的「准确指向」方式，而且是最贵的那种（每条边多 16 B 列、WAL 记录从 42 B 涨到 58 B、还要保证生成规则的全局唯一性）。

**物理位置 `(segment_id, row)` 是更强的方案**：

- 段不可变 ⇒ 位置永久有效，O(1) 定位，不依赖任何排序或哈希。
- 不需要改 base 段的列结构（GTVSeg v1 原样可读）。
- 12 字节 vs 32 字节，且 `segment_id` 本来就在 manifest 里。
- 与 Iceberg 的 `POSITION_DELETES` 完全同构——而 `delta-segment.md` §4.2 自己引用了 Iceberg 的三种 content file。

代价（必须写进契约）：**compaction 必须重写 retract 集**，因为位置空间随段一起失效（§4.7）。

### 2.6 必须保留的两点

1. **PUT 存完整 after-image，不存 patch**（§4.1）。理由文档说对了：避免 patch chain、避免 compaction 理解历史语法。补充一条文档没说的：patch 还要求「属性字典版本」在 patch 与应用之间不变，这在 schema 演进下是不成立的。
2. **retract 与 put 在同一个文件、由同一个 manifest 条目发布**。原子性免费获得；分成两个文件就要两次发布，必然出现「撤销可见但新版本不可见」的中间态。

---

## 3. 现状核对：5 个硬阻断

### 3.1 `graph_at` 是纯并集，更正会双计

`crates/gtv-storage/src/store.rs:616-623`：

```rust
pub fn graph_at(&self, t: i64) -> Result<TemporalCSR> {
    let mut edges = self.mem.edges_at_system(t);
    for entry in self.entries_at_system(t) {
        let seg = self.open_entry(entry)?;
        append_segment_edges(&seg, entry.system_ts, &mut edges);   // 无条件并入
    }
    MemSegment::to_csr(&edges, self.node_count)
}
```

`entries_at_system` 的判据是 `e.system_ts <= t`（`store.rs:537`）。于是：

```
seg1 (S1): E = A->B, weight .5, valid [D1, D10)
seg2 (S2): E = A->B, weight .9, valid [D5, D10)   ← 一次「更正」

graph_at(S2)  →  两条都在
neighbors(A, D6)  →  返回 2 条边（应为 1 条）
```

这不是"性能问题"，是**错误结果**。

### 3.2 没有边的身份，retract 无处可指

`crates/gtv-storage/src/wal.rs:77-86` 的 `EdgeRecord` 只有 `(src, dst, valid_from, valid_to, edge_type, system_ts)`；`crates/gtv-core/src/csr.rs:97-103` 的 `EdgeRow` 一模一样。

推论：

- 用 `(src, dst, edge_type)` 当身份 → 平行边会被误杀。
- 用全五元组当身份 → **区间切分会改变 `valid_from/valid_to`，身份自己就变了**，无法表达「这条旧版本的这一段被替换」。
- 这两条都不行，所以下游只能靠物理位置（§2.5）。

### 3.3 读路径没有系统轴

`crates/gtv-core/src/csr.rs:497` 的 `neighbors(&self, src, valid_at)` 只接业务时间；`Neighbors` 迭代器只做 `valid_from/valid_to` 过滤。CSR 里根本没有 `system_*` 列。

这不是缺陷，是**正确的分层**：系统时间在「哪些行进入 CSR」这一步就折掉了，不该进遍历热循环（§4.1 I2）。但要求物化层先做对——而物化层现在（§3.1）没有做。

### 3.4 WAL 单 op，且未知 op 直接报错

`wal.rs:272-276`：

```rust
if op != OP_INSERT {
    return Err(err(format!(
        "WAL frame at {at} has valid checksum but unknown op {op}"
    )));
}
```

`store.rs:305-310` 的重放同样只接受 `OP_INSERT`。这是**正确但不可扩展**的选择：老 reader 遇到新 op 会硬失败（而不是误重放）。所以加 mutation 必须同时升 `WAL_VERSION`（header 里已校验，`wal.rs:220-225`），让老 reader 明确拒绝，而不是写一半。

### 3.5 manifest 没有 snapshot 链

`store.rs:165-175`：

```rust
pub struct Manifest {
    pub version: u32,
    pub node_count: u64,
    pub checkpoint_txn: u64,
    pub system_ts: i64,      // 单调水位，不是快照 id
    pub next_segment_id: u64,
    pub segments: Vec<SegmentEntry>,
}
```

没有 `snapshot_id` / `parent` / `branch`。这对**线性更正够用**（靠 `system_ts <= t` 就能时间旅行，因为所有旧段都还在列表里），对**分支不够用**（同一 `system_ts` 的两次提交无法区分）。

结论：**v1 做线性更正，不要碰 branch。** branch 需要 snapshot 链，而 snapshot 链属于 L1（`gtv-catalog` 已有 `Snapshot`，但和 storage manifest 是两套真相——这正是上一轮 `metadata-design-comparison.md` 已经点名的问题）。

### 3.6 两条其实够用的现状

- **L2 判定式不用动**：`SegmentMeta` 的六个标量判定（`segment_meta.rs`）对 delta 同样成立，只要 delta 也带自己的包围盒。
- **`SegmentEntry.system_ts` 单调** 恰好可以当 delta 的 `system_from`，不需要新增字段；缺的只是 `system_to`，而它是每行的属性（由 retract 决定），本来就不该在段级。

---

## 4. 设计

### 4.1 三条不变量（先立规矩，再谈结构）

**I1 — 读路径永不 dedup。**
冲突在**提交期**解决，不在查询期。理由：`neighbors()` 是 CSR 热循环，任何「按身份去重」都需要哈希表或额外排序，直接把 O(degree) 变成 O(degree·log)。一旦读路径允许去重，性能模型就崩了。
代价：提交时必须读一次受影响的邻接切片（§4.3）。这个代价有界（degree），且只在写路径付一次。

**I2 — 系统时间在物化前折叠。**
CSR 永远不需要 `system_from/system_to` 列。系统时间只影响「哪些行进入这次物化」，进入之后按业务时间遍历。这正是现有 `neighbors` 的形态，保持不变。

**I3 — retract 指物理位置 `(segment_id, row)`。**
不引入 edge id、不引入 version id。位置在段不可变的前提下永久有效，且 O(1)。

### 4.2 数据模型

```rust
/// 撤销一条已发布的行。`row` 是该段 `dst` 列里的行号（段内全局唯一）。
pub struct Retract {
    pub target_segment: u64,
    pub target_row: u32,
    pub reason: RetractReason,   // PropertyUpdate | ValidTimeCorrection | LogicalDelete | ...
}

/// 一次已提交变更的最小单元。
pub enum Mutation {
    Put(EdgeVersion),
    Retract(Retract),
}

/// after-image；`system_from` 恒等于本次 commit 的 system_ts。
pub struct EdgeVersion {
    pub src: u64,
    pub dst: u64,
    pub edge_type: u16,
    pub valid_from: i64,
    pub valid_to: i64,
}
```

注意 `EdgeVersion` 就是现有的 `EdgeRecord` 去掉 `system_ts` 后原样。**没有新增列**，GTVSeg v1 的 `dst/valid_from/valid_to/edge_type` 列结构不动。

### 4.3 写路径：提交期解决冲突

```
update(key = (src,dst,type), window = [C,D), new_payload)
 1. 确定可见版本：base 段中 system_ts <= now 的行 + 已发布 delta 的 PUT − 已发布 delta 的 retract
    —— 只读 src 的邻接切片（base O(1) 定位，delta O(log k)）
 2. 对每个与 [C,D) 重叠的 live 版本：生成 Retract + 左右保留 PUT
 3. 生成修正区间的 PUT（覆盖策略见 §2.4）
 4. 提交前校验：应用本次 mutation 后，同一 key 的 valid 区间不得重叠
 5. WAL.append(mutations) + fsync
 6. 写 .gtvd（PUT 段 + retract 表），fsync，rename
 7. 写 manifest（新段 + 指向它的 delta 条目），fsync，rename
 8. WAL checkpoint
```

第 4 步是本设计里最容易被跳过、但最值钱的一步。它不只是给调用方查错——**它同时是「第 1 步漏读了某个段」的检测器**。因为漏读的版本不会被 retract，于是与新 PUT 重叠，第 4 步立刻报错。没有它，漏读会变成静默的双计。

第 1 步的复杂度：按 `src` 分组，每个 src 读一次切片，`O(degree(src) + log R)`（R = retract 总数）。单边更新的代价是一次邻接扫描，可接受。

### 4.4 正确的区间切分（跨多版本 sweep）

```rust
fn correct(key, window: (c, d), new_payload, commit_ts, policy) -> Vec<Mutation> {
    let mut out = Vec::new();
    let mut covered: Vec<(i64, i64)> = Vec::new();

    // 关键：遍历「所有」与窗口重叠的 live 版本，不是「一个」旧版本
    for v in read_live_versions(key, commit_ts)? {
        let (a, b) = (v.valid_from.max(c), v.valid_to.min(d));
        if a >= b { continue; }                       // 不重叠，原样保留
        covered.push((a, b));

        out.push(Mutation::Retract { target_segment: v.seg, target_row: v.row, .. });

        if v.valid_from < a {                         // 左残段，原值
            out.push(Mutation::Put(EdgeVersion { valid_from: v.valid_from, valid_to: a, ..v.payload }));
        }
        if b < v.valid_to {                           // 右残段，原值
            out.push(Mutation::Put(EdgeVersion { valid_from: b, valid_to: v.valid_to, ..v.payload }));
        }
        // 中间 [a,b) 不保留：由下面的修正 PUT 覆盖
    }

    // 修正值的作用域：整窗（默认）或仅已存在部分
    let spans = match policy {
        Policy::AssertOverWindow => vec![(c, d)],
        Policy::OnlyWhereExists  => merge_adjacent(&covered),
    };
    for (a, b) in spans {
        out.push(Mutation::Put(EdgeVersion { valid_from: a, valid_to: b, ..new_payload }));
    }
    out
}
```

文档 §8 的伪代码缺的正是这个 `for` 循环：它假定 `old` 只有一条。上面的版本在「窗口跨两个历史版本」时依然给出正确结果：

```
旧: [D1,D5)=A, [D5,D10)=B        修正 [D3,D8) := C
新: Retract(两条)
    PUT [D1,D3)=A   PUT [D8,D10)=B
    PUT [D3,D8)=C
```

点查 D6 得到一条边（C），D2 得到 A，D9 得到 B。行数 5，活跃边数 1。正确。

**幂等性**：若新值与旧值在某个子区间完全相同，可跳过该 PUT（`skip_if_unchanged`）。否则一次重试就多三行版本，长期会把 `graph_at` 的物化成本抬上去。建议默认开启。

### 4.5 Delta 文件（`.gtvd` v1 最小集）

相比 `delta-segment.md` 砍掉了 Node PUT 段、Property Blob、96 字节通用 Column Directory。只留四段：

```
+------------------------------------------+
| Header 4096 B                            |  （复用 GTVSeg 的 4096/64 对齐约定）
+------------------------------------------+
| Section Table (固定 32 B × n)             |
+------------------------------------------+
| PUT: 稀疏 src 目录 + 平行定宽列            |
+------------------------------------------+
| RETRACT: 16 B × r                         |
+------------------------------------------+
| Footer: data_checksum                     |
+------------------------------------------+
```

PUT 列（全部定宽、可 mmap、可 `cast_slice`）：

| 列 | 类型 | 说明 |
|:--|:--|:--|
| `src_nodes` | `[u64; k]` | **稀疏**源目录，只列出现过的源 |
| `src_offsets` | `[u32; k+1]` | 进 `dst` 的偏移 |
| `dst` | `[u64; n]` | |
| `edge_type` | `[u16; n]` | |
| `valid_from` / `valid_to` | `[i64; n]` | |
| `system_from` | `[i64; n]` | 通常整列等于 `commit_ts`，保留逐行是为将来 |

`RETRACT` 表：`target_segment u64 | target_row u32 | reason u8 | pad[3]` = 16 B。

**为什么必须稀疏**：上一轮基准已经量化过——GTVSeg 是稠密 CSR，每段付 `O(node_count)` 的 offsets。1M 节点的图上更新 5 条边，稠密偏移数组要 4 MB。稀疏目录用 `binary_search(src_nodes)` 换 `O(log k)`，k 是本次更新涉及的源数，通常是 1 到几十。

排序键：`(src, dst, edge_type, valid_from)`。与 base 段一致，合并时无需重排。

Header 需要的标量（比文档的 20 个少很多）：`magic / format_version / header_size / delta_id / table_id / partition_id / commit_ts / base_snapshot_id / put_count / retract_count / src_id_min,max / valid_from_min,max / 三段 offset / header_checksum / data_checksum`，4096 B 绰绰有余。

### 4.6 读路径：K+1 源合并 + 位置位图

```
neighbors_bitemporal(src, business_t, system_t):
  1. base 段选择： system_ts <= system_t  且  business 盒命中          （L2，manifest 内完成）
  2. delta 选择： commit_ts  <= system_t  且  business 盒命中 且 src 盒命中
  3. 对每个候选源取切片：
       base  → O(1) offsets 定位
       delta → binary_search(src_nodes) + offsets
  4. 每个切片内按 valid_from <= business_t < valid_to 过滤
  5. 对每条候选行查 retract：binary_search((segment_id, row))，命中即丢弃
  6. 收集到小 Vec 后排序（degree 小，收集+排序 优于 k 路归并迭代器）
```

复杂度：`O(S + D + (K+1)·log k + degree·log R)`。相比现有 `neighbors` 只多两个小量（delta 定位、retract 二分）。

**关键性质**：因为是「整行撤销」，不是「按身份覆盖」，第 5 步之后同一 `(key, business_t, system_t)` 天然只有一行——**I1 成立，读路径不需要去重**。

**旧快照免费正确**：retract 存在 delta 文件里，delta 在 `commit_ts > system_t` 时不可见，于是「更早的 system_t 看不到这次撤销，也看不到新版本」自动成立。不需要额外的 MVCC 机制。

### 4.7 Compaction 契约

位置化 retract 的唯一代价在这里，必须写死：

1. **输入**：目标 snapshot 的 base 段 + 可见 delta + retract 集。
2. **输出**：新 base 段（只含 live 行）+ **重写后的 retract 集**。
   - 指向「被压缩范围」的 retract → **全部丢弃**（已物化进新段）。
   - 指向范围外的 → 原样保留（target 段未变，位置仍有效）。
3. **不允许部分压缩**：某段的全部 retract 必须与该段全部行一起处理，否则位置空间失效（悬空 retract 会静默指向新段的无关行——这是最危险的失败模式，必须由 `segment_id` 版本号或「段一旦参与压缩就整体作废」来防）。
4. **旧文件延迟 GC**：旧段、旧 delta 从新 manifest 摘除引用，但文件保留到无 snapshot / 无 mmap reader 引用。
5. **等价性**：压缩前后，同一 `(business_t, system_t, lineage)` 必须返回完全相同的边集。这一条要写成属性测试，而不是靠人看。
6. **v1 不合并相邻区间**：即使压缩后 `[D1,D5)` 和 `[D5,D8)` 同值可以并成一段，v1 也不并。理由：一旦合并，「压缩是恒等变换」就不再成立，第 5 条无法用边集相等来验证，只能改成「语义相等」，验证强度立刻下降。这是有意的取舍。

### 4.8 区间汇总语义（必须现在定）

切分之后，一条逻辑边在 `[T1,T2)` 内可能对应多行。所以：

- **点查** `active_at(t)`：行数 = 边数，无影响。
- **区间汇总** `active_during([T1,T2))`：**必须按 `(src, dst, edge_type)` 去重**，否则一次更正让计数从 1 变成 3。

这条去重是聚合层的，不进热循环（聚合本来就要扫全窗口）。副作用：(src,dst,type) 相同的平行边会被并成一条。这是 v1 的已知限制，写进文档；要真多重图必须加 edge id，但那是另一个功能（§7）。

---

## 5. 对 L1 / L2 / L3 的影响

### 5.1 L1：线性更正可以沿用现有 manifest，分支不行

现有 `Manifest`（`store.rs:165`）是单条线 + 单调 `system_ts`。**线性更正在这个结构里是自洽的**：所有历史段都还在 `segments` 里，`graph_at(t)` 用 `system_ts <= t` 就能回到旧答案；retract 放在 delta 里，同样被 `commit_ts` 挡住。

但一旦要 branch（两条时间线共享前缀），就必须有 snapshot 链，而 snapshot 链会立刻暴露「`FsCatalog` 与 storage manifest 两套真相」这个老问题。**所以顺序应该是：先做线性更正（不碰 L1 结构），做 branch 之前先统一 L1。**

### 5.2 L2：只加一个 system 盒，判定式原样复用

`SegmentMeta` 的六个标量（`valid_from_min/valid_to_max/max_valid_from/min_valid_to/node_id_min/max`）对 delta 完全适用。新增的只有：

- `DeltaEntry.system_from`（= `commit_ts`，单调，可直接用标量比较）
- 可选 `system_to`（该 delta 是否已被后续提交完全取代）——**v1 不加**，因为它只影响剪枝率，不影响正确性，而它的失效判定必须跨全部后续 delta，属于优化不是功能。

`PruneOutcome` 三态（`Skip/FullHit/Scan`）不变。delta 的 `FullHit` 同样成立：若 delta 全部行在窗口内活跃，答案就是 `put_count` 减去该窗口内的 retract 数。

### 5.3 L3 与图漫步：零改动，但要注意统计口径

`khop_*` 消费的是「一个 CSR + 业务时间」。只要物化（`graph_at`）先把 retract 折掉，图漫步天然正确，**一行都不用改**。

唯一要注意的是 L3 的度统计：切分会让同一逻辑边变成多行，`TemporalCsrStats::edge_count` / `degree_histogram` 数的是**行**。所以 **L3 必须从「某个业务时点的视图」构建，而不是从原始段构建**——否则一次更正就会把平均度抬高 3 倍，代价模型随之失真。这是 L3 实现时的一个硬约束，不是可选项。

---

## 6. 分阶段实施

| 阶段 | 内容 | 验收 |
|:--|:--|:--|
| **U0 语义层** | `Mutation`/`Retract`/`EdgeVersion` 类型；`correct()` sweep；提交期冲突校验；retract 集的内存解析。**先不落盘**，用现有段 + 内存 retract 集跑测试 | 属性测试：对随机边集与随机窗口做更正，`active_at(t)` 与「朴素重放全部 mutation 历史」逐点相等；同一 key 无重叠；窗口跨多版本正确 |
| **U1 持久化** | WAL v2（tagged record、after-image + retract）；`.gtvd` 读写；manifest `DeltaEntry`；`graph_at` 折 retract；崩溃重放测试（torn tail / checkpoint 前崩溃 / checkpoint 后崩溃） | kill 点注入：任一阶段崩溃后重开，结果等于「未崩溃」；`graph_at(S1)` 在更正后仍返回旧答案 |
| **U2 Compaction** | base + delta + retract → 新 base；retract 集重写；旧文件延迟 GC | 压缩前后 `(tb, ts)` 边集完全相等（含被 retract 的旧快照）；悬空 retract 检测（指向不存在段的 retract 必须报错，不得忽略） |
| **U3 Branch** | snapshot 链；`main` / 命名分支 | **前置条件**：L1 与 storage manifest 统一，否则不做 |

U0 是关键：区间切分、冲突校验、跨版本 sweep 这三件事最容易写错，而且**错了之后在 U1 的二进制格式里很难看出来**（只会表现为「偶发多一条边」）。用纯内存版本把语义钉死，再谈格式。

---

## 7. 明确不做（以及为什么）

| 不做 | 理由 |
|:--|:--|
| `EdgeId`/`VersionId` 的 128-bit 列 | 位置化 retract 已满足「准确指向版本」的全部要求，且不改 base 段格式（§2.5） |
| Property Blob / `PropertyRef` / canonical property map | gtvdb 目前**没有属性存储**——`EdgeTable`/`NodeTable` 只有 `(src,dst,type,valid)` 与任意附加列，且不落 GTVSeg。做属性是独立功能，不该混进删除层 |
| Node 版本表 | 同上：没有节点持久化，谈不上节点版本。节点属性一旦落地，复用同一套 `(segment_id, row)` retract 即可，不需要新机制 |
| Branch / Merge / Rebase | 需要 snapshot 链 + 跨分支冲突解决；且会暴露 L1 两套真相。U3 且前置 L1 |
| system-time 回写（`REPLACE` 篡改历史） | 破坏审计保证。只能作为 branch 语义提供 |
| `DeltaFlags` / `EntityMask` / `OperationMask` 九个 flag | 单文件内容由 section table 直接描述，flag 位是给「不开文件就要知道内容」的剪枝用的——而那件事用 manifest 里的标量盒更便宜 |
| `mutation_ordinal` 提交内排序 | 位置化 retract 的目标永远在**旧**段，PUT 永远落在**新**段，两者不可能同段同序，提交内顺序无关紧要 |

---

## 8. 一句话交付

**现有设计要改，但改的是一个「删除层」，不是「可变图引擎」：给不可变段加一个 `(segment_id, row)` 的位置化 retract，写路径在提交期用 `BitemporalRange::overlaps` 判「该 append 还是该 retract」并做一次跨多版本的区间 sweep，读路径在物化时把 retract 折掉。CSR、`neighbors()`、L2 判定式、`khop_*` 全部不动；u128 版本号、property blob、branch 三层应当推迟。**

---

## 9. 证据索引

**代码事实（`gtvdb_core`，HEAD `233b32b`）**

- `crates/gtv-storage/src/store.rs:616` — `graph_at` 无条件并集（§3.1）
- `crates/gtv-storage/src/store.rs:537` — `entries_at_system` 判据 `system_ts <= t`
- `crates/gtv-storage/src/store.rs:165` — `Manifest` 无 snapshot 链（§3.5）
- `crates/gtv-storage/src/wal.rs:77` — `EdgeRecord` 无边身份（§3.2）
- `crates/gtv-storage/src/wal.rs:272` — 未知 op 硬报错（§3.4）
- `crates/gtv-storage/src/store.rs:305` — 重放只接受 `OP_INSERT`
- `crates/gtv-core/src/csr.rs:97` — `EdgeRow` 五元组
- `crates/gtv-core/src/csr.rs:497` — `neighbors` 只接业务时间（§3.3）
- `crates/gtv-core/src/bitemporal.rs:120` — `overlaps`（现为死代码，§1.2）
- `crates/gtv-storage/src/segment.rs:90` — `HEADER_SIZE = 4096`，`ALIGN = 64`

**需求稿**

- `docs/delta-segment.md` §4.1（after-image / patch 位置）、§13.1（retract 指版本身份）、§8（单版本 split）
- `docs/update-segment.md` §8（`correct_interval` 伪代码）、§11–12（snapshot 链）、§19（七条不变量）

**同类实现**

- Apache Iceberg `POSITION_DELETES`（位置删除，与 §4.2 同构）
- 上一轮结论：`docs/metadata-design-comparison.md`（L1 单一权威 / L2 OB skip index / L3 只存可加可并量）
