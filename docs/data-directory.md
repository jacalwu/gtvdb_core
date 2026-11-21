# 数据目录所有权：`--data-dir`、三种打开模式与锁

> 适用：`gtv-storage`（`SegmentStore`）、`gtv-server`、`gtv-cli`
> 状态：P2 地基已落地（2026-11-22）。**未做**：当前快照入口、会话默认 pin、跨进程回收窗口。

## 1. 这份文档回答什么

「打开一个数据目录」到底是几个问题？在 P2 之前它只问了一个（`node_count` 是多少），其余三件事没有答案：

1. 谁可以写？——没有概念，任何句柄都能 `ingest` / `flush_at` / `gc`。
2. 能不能两个人同时打开？——可以。两个进程各持一份 `Manifest`、各写一份 `wal.log`、各自 checkpoint，**都不报错**。
3. 进程被 `kill -9` 之后，留下的东西会不会挡住下一次启动？——没有锁，所以这个问题当时不存在；一旦加锁就必须回答。

P2 的答案是：**目录级所有权** + **三种打开模式** + **OS advisory lock**。

## 2. 三种打开模式

| 模式 | OS 锁 | 允许 | 典型用途 |
|:--|:--|:--|:--|
| `rw-exclusive` | 排他 | 读 + 写 + 刷新 + GC | 网络模式主节点；本地单用户会话 |
| `ro-shared` | **无** | 只读，打开时固定 manifest | 报表副本、CI、只读服务 |
| `ro-shared-locked` | 共享 | 只读，且拒绝 `rw-exclusive` 同时打开 | 「不会有人在下面换资料」的副本 |

代码入口：

```rust
// 写（等价于 open_with(root, Some(n), ReadWriteExclusive)）
let store = SegmentStore::open(&dir, node_count)?;

// 读：node_count 从 manifest 读回来，操作员不必重复声明（也不容易声明错）
let replica = SegmentStore::open_existing(&dir, OpenMode::ReadOnlyShared)?;

// 读 + 共享锁
let pinned = SegmentStore::open_existing(&dir, OpenMode::ReadOnlySharedLocked)?;
```

`ro-shared` **故意不加锁**：它读的是一份已经固定的 manifest，加锁只会让一个报表进程有能力阻塞写入者，换不到任何好处。反过来，`ro-shared-locked` 才是「我要保证脚下不会被换盘子」的那个模式。

## 3. 锁是什么，不是什么

`OWNER.lock` 由 OS advisory lock 持有（Unix `flock`、Windows `LockFileEx`），且**永不 unlink**。

**为什么必须是 OS 锁**：`kill -9` 之后锁随进程消亡而释放，目录立刻可再次打开。pid 文件做不到这一点——它需要自己发明「持有者是否还活着」的判定，而 pid 复用会直接把它变成随机拒绝启动。

**为什么永不删除锁文件**：Unix 上锁属于 open file description。删掉路径后，下一个进程会创建一个**新 inode** 并锁住它，而第一个进程仍锁住旧 inode——双方都认为自己独占。留一个看起来像残留的锁文件，是正确的代价；而「残留的锁」不存在。

**`OWNER.json` 只是提示，不是权威**。它记录 pid / 主机名 / 获取时刻 / 模式，唯一用途是把拒绝写清楚（R5）：

```
data directory `D:\data\bank` is already open by pid 24104 on GB-01 (rw-exclusive, since unix 1763740800) — cannot open as rw-exclusive
```

它必须与锁文件分开，原因在 Windows：`LockFileEx` 是**强制锁**，覆盖整个文件，持有期间其他句柄连读都读不到。P2 的第一版把记录写进锁文件，结果所有拒绝信息都退化成「holder record unavailable」——这个失败在集成测试里被抓到，才拆成两个文件。

两个已知的、可接受的提示失真：获取锁与写记录之间有微秒级空窗（此时拒绝会读到上一位持有者的记录），以及 `kill -9` 会让记录停留在死者身上。两者都只影响**错误文案**，不影响开合判定。

## 4. 次序：先取锁，再重放 WAL

```text
open_with(root, node_count?, mode)
  1. StoreLock::acquire(root, mode)          <- 所有权
  2. 读 manifest（ro 模式下这就是这份句柄看到的全部）
  3. 读 delta / observation 文件
  4. Wal::open + 重放 checkpoint 之后的帧    <- 只在拿到锁之后
```

反过来的次序允许两个进程把同一批 WAL 帧各自重放进自己的 `MemSegment`，然后各自 checkpoint——帧被应用两次，后写者的 manifest 静默胜出。这一步没有别的保护机制可依赖，次序就是机制。

`ro-shared` 不取锁，因此它也有一个诚实的弱点：写入者可以在这份句柄脚下替换文件。它的承诺只有一条——**manifest 在打开时固定**，因此 `system_ts()` / `segments()` 不会中途变形。需要更强的保证就用 `ro-shared-locked`。

## 5. 只读 = 写会被拒绝，而不是「没给你 `&mut`」

只读句柄仍然完整加载 manifest / delta / WAL，只是每个写路径开头都有 `require_write()`：

| 被拒绝的调用 | 说明 |
|:--|:--|
| `ingest` / `flush_at`（含 `flush`） | WAL 与段发布 |
| `publish_delta`（含 `correct_edge` / `correct_node`） | 修正 |
| `materialize_at` / `apply_compaction`（含 `materialize`） | 重写位置空间 |
| `create_tag` / `drop_tag` | manifest 标签 |
| `publish_observation` | 观测版本 |
| `gc` | unlink 文件 |

错误形如：

```
data directory `D:\data\bank` is open ro-shared — refusing to write
```

为什么不是靠类型（`&self` / `&mut self`）？因为**只读服务器内部仍然持有 `&mut`**（GC、compaction 都是内部动作），类型的可变性不表达「这个句柄的语义是只读」。显式检查才能保证「拒绝」是句柄的属性，而不是调用者的记性。

集成测试 `p2_directory_lock.rs` 在全部拒绝之后逐字节比对 `manifest.json` 并数一遍 `segments/`：一次被拒绝的写如果仍然改了盘，比不拒绝更糟。

## 6. 服务端：`gtv-server --data-dir`

```bash
gtv-server --data-dir /var/lib/gtv/bank --addr 0.0.0.0:50051
gtv-server --data-dir /var/lib/gtv/bank --read-only            # ro-shared
gtv-server --data-dir /var/lib/gtv/bank --read-only-locked     # ro-shared-locked
gtv-server --data-dir /new/dir --node-count 1000000            # 初始化新目录（rw）
```

环境变量回退：`GTV_DATA_DIR`、`GTV_READ_ONLY=1`、`GTV_NODE_COUNT`；`GTV_ADDR` 不变。

启动时做的事：

1. `SegmentStore::open_existing`（或 `open_with(.., Some(n), ..)` 初始化）——**取锁先于重放**；
2. `register_segment_store(&store, prefix)`：段、invalidation 日志、观测版本、聚合投影一次性挂上；
3. 每个查询包一层 `ReadScope`：GC 看见非零计数就不 unlink，在飞查询与回收不再互相踩；
4. 进程存活期间持有 `Arc<SegmentStore>`，锁因此一直有效到进程退出。

**SQL 面**（这是 P2 的诚实边界，不是「数据目录 = 一张表」）：

| 对象 | 用途 |
|:--|:--|
| `gtv_segments()` | 已发布段清单（id / 时间盒 / 行数 / prune 信息） |
| `gtv_segment_edges('<prefix>_<id>')` | 读单个段 |
| `gtv_invalidations()` | 修正事件流 |
| `gtv_active_edge_count()` / `gtv_range_agg(..)` | 折叠聚合投影 |
| `as_of('<table>', ..)` | 观测版本（经 `load_observations_into` 重放） |

没有 `--data-dir` 时仍是原来的演示表（`nodes` / `edges` / `prices` / `numbers` + KNN），因此既有测试与 `serve_demo` 不受影响。

## 7. 明确的缺口（P2 之后）

| # | 缺口 | 为什么会痛 |
|:--|:--|:--|
| C3 | 没有「当前快照」入口 | 服务端无法宣告 head snapshot，于是任何 `snapshot_id` 都只能记为 `declared`（见 `docs/time-point-contract.md`）。**不能用 `manifest.system_ts` 冒充**：`flush_at` 接受重复时间戳，它不唯一 |
| C6 | `register_segment_store` 不持有 store | 外部刷新后句柄不会自己跟随；P2 的服务器是「启动时挂一次」 |
| C7 残余 | 只接了服务端查询 | CLI / 嵌入式调用仍未统一走 `ReadScope` |
| G7 | `node_count` 固定 | 本地模式第一次新增实体就会 hard error；本地模式不能先于增长策略交付 |
| C11 | 多进程读 manifest | `ro-shared` 无锁，回收与副本并存时的窗口尚未定义 |

## 8. 验收对应

| 标准 | 本轮的验证 |
|:--|:--|
| R5 第二个 `rw-exclusive` 失败且指出持有者 | `lock.rs::exclusive_lock_...names_the_holder`、`p2_directory_lock::a_refused_open_...` |
| R6 `kill -9` 后可重启 | `lock.rs::a_stale_record_never_blocks_a_restart`、`p2_directory_lock::the_lock_file_survives_and_never_blocks_a_restart`（进程内以 drop 模拟） |
| 只读不写盘 | `p2_directory_lock::a_read_only_handle_refuses_every_write`（manifest 逐字节比对） |
| 只读固定快照 | `p2_directory_lock::a_read_only_handle_keeps_the_snapshot_it_opened_with` |
| 共享锁排斥写者 | `p2_directory_lock::a_shared_locked_replica_excludes_the_writer_until_it_closes` |

真正跨进程的 `kill -9` 验证仍在扩展仓做（需要起两个进程），见 `gtvdb_ext/crates/gtv-server/tests/data_dir.rs`。
