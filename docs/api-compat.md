# 公开 API 与协议兼容策略

> 适用：`gtv-core`、`gtv-engine`、`gtv-proto`、`gtv-pgq`（`gtvdb_core` 工作区）
> 状态：v0.1.0，尚未发布到 crates.io；扩展仓目前以固定 git rev 依赖。

## 1. 为什么要写这份文档

`ROADMAP.md` 把 M8/M9 从主线下架改为在范围（见 `docs/decisions/2026-11-core-enablement.md` D2）。一旦有外部用户 `cargo add`，任何一个无意的小改动都会变成别人的编译错误。这份文档定义「什么算破坏性」「多久废弃」，让版本号有意义。

## 2. crate 角色与稳定级别

| crate | 角色 | 稳定级别 | 破坏性改动窗口 |
|:--|:--|:--|:--|
| `gtv-core` | 内核数据结构、时序 CSR、候选集、回执/证据契约 | **稳定** | 仅次版本号提升 |
| `gtv-engine` | DataFusion SQL 面、算子注册、会话 | **稳定** | 仅次版本号提升 |
| `gtv-proto` | 线协议 v1（gRPC + Arrow IPC） | **稳定** | 协议版本号提升 + 一个兼容周期 |
| `gtv-pgq` | SQL/PGQ L1 改写前端 | 稳定（范围见 conformance 表） | 仅次版本号提升 |
| 其余 `gtv-*`（storage / index / accelerator / array / params / catalog / pattern / ingest / delta / udf / cli） | 实现细节 | **不承诺** | 随时可改，不写变更日志 |

扩展仓只应依赖上表前四个；依赖其余 crate 视为「同仓协作」，升级时需一起改。

## 3. 什么算破坏性

对上述稳定 crate，以下都算：

- 删除或重命名公开类型 / 函数 / 字段 / 枚举变体；
- 给公开结构体加**必填**字段（Rust 结构体字面量构造会失败）；
- 改变函数签名（含参数顺序、`Result` 的错误类型）；
- 改变已文档化的语义（例如 `TriState` 的三态含义、`QueryReceipt` 的字段含义、`edge_idx` 与 `edge_pos` 的区别）；
- 改变序列化格式（`QueryReceipt` 的 JSON / Arrow metadata key），且未提供迁移读取；
- 提升协议版本并要求对端同时升级（不允许无兼容窗口的强升级）。

以下**不**算：

- 新增公开类型 / 函数 / 枚举变体（在 `#[non_exhaustive]` 之外的新增仍需谨慎）；
- 新增可选字段并带默认值；
- 性能优化、内部重构、日志与错误文案调整；
- 新增 `gtv-pgq` 支持的 SQL/PGQ 特性（conformance 表增行，属能力扩展）。

一处需要说清的边界：`QueryReceipt` 的字段是 `pub` 的，给带 `Option` 的新字段加默认值，对**序列化**是兼容的（旧 JSON 缺该键即 `None`），但对**结构体字面量**构造是编译错误。因此 `QueryReceipt` 只保证「经由 `new()` + `with_*` 构造」这一条路径的兼容性；外部直接写字面量不在承诺内（等价于 `#[non_exhaustive]` 的效果，但不必真的加该属性）。`QueryReceipt` 的构造路径本身就是「先 new 再链式设置」，这也是本文档推荐写法。

## 4. 版本与废弃

- 采用语义化版本；`0.x` 期间次版本号提升可含破坏性改动，但必须在本文件「变更日志」留一行。
- 废弃流程：标注 `#[deprecated(since = "x.y.z", note = "…")]`，**保留至少一个次版本号**，并在 `docs/` 给出替代写法。
- 已废弃项示例：`gtvquery::QueryService::Execute`（v0 协议入口）——保留一个兼容周期，v1 客户端走 `ExecuteQuery`。

## 5. 协议版本策略

| 项 | 当前值 | 规则 |
|:--|:--|:--|
| `PROTOCOL_VERSION` | 1 | 新增可选字段不提升；语义变化或新增必填字段才提升 |
| `MIN_SUPPORTED_PROTOCOL` | 1 | 提升 `PROTOCOL_VERSION` 后，此处至少保留一个发布周期 |
| v0 兼容 | `Execute` RPC | 保留一个周期；`ExecuteQuery` 收到 `protocol_version = 0` 时返回 `UNSUPPORTED_VERSION` |
| 版本不重叠 | 拒绝并给出可读原因 | 客户端不得静默降级到「能连上但读错」的状态 |

`QueryReceipt` 同时出现在 `QueryResponse.receipt_json` 与 Arrow schema metadata（`gtv.receipt.json`）。**两者必须一致**；只改一处视为破坏性改动。

## 6. 兼容矩阵

Rust MSRV 不是单一数字：它由每个 crate 的依赖闭包决定，**并且必须与实际锁定的依赖版本一致**，而不是与编写代码时用的版本一致。下表是 `cargo metadata --format-version 1` 对 `Cargo.lock` 逐个 crate 求依赖闭包上界的结果（2026-11-23 核实）；每个 crate 的 `Cargo.toml` 现已写入对应 `rust-version`，`rust-toolchain.toml` 把开发环境钉在完整 workspace 的下界。

| crate（组） | 真实 MSRV | 决定它的依赖 |
|:--|:--|:--|
| `gtv-core` / `gtv-array` / `gtv-index` / `gtv-delta` / `gtv-pattern` / `gtv-accelerator` / `gtv-params` / `gtv-pgq` | 1.85（workspace 基线） | `arrow 59.2`（MSRV 1.85）；`gtv-pgq` 自身更低，随基线 |
| `gtv-storage` / `gtv-catalog` / `gtv-index-store` / `gtv-ingest` / `gtv-cli` | 1.85 声明 / 1.94 实编 | 声明继承基线；`gtv-cli` 经 `gtv-engine → datafusion 55` 实需 1.94 |
| `gtv-proto` | 1.88 | `tonic 0.14`（MSRV 1.88） |
| `gtv-engine` | 1.94 | `datafusion 55`（MSRV 1.94） |
| `gtv-udf` | **1.95** | `wasmtime 48`（MSRV 1.95）—— **这是完整 workspace 的真实下界** |

> **重要更正（2026-11-23）**：本表此前写「0.1.x = 1.82、下一版 = 1.89」。两个数字都不成立——1.82 是 `edition = 2021` 的默认值而非本项目测得的下界；1.89 只描述了 `std::fs::File::try_lock` 这一个 API，却被当成了整个 workspace 的 MSRV。实际上 1.93 连编译 `Cargo.lock` 都会被拒（`datafusion 55` 要 1.94、`wasmtime 48` 要 1.95），而 CI 用 `@stable` 永远拿最新版，所以从未暴露这个矛盾。现已用 `rust-toolchain.toml` 钉死 1.95，CI 显式校验。数据目录 `try_lock` 的 1.89 事实仍成立，但它只是下界之一，不是上界。

| gtvdb_core | 协议 | Arrow | DataFusion | 完整 workspace MSRV | 备注 |
|:--|:--|:--|:--|:--|:--|
| 0.1.x | v1 | 59 | 55 | **1.95** | 当前（回执无 `freshness`）；由 `gtv-udf → wasmtime 48` 决定 |
| 下一版（未发布） | v1 | 59 | 55 | **1.95** | 回执含 `freshness`；协议仍 v1，0.1 对端忽略新键即可。1.89 的 `try_lock` 只是下界，不构成上界 |
| 0.0.x | v0 | 59 | 55 | **1.95** | `Execute` only |

扩展仓升级路径：改 `Cargo.toml` 里的 git rev → `cargo test` → 若编译失败，先看本文件 §3 是否命中破坏性改动，再决定是否同步改扩展代码。CI 的 `core-boundary` 继续阻止 core 依赖领域 crate。

## 7. 发布基线检查清单

发布前逐项确认：

- [ ] `rustc --version` 与 `rust-toolchain.toml` 的 `channel` 一致（当前 1.95.0），`Cargo.lock` 的依赖 MSRV 不超过它；
- [ ] `cargo test -p gtv-core -p gtv-proto -p gtv-pgq` 全绿；
- [ ] `cargo test -p gtv-engine --test p0_motif_bounded --test pgq_l1_end_to_end` 全绿；
- [ ] `docs/sql-pgq-conformance.md` 与 `gtv-pgq` 测试一致（新增特性先加表行）；
- [ ] 本文件 §5 的协议版本与 `gtv-proto` 常量一致；
- [ ] 改了 `gtvquery.proto` 就跑 `python python/gen_stubs.py`（CI 用 `--check` 比对，防止 Python stub 与线协议漂移）；
- [ ] 发 Python 包前跑 `python python/check_release.py`（许可证副本、stub 新鲜度、单测、`twine check`、产物内容、PyPI 上版本是否已存在）；
- [ ] 变更日志已写；破坏性改动已按 §4 走废弃；
- [ ] 未在 core 引入领域名词或 quantum 依赖（`core-boundary` 与文档扫描）。

## 8. 变更日志

| 版本 | 日期 | 变更 |
|:--|:--|:--|
| 未发布 | 2026-11-23 | **MSRV 诚实化**：修正此前 §6 表中「0.1.x = 1.82 / 下一版 = 1.89」两处不成立声明。按 `cargo metadata` 对 `Cargo.lock` 求依赖闭包上界，逐 crate 写入 `rust-version`（基线 1.85 / `gtv-proto` 1.88 / `gtv-engine` 1.94 / `gtv-udf` 1.95）；新增 `rust-toolchain.toml` 钉 1.95.0（完整 workspace 下界，由 `wasmtime 48` 决定）；CI 从 `dtolnay/rust-toolchain@stable` 改为显式校验锁定版本。此前 1.93 下 `cargo test --workspace` 直接拒编，而 CI 的 `@stable` 掩盖了它。 |
| 0.1.0 | 2026-11-20 | 首次定义：回执/证据契约、协议 v1、SQL/PGQ L1；M8/M9 反转为在范围 |
| 未发布 | 2026-11-21 | 回执新增 `freshness`（业务时间/认知时间分离，见 `docs/freshness-contract.md`）；新增 `Freshness` / `KnowledgeClock` / `FreshnessVerdict` / `Absence`；`SegmentStore::freshness()`；`GtvContext::{set_freshness,set_freshness_from_store,clear_freshness,freshness}`。JSON 与 Arrow metadata 各新增可选键（`as_known_at` / `knowledge_clock` / `history_floor`），**协议版本仍为 v1** |
| 未发布 | 2026-11-21 | 新增可安装的 Python 驱动 `python/`（包名 `gtvdb`，DB-API 2.0 + 协议 v1 + v0 回退）与 CI 可用的 stub 一致性检查；`gtv-engine` 仅新增 dev-dependency `gtv-proto`（供 `examples/serve_demo.rs`）；`serve_demo` 按 DataFusion 错误变体映射 `INVALID_ARGUMENT`（见 §8 注） |
| 未发布 | 2026-11-21 | 修复 v0 `Execute` 无 `request_id` 时必然失败：服务端按 SQL 派生稳定 id（`v0-<fnv>`），否则经过 `validate()` 的真实 handler 会以 `invalid argument: receipt has an empty query_id` 拒绝每一次 v0 调用 |
| 未发布 | 2026-11-21 | Python 驱动进入可发布状态：版本单一来源（`gtvdb/_version.py`，`pyproject.toml` 用 `dynamic`/`attr` 读回）；许可证改为 SPDX 表达式 `Apache-2.0` + PEP 639 `license-files`（`python/LICENSE.md` 为仓库根文件的逐字节副本，PEP 639 不允许 `..`）；新增 `py.typed`、`MANIFEST.in`、`check_release.py` 预检；`.github/workflows/publish-python.yml` 以 `py-v*` 标签 + OIDC 发布。**驱动版本与 crate 版本解耦**：协议兼容性靠 `Negotiate` 协商，不靠版本号推断 |
| 未发布 | 2026-11-21 | **回执新增时点绑定**（`PointBinding{Applied,Declared}`）：`as_of` / `snapshot_id` 必须说明是"被执行"还是"仅被记录"，只是记录时不得给结论性 `tri-state`（`validate` 拒绝）。新增 `IncompleteReason::TimePointNotApplied`、`QueryReceipt::{with_as_of_applied,with_snapshot_verified,as_of_applied,snapshot_verified,has_declared_point}`、`GtvContext::{set_head_snapshot,clear_head_snapshot,head_snapshot,verify_snapshot,sql_with_receipt_binding,sql_with_attached_receipt_binding}`、`gtv_engine::QueryBinding`。`receipt_for_statement` 改收 `QueryBinding`（签名破坏性，见 §3 末段关于字面量构造的边界）。Arrow metadata 新增 `as_of_binding` / `snapshot_binding`；新字段均 `#[serde(default)]`，旧回执仍可解析。`GtvError::SnapshotMismatch` 文案改为中性的 `expected/got`。**协议版本仍为 v1**（纯增量，0.1 对端忽略新键即可） |
| 未发布 | 2026-11-21 | 语义变更（无破坏性编译影响，但会改变结果）：带 `as_of` / `snapshot_id` 且未声明执行时，`sql_with_receipt*` 的结果从 `found`/`no_match` 变为 `incomplete` + `time_point_not_applied`。此前回执把请求原样抄回，等于为未执行的时点背书（见 `docs/time-point-contract.md`）。`serve_demo` 声明 head snapshot=1，并在此之前调用 `verify_snapshot`，因此错配会得到 `SNAPSHOT_MISMATCH`（该错误码此前从未被产生过） |
| 未发布 | 2026-11-22 | **数据目录所有权（P2 地基）**：`gtv-storage` 新增 `OpenMode{rw-exclusive,ro-shared,ro-shared-locked}`、`StoreLock`、`LockOwner`、`StorageError::Locked`、`SegmentStore::{open_with,open_existing,mode,lock}`，锁文件 `OWNER.lock` + 提示文件 `OWNER.json`。**行为变更（重要）**：`SegmentStore::open` 现在取目录级排他锁，同一目录第二个写句柄会被拒绝并指出持有者——此前两个进程可同开一目录（`docs/data-directory.md`）。次序固定为「先取锁，再重放 WAL」。只读句柄的每个写路径返回「refusing to write」。`gtv-storage` 不在承诺列表（§2），但这条语义变化会改变部署行为，故记录 |
