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

| gtvdb_core | 协议 | Arrow | DataFusion | Rust MSRV | 备注 |
|:--|:--|:--|:--|:--|:--|
| 0.1.x | v1 | 59 | 55 | 1.82 | 当前（回执无 `freshness`） |
| 下一版（未发布） | v1 | 59 | 55 | 1.82 | 回执含 `freshness`；协议仍 v1，0.1 对端忽略新键即可 |
| 0.0.x | v0 | 59 | 55 | 1.82 | `Execute` only |

扩展仓升级路径：改 `Cargo.toml` 里的 git rev → `cargo test` → 若编译失败，先看本文件 §3 是否命中破坏性改动，再决定是否同步改扩展代码。CI 的 `core-boundary` 继续阻止 core 依赖领域 crate。

## 7. 发布基线检查清单

发布前逐项确认：

- [ ] `cargo test -p gtv-core -p gtv-proto -p gtv-pgq` 全绿；
- [ ] `cargo test -p gtv-engine --test p0_motif_bounded --test pgq_l1_end_to_end` 全绿；
- [ ] `docs/sql-pgq-conformance.md` 与 `gtv-pgq` 测试一致（新增特性先加表行）；
- [ ] 本文件 §5 的协议版本与 `gtv-proto` 常量一致；
- [ ] 变更日志已写；破坏性改动已按 §4 走废弃；
- [ ] 未在 core 引入领域名词或 quantum 依赖（`core-boundary` 与文档扫描）。

## 8. 变更日志

| 版本 | 日期 | 变更 |
|:--|:--|:--|
| 0.1.0 | 2026-11-20 | 首次定义：回执/证据契约、协议 v1、SQL/PGQ L1；M8/M9 反转为在范围 |
| 未发布 | 2026-11-21 | 回执新增 `freshness`（业务时间/认知时间分离，见 `docs/freshness-contract.md`）；新增 `Freshness` / `KnowledgeClock` / `FreshnessVerdict` / `Absence`；`SegmentStore::freshness()`；`GtvContext::{set_freshness,set_freshness_from_store,clear_freshness,freshness}`。JSON 与 Arrow metadata 各新增可选键（`as_known_at` / `knowledge_clock` / `history_floor`），**协议版本仍为 v1** |
