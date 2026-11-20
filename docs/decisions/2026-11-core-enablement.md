# 决策记录：核心启用（2026-11）

> 状态：**已生效**（业主 2026-11-20 指令）
> 影响仓：`gtvdb_core`
> 取代：`ROADMAP.md` §4「M8 crates.io、M9 pyo3 主线下架」一行

本记录把 2026-11-15 产品定位设计文档里的六项裁决落成文字，避免出现「代码已发、路线图还写下架」的矛盾状态。

## D1｜core / ext 分层是**契约**，不是待建结构

- 核心仓仍只做平台：时序图内核、SQL 面、mmap 存储、扩展 ABI。
- 扩展仓经固定 git rev 单向依赖 core，CI 的 `core-boundary` 保持。
- 本轮新增的不是分层，而是**带 semver 的公开 API 契约**：`docs/api-compat.md` 定义了哪些改动算破坏性、废弃窗口多长。

## D2｜M8 / M9 反转为**在范围**

- 结论：`crates.io` 发布（M8）与语言绑定（M9）从「主线下架」改为 **Q2 目标**。
- 原因：没有可安装产物，评估在第一轮就被「装不上」淘汰；这不是性能问题，是入场问题。
- 落法（与设计文档一致，避免一次开两个 ABI）：
  1. **Python / Java 走标准协议**：`Arrow Flight SQL`（Rust `arrow-flight` 的 `flight-sql` trait、Python `adbc_driver_flightsql`、Java `org.apache.arrow.adbc:adbc-driver-flight-sql`），本仓只写薄封装；
  2. **不做 pyo3 + JNI 两套原生 ABI**（M9 按此口径反转，而非照旧实现）；
  3. Rust 侧不是新库：把 `GtvContext` 变成**版本化发布基线**（`gtv-core` / `gtv-engine` / `gtv-proto` / `gtv-pgq` 四个 crate 一起发）。
- 本轮已交付的地基：协议 v1（版本协商 + `QueryReceipt` + `TriState` + 错误码）、`gtv-proto` 的客户端/服务端实现与 7 项契约测试。

## D3｜连接模型：进程内优先，进程边界可跨越，**不做集群**

- 默认仍是嵌入式（同一进程调 `GtvContext`）。
- 允许 sidecar / loopback 服务形态（`gtv-proto::serve`），但**不承诺**集群、HA、多租户。
- 因此：服务型图数据库的评估清单（Cypher 覆盖率、ACID 集群、故障转移）继续不适用；这不等于「不能连」，而是「连的是单机引擎」。

## D4｜「支持 SQL:2023」的口径收窄

- 对外只写：**SQL/PGQ（ISO/IEC 9075-16:2023）公开子集 + conformance 表**。
- 实现范围见 `docs/sql-pgq-conformance.md`：单路径、1–3 跳、方向、单标签、内联/外层 WHERE、`COLUMNS`、有界量词（同标签端点）为 ✅/△；`ANY SHORTEST`、无界量词、多路径模式、PGQ DDL 为 ❌ 并**明确报错**。
- 不 fork `sqlparser`、不新增执行引擎；`GRAPH_TABLE` 改写为普通 join，交给 DataFusion 的 CBO。
- GQL（ISO/IEC 39075）另立议题，不与本轮混谈。

## D5｜AML 拆两层，core 零领域词

- **第一层（非量子，已在设计文档定稿）**：图特征 + 时点正确的证据链，落在扩展仓的 `gtv-ext-aml-*`；core 只提供原语（`Sel` / `edge_idx` / 预算三态 / `PathEvidence` / `QueryReceipt`）。
- **第二层（量子）**：只留适配 SPI，core **不出现** quantum 依赖、QAOA/QuEST 求解或 QPU 加速宣称（`positioning.md` §7 保持有效）。
- 本轮 core 侧交付的是第二层所需的**结构与协议**（receipt、证据引用、错误码），不是量子能力本身。

## 本轮 core 交付清单

| 项 | 落点 | 验收 |
|:--|:--|:--|
| 查询回执 / 证据契约（A1/A3） | `gtv-core/src/receipt.rs` | `cargo test -p gtv-core --test receipt_contract`（17 项） |
| 有界 motif + 严格/宽松分流（A4/C4） | `gtv-engine/src/motif.rs`、`context.rs` | `cargo test -p gtv-engine --test p0_motif_bounded`（7 项） |
| 语句级回执 + Arrow metadata 搭载（A1） | `gtv-engine/src/receipt.rs`、`context.rs::sql_with_receipt` | 同上 |
| 协议 v1 + v0 兼容（A1/B5） | `gtv-proto`（proto + lib） | `cargo test -p gtv-proto`（12 项，含 4 项 loopback） |
| SQL/PGQ L1 改写（C1/C2/C3） | `gtv-pgq` + `context.rs::sql_pgq` | `cargo test -p gtv-pgq`（14 项）+ `--test pgq_l1_end_to_end`（5 项） |

## 仍未裁决（不阻塞本轮代码，阻塞对外承诺）

1. **CRM 口径**：应统一为 Credit Risk Mitigation（信用风险缓释），`ROADMAP.md` §0 仍写作「客户关系与担保网络」——需业务负责人确认后改文档。
2. **发行渠道**：crates.io 与 Maven / PyPI 的发布节奏、包名占用、签名与供应链要求。
3. **服务端定位**：`gtv-server`（扩展仓）是否升为「受支持端点」，还是仅 sidecar。
4. **基准口径**：对外的时点正确性 / 性能数字需按「同机 A/B、非对外承诺」标注。
5. **监管辖区与报表版本**：MA(BS)28 / BCBS 239 的具体版本与辖区，影响领域层而非 core。
