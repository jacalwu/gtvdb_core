# SQL/PGQ 支持范围（conformance 表）

> 适用版本：`gtvdb_core` v0.1.0（协议 v1）
> 本文描述的是 `gtv-pgq` 的**实现范围**，不构成对 SQL:2023 的一致性声明。

## 0. 口径

「支持 SQL:2023」这句话在对外材料里必须写成：

> **支持 SQL/PGQ（ISO/IEC 9075-16:2023）的公开子集，并发布 conformance 表。**

三条限制要同时说清楚，否则第一轮技术尽调就会失分：

1. **SQL:2023 是一个多部分标准**，属性图查询是其中的第 16 部分 SQL/PGQ；说「支持 SQL:2023」会被理解为整个标准（含 JSON、多态表函数等）都支持。
2. **没有任何 DBMS 宣称完整符合 Core SQL:2023**。Oracle 23ai/26ai 原生 GA 了 SQL/PGQ，PostgreSQL 19 实现了 26 项后在 Beta 4 因 catalog 表示问题整体回退，DuckPGQ 覆盖 `ANY SHORTEST` 等更多路径特性但语法是扩展。本项目只实现 L1。
3. **GQL（ISO/IEC 39075）是另一个标准**，不是 SQL/PGQ 的超集关系，不在本文范围；`ASOF JOIN` 也不在 SQL 标准内（它是时序扩展）。

## 1. 特性矩阵

图例：✅ 支持｜△ 部分支持（有明确限制）｜❌ 不支持（报错 `PgqError::Unsupported`）

| # | 特性 | 状态 | 标准位置 | 首版本 | 限制 / 说明 | 测试 |
|:--|:--|:--:|:--|:--|:--|:--|
| 1 | `GRAPH_TABLE ( g MATCH … COLUMNS … )` | ✅ | 9075-16 §7 | v0.1.0 | 必须有 `COLUMNS`，必须有 `AS` 别名 | `gtv-pgq` 单测、`tests/pgq_l1_end_to_end.rs` |
| 2 | 单条路径模式 `(a)-[e]->(b)` | ✅ | 9075-16 §7.2 | v0.1.0 | 一个 `GRAPH_TABLE` 内仅一条路径；多条（逗号分隔）为 L2 | 同上 |
| 3 | 方向 `->` / `<-` | ✅ | 9075-16 §7.2 | v0.1.0 | 无向 `-[e]-` 不支持 | 单测 `backward_edge_swaps_endpoints` |
| 4 | 节点 / 边标签 `:Account` / `:Transfer` | ✅ | 9075-16 §7.3 | v0.1.0 | 每元素仅一个标签；`:A|B` 为 L2 | 单测 `unknown_label_is_named_in_the_error` |
| 5 | 元素内 `WHERE`（节点 / 边） | ✅ | 9075-16 §7.4 | v0.1.0 | 谓词按变量替换后下推到 join 的 `WHERE` | 单测 `predicates_and_columns_are_substituted` |
| 6 | 外层 `WHERE` | ✅ | 9075-16 §7.4 | v0.1.0 | 同上 | 同上 |
| 7 | `COLUMNS (expr AS name, …)` | ✅ | 9075-16 §7.5 | v0.1.0 | 别名必填；`var` 裸引用解析为顶点主键 | 单测 `bare_variable_becomes_the_key` |
| 8 | 固定长度多跳（2–3 跳） | △ | 9075-16 §7.2 | v0.1.0 | 由边表链式自连接实现；同一 `GRAPH_TABLE` 内需显式写多个关系模式（L1 限定单关系 + 量词） | 单测 `quantifier_expands_to_a_union` |
| 9 | 有界量词 `{1,n}`（n ≤ 3） | △ | 9075-16 §7.6 | v0.1.0 | **要求两端节点同标签**；中间节点绑定到该标签；展开为 `UNION ALL` | 单测 `quantifier_across_labels_is_refused` |
| 10 | 边类型过滤（`edge_type` 映射） | △ | 扩展 | v0.1.0 | 需 `PropertyGraphRegistry::set_edge_type`；物理表必须有 `edge_type` 列 | `labelled_edge_and_typed_edges_compose` |
| 11 | `ANY SHORTEST` / `ALL SHORTEST` | ❌ | 9075-16 §7.7 | — | L2 | — |
| 12 | 无界量词 `*` / `+` | ❌ | 9075-16 §7.6 | — | 无法保证有界完成；需显式上限 | `unsupported_features_are_named` |
| 13 | 多路径模式（逗号分隔） | ❌ | 9075-16 §7.2 | — | L2 | 同上 |
| 14 | 路径模式变量 / `PATH` 函数 | ❌ | 9075-16 §7.8 | — | L2 | — |
| 15 | `CREATE / DROP PROPERTY GRAPH` DDL | ❌ | 9075-16 §11 | — | 图定义走 `PropertyGraphRegistry` API，不进 DataFusion catalog（见 §3） | — |
| 16 | 路径模式内的聚合 / 子查询 | ❌ | — | — | 用外层 SQL 表达 | — |

## 2. 明确不做的部分

- **不 fork sqlparser、不新增执行引擎。** 改写结果就是 `SELECT … FROM … JOIN …`，交给 DataFusion 的 CBO。这是刻意的：一旦引入第二个规划器，`EXPLAIN`、统计信息、谓词下推都要重做一遍，而收益只有语法糖。
- **不把 `GRAPH_TABLE` 注册进 DataFusion catalog。** 图定义单独放 `PropertyGraphRegistry`，规避 PostgreSQL 19 遇到的 catalog 表示问题（2026-09-24 Beta 4 因此整体回退）。
- **不支持 L2 时返回错误，不返回近似结果。** `PgqError::Unsupported` 会带上特性名，便于用户改写；静默降级会让「看起来跑通了」变成最难查的一类问题。

## 3. 使用方式

```rust
use gtv_pgq::PropertyGraphRegistry;

let mut graphs = PropertyGraphRegistry::new();
graphs.add_graph("payments");
graphs.add_vertex("payments", "Account", "accounts", "id")?;
graphs.add_edge("payments", "Transfer", "transfers", "src_id", "dst_id")?;

// 与普通 SQL 同一入口；无 GRAPH_TABLE 时原样执行
let rows = ctx.sql_pgq(
    "SELECT t.src, t.dst FROM GRAPH_TABLE (payments \
     MATCH (a:Account)-[e:Transfer]->(b:Account) \
     WHERE b.status = 'open' \
     COLUMNS (a.id AS src, b.id AS dst)) AS t",
    &graphs,
).await?;
```

边关系可以直接指向引擎注册的图算子表（例如 `gtv_batch_neighbors(src, dst, edge_type, valid_from, valid_to)`），此时 PGQ 与 `as_of` / 区间边剪枝自动组合：

```rust
graphs.add_edge("payments", "Transfer", "gtv_batch_neighbors", "src", "dst")?;
```

## 4. 验收与回归

| 门禁 | 命令 | 断言 |
|:--|:--|:--|
| 改写正确性 | `cargo test -p gtv-pgq` | 每条支持特性都有单测；不支持特性必须报错 |
| 端到端等价 | `cargo test -p gtv-engine --test pgq_l1_end_to_end` | PGQ 结果与手写 join 逐批相等；`EXPLAIN` 出现基表 |
| 截断语义 | `cargo test -p gtv-engine --test p0_motif_bounded` | 标准路径遇截断报错，扩展路径带 receipt |

新增特性时，先在本表加一行并写测试，再改代码；表与测试不一致视为回归。
