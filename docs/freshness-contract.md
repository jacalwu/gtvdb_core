# 回执的新鲜度契约（业务时间 / 认知时间 / 缺失判定）

> 适用：`gtv-core`（`QueryReceipt`、`Freshness`、`Absence`）、`gtv-engine`（`GtvContext::set_freshness*`）、`gtv-storage`（`SegmentStore::freshness`）
> 状态：v0.1.0 已实现；刷新排程、SLA、CDC 适配均**不在**本文范围
> 来源：`06_銀行三場景定位盤點與分析工作計劃_2026-11-21_v3.md` 的 G8 / G9 与验收 R1 / R2 / R3

## 1. 要解决的三个问题

一个批量刷新、带双时间的图，回答「有没有」时会同时踩三个坑：

| 问题 | 错误答案 | 正确答案 |
|:--|:--|:--|
| 报表在 15:00 生成，图在 14:45 刷新，14:30 有一笔交易 | 「无此交易」 | 「本结果截至 14:45，14:30–15:00 之间可能尚未进入」 |
| 查询的是三年前的业务时点，那段历史已被延迟 GC 回收 | 「无此交易」 | 「该时点的历史已回收，无法回答」 |
| 调用方没有传水位线 | 「延迟为 0」 | 「新鲜度未知」 |

三者都是**静默错误**：返回空集、不报错、下游当成结论。本文定义的契约让它们变成可读的返回值。

## 2. 两条时间轴

| 轴 | 字段 | 含义 | 谁填 |
|:--|:--|:--|:--|
| 业务时间 | `QueryReceipt::as_of` | 查询框定的业务时点（边区间的 `AS OF`） | 调用方 |
| 认知时间 | `QueryReceipt::freshness.as_known_at` | 本结果**包含到**哪个知识时点 | 宿主从存储或刷新水位线取 |

两者不可互换：`as_of = 150` 且 `as_known_at = 90` 是完全合法的组合，含义是「以 90 时点的知识，看 150 的业务时间」——它回答的是「截至 90 我们知道 150 会发生什么」，而不是「150 的事实」。

### 2.1 权威时钟必须显式

```rust
pub enum KnowledgeClock {
    RecordedAt,        // 入库行的提交时间（刷新管线带过来的）
    SegmentSystemTs,   // manifest.system_ts，最新已发布段
}
```

**core 不选边。** `SegmentStore::freshness()` 报 `SegmentSystemTs`，因为它物理上只知道这个；刷新管线如果带 `recorded_at`，用自己的水位线覆盖。这个选择属于 v3 计划的 Q22，未裁决前不写死。

## 3. 三个返回值，而不是一个布尔

`Absence` 把「空结果」拆成四种判定：

| 判定 | 含义 | 调用方应做什么 |
|:--|:--|:--|
| `Proven` | 搜索完整，且水位线覆盖所请求的认知时点 | 可以当结论 |
| `NotYetRefreshed { behind_by }` | 水位线落后 | 报告里标注刷新时点，或等下一轮刷新 |
| `HistoryCollected { floor }` | 业务时点低于 GC 保留下限 | 不能回答，需扩保留窗口或重放 |
| `NotApplicable` | 有行，或搜索被截断（`TriState::Incomplete`） | 缺失不是问题；截断另有 `incomplete` 说明 |

`QueryReceipt::absence_at(requested_k, business_t)` 是唯一入口；`proves_absence_at` 只是它的布尔折叠，**只在 `Proven` 时为真**。

## 4. 四条硬规则

| # | 规则 | 落实位置 |
|:--|:--|:--|
| F1 | 任何可能被读成「缺失」的结果都要带水位线 | `GtvContext::set_freshness*`；未设置时 `absence_at` 返回 `NotApplicable`，不返回 `Proven` |
| F2 | 权威时钟显式记录，不猜 | `KnowledgeClock` 为必填字段，无默认值 |
| F3 | `NoMatch` 且 `as_of < history_floor` 的回执**不允许生成** | `QueryReceipt::validate` 报错，`sql_with_receipt` 因此返回 `Err` |
| F4 | 未设置水位线 ≠ 延迟为 0 | `freshness` 为 `None`，`freshness_verdict` 返回 `None`（而非 `Current`） |

F3 是这几条里最有争议的一条：它让一个本来「能返回」的查询失败。这是刻意的——把已回收的历史报成空集，比报错更难在事后发现。

## 5. 用法

```rust
// 1) 从存储手柄取（嵌入式、单进程）
ctx.set_freshness_from_store(&store);

// 2) 刷新管线带 recorded_at 时，显式覆盖时钟
ctx.set_freshness(
    Freshness::new(high_water_mark, KnowledgeClock::RecordedAt)
        .with_history_floor(gc_floor)
        .with_business_max(business_max),
);

// 3) 查询后判断「没有」是不是结论
let outcome = ctx.sql_with_receipt("q-1", sql, Some(as_of), None, &[]).await?;
match outcome.receipt.absence_at(report_time, Some(as_of)) {
    Absence::Proven => { /* 可以当结论 */ }
    Absence::NotYetRefreshed { behind_by } => { /* 报表标注刷新时点 */ }
    Absence::HistoryCollected { floor } => { /* 无法回答 */ }
    Absence::NotApplicable => { /* 有行或已截断 */ }
}
```

## 6. 确定性

`Freshness` 的四个值全部来自存储状态，**不读系统时钟**，因此：

- 同一快照重放两次，回执 JSON 逐字节相同；
- 回执可作审计凭据（与 `QueryReceipt` 其余字段同等地位）；
- 想记录「报表实际生成时刻」，那是调用方的日志/批次号，不属于回执。

## 7. 明确不做

- 不做刷新排程、重试、水位线持久化——属刷新管线（`gtv-ingest` / 扩展仓）。
- 不定义新鲜度 SLA（分钟级 / 日终），属业务口径（v3 计划 Q13）。
- 不选权威时钟（Q22），不规定 `recorded_at` 填充责任（Q23）。
- 不改协议版本：`freshness` 是 JSON / Arrow metadata 的**新增可选字段**，v1 对端忽略即可；`QueryResponse` 的既有字段编号不变。
