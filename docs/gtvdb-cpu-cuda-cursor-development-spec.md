# GTVDB CPU + CUDA 双后端开发规范（供 workbuddy 执行）

> 状态：Draft for implementation  
> 适用仓库：`gtvdb-core`，CUDA 后端可作为 feature-gated crate；AML 领域能力继续放在 `gtvdb-ext-lib`  
> 目标：在不破坏 CPU 正确性、存储边界和默认构建的前提下，为 Graph、Temporal、Vector、Columnar、In-Memory 算子提供 CPU reference 与可选 CUDA 加速后端。

---

## 0. 给 workbuddy 的执行指令

你正在修改一个 Rust workspace。开始编码前必须：

1. 阅读本文件全部内容。
2. 阅读仓库中的设计文档、`Cargo.toml`、workspace crate 列表、现有测试和 CI。
3. 先输出一份“现状核查”，列出：
   - 实际 crate 名称与目录；
   - 已存在的 accelerator、CUDA、KNN、window、TemporalCSR、snapshot、Tag/GC 接口；
   - 本任务将修改和新增的文件；
   - 与本文假设不一致的地方。
4. 如果仓库事实与本文冲突，以代码事实为准，但必须在变更说明中明确记录差异，不得静默改变架构。
5. 每个阶段只提交该阶段要求的最小改动。不要顺手重构无关模块。
6. CPU 后端始终是正确性 reference。CUDA 失败、不可用、显存不足或规模不适合时，必须自动回落 CPU。
7. 默认构建不得要求 CUDA Toolkit、NVIDIA 驱动或 GPU。
8. 不得修改 WAL、manifest、publish、retract、fsync、GC、Tag 或快照选择的正确性语义。
9. 不得先写完整 CUDA kernels 再补抽象。必须按本文阶段顺序实施。
10. 每完成一个阶段，运行该阶段测试及 workspace 相关回归，并更新本文末尾的执行记录。

---

## 1. 产品与架构边界

GTVDB 的通用能力包括：

- Graph：邻居、批量 k-hop、路径和 motif；
- Temporal：business-time、system-time snapshot、as-of、window；
- Vector：flat、IVF、HNSW、filtered KNN、exact rerank；
- Columnar：Arrow、DataFusion、窗口及聚合 kernel；
- In-Memory：TemporalCSR、mmap、共享只读快照、低分配执行。

CPU 与 CUDA 是同一逻辑算子的两个执行后端，不是两套产品，不得形成不同查询语义。

### 1.1 必须留在 CPU 的边界

以下逻辑永远由 CPU 和存储层负责：

- system-time snapshot 选择；
- `live_rows_at()` 或等价版本折叠；
- WAL append/replay/checksum；
- `write_atomic`、fsync、rename、manifest；
- publish、retract、compaction commit；
- GC、Tag pin、ReadScope；
- 段级 `Skip/FullHit/Scan` 元数据判定；
- 单点短 run 的 `neighbors(src, t)`；
- Arrow `RecordBatch` 和 SQL 结果组装。

CUDA 只接收已经由 CPU 解析完成的不可变 `SnapshotView`。

### 1.2 可由 CUDA 加速的候选

按优先级实施：

1. flat batch KNN，向量设备常驻；
2. rolling/window reduce；
3. valid-time filter 与后续 GPU 聚合链式执行；
4. 大批量 k-hop，仅在规模门槛触发后实施；
5. ring3 候选生成，完整语义验证仍由 CPU 完成。

首轮不得实现 CUDA compaction、字符串 join、单点 neighbor、独立 point-in-time 查询。

---

## 2. 核心设计原则

### 2.1 CPU 是 reference，不是 fallback-only

每个加速算子必须存在相同输入输出契约的 CPU 实现。测试必须比较：

- 整数结果逐位一致；
- bitmap 逐位一致；
- 路径、节点、边集合一致，并使用确定性排序；
- 浮点值满足：`abs(a-b) <= 1e-5 * max(1, abs(a))`；
- KNN 的 Top-K ID 集合必须一致；距离允许按上述浮点容差比较；
- tie 场景必须定义稳定的二级排序，例如 `(distance, object_id)`。

### 2.2 设备内存是缓存，不是事实源

必须满足：

- CPU mmap/内存快照是权威副本；
- 每个设备对象绑定 `snapshot_id`；
- snapshot 变化后旧句柄不可继续返回结果；
- CUDA OOM、驱动错误和 kernel 错误不得影响存储正确性；
- 驱逐后自动回落 CPU或重新加载，不向用户暴露“显存不足导致查询失败”；
- 驻留期间必须通过现有 Tag/ReadScope 机制保护底层段；
- 默认设备预算为可配置项，不得在算法内写死。

### 2.3 冷路径与常驻路径必须分开测量

所有 CUDA benchmark 必须同时输出：

- CPU reference；
- CUDA kernel-only；
- H2D 时间与字节数；
- D2H 时间与字节数；
- cold end-to-end；
- resident end-to-end；
- p50、p95、p99；
- 加速比；
- 回落次数和原因。

不得只报告 kernel-only 数字。

---

## 3. 建议的 crate 与模块结构

workbuddy 必须先检查实际 workspace。若不存在同名 crate，可按现有命名规范调整，但保持职责不变。

```text
crates/
  gtv-accelerator/
    src/
      lib.rs
      api.rs
      cpu.rs
      policy.rs
      resident.rs
      error.rs
      metrics.rs

  gtv-accelerator-cuda/           # optional，默认不编译
    src/
      lib.rs
      backend.rs
      resident.rs
      kernels.rs
      error.rs
    kernels/
      knn_flat.cu
      window_reduce.cu
      valid_time_filter.cu
    ptx/
      sm_89/

  gtv-engine/
    src/
      accelerator_registry.rs
      execution_policy.rs

  gtv-index/
  gtv-array/
  gtv-core/
  gtv-storage/

tools/
  xtask/
    src/cuda_ptx.rs
```

### 3.1 依赖方向

```text
gtv-core / gtv-array / gtv-index
              ↑
        gtv-accelerator
          ↑          ↑
 CPU reference    CUDA optional
              ↑
          gtv-engine
```

禁止：

- `gtv-core` 依赖 CUDA crate；
- 存储层依赖 CUDA runtime；
- CUDA crate 依赖 AML ext crate；
- 默认 feature 间接开启 CUDA；
- CUDA 类型泄漏到稳定公共查询 API。

---

## 4. 公共 API 草案

workbuddy 应依据仓库现有类型调整名称，禁止机械复制与重复定义。

```rust
pub type SnapshotId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone)]
pub struct AcceleratorCaps {
    pub backend: BackendKind,
    pub resident: bool,
    pub max_device_bytes: Option<usize>,
    pub supports_knn_flat: bool,
    pub supports_window_reduce: bool,
    pub supports_valid_time_filter: bool,
    pub supports_batch_khop: bool,
}

pub struct SnapshotView<'a> {
    pub snapshot_id: SnapshotId,
    pub offsets: &'a [u32],
    pub dst: &'a [u32],
    pub valid_from: &'a [i64],
    pub valid_to: &'a [i64],
    pub edge_type: &'a [u16],
}

pub struct VectorView<'a> {
    pub snapshot_id: SnapshotId,
    pub vector_space_id: u64,
    pub rows: usize,
    pub dims: usize,
    pub values: &'a [f32],
    pub object_ids: &'a [u64],
}

pub trait Accelerator: Send + Sync {
    fn name(&self) -> &'static str;
    fn caps(&self) -> AcceleratorCaps;

    fn load_graph(
        &self,
        view: SnapshotView<'_>,
    ) -> Result<Option<ResidentGraphHandle>, AcceleratorError>;

    fn load_vectors(
        &self,
        view: VectorView<'_>,
    ) -> Result<Option<ResidentVectorHandle>, AcceleratorError>;

    fn release_graph(&self, handle: ResidentGraphHandle);
    fn release_vectors(&self, handle: ResidentVectorHandle);

    fn knn_flat(
        &self,
        handle: &ResidentVectorHandle,
        queries: &[f32],
        query_count: usize,
        k: usize,
        filter: Option<&SelectionBitmap>,
    ) -> Result<Vec<NeighborList>, AcceleratorError>;

    fn window_reduce(
        &self,
        input: &[f64],
        window: usize,
        op: WindowOp,
    ) -> Result<Vec<f64>, AcceleratorError>;

    fn valid_time_filter(
        &self,
        handle: &ResidentGraphHandle,
        business_t: i64,
        endpoint_mask: u64,
    ) -> Result<SelectionBitmap, AcceleratorError>;
}
```

### 4.1 API 约束

- `business_t` 不得成为 `ResidentGraphHandle` 的身份键；它只影响 active-edge bitmap。
- `snapshot_id` 必须成为图和向量驻留句柄身份的一部分。
- 图句柄与向量句柄分开管理，因为更新频率和容量不同。
- `SelectionBitmap` 应复用现有 bitmap 类型；不得创建第三套选择向量。
- API 必须支持 filtered KNN，以便 Temporal 或 Graph 候选集直接约束 Vector 搜索。
- CPU 后端也实现同一 trait，便于 parity 测试。

---

## 5. 执行策略与自动回落

新增一个纯 CPU 的策略决策器，不能把阈值散落在各 kernel 调用点。

```rust
pub struct ExecutionPolicy {
    pub prefer_cuda: bool,
    pub min_knn_work: usize,
    pub min_khop_seeds: usize,
    pub min_khop_frontier: usize,
    pub max_device_bytes: usize,
}

pub enum BackendDecision {
    Cpu { reason: DecisionReason },
    CudaResident,
    CudaLoadThenRun,
}
```

### 5.1 决策输入

- CUDA 是否编译和可用；
- 算子是否被后端支持；
- snapshot/vector space 是否已常驻；
- 输入规模；
- 预计 H2D/D2H 字节数；
- 设备预算；
- 上一次设备错误；
- 用户是否强制 CPU；
- benchmark 证明过的准入门槛。

### 5.2 回落规则

以下情况必须回落 CPU：

- 未启用 `cuda` feature；
- 没有兼容设备；
- 驱动初始化失败；
- 显存预算不足且无法安全驱逐；
- 输入规模低于准入阈值；
- kernel 或结果校验失败；
- snapshot 句柄失效；
- 向量空间、维度或 metric 不匹配。

回落需要记录结构化 reason，但默认不得把可恢复情况作为查询错误返回。

---

## 6. 驻留缓存与失效

### 6.1 缓存键

```text
ResidentGraphKey  = snapshot_id
ResidentVectorKey = snapshot_id + vector_space_id + model_version/feature_schema_id
ActiveEdgesKey    = snapshot_id + business_t + endpoint_mask
```

若当前数据模型中 `model_version` 与 `feature_schema_id` 由 `vector_space_id` 唯一确定，可只使用稳定的 `vector_space_id`，但必须在文档中写明不变量。

### 6.2 生命周期

```text
CPU materialize snapshot
→ Accelerator.load_*
→ 创建 Tag/ReadScope pin
→ 上传并创建 ResidentHandle
→ 多次查询复用
→ snapshot publish / LRU / error
→ 句柄失效
→ release device memory
→ 释放 pin
```

### 6.3 必测失效场景

1. 发布新 snapshot 后，旧句柄不得服务新 snapshot 查询；
2. LRU 驱逐后，查询结果与 CPU 一致；
3. CUDA OOM 后自动回落 CPU；
4. CUDA context/driver 错误后自动回落 CPU；
5. 销毁句柄后 Tag/ReadScope pin 释放；
6. 并发读不能使用已经释放的底层设备缓冲；
7. 同一 snapshot 不同 `business_t` 不得重复上传整图。

---

## 7. Cargo feature 和构建规则

建议：

```toml
[features]
default = []
cuda = ["dep:gtv-accelerator-cuda"]
```

必须满足：

```bash
cargo test --workspace --no-fail-fast
cargo test -p gtv-accelerator
cargo check --workspace --all-targets
```

在无 CUDA 环境下全部通过。

CUDA job 单独执行：

```bash
cargo test -p gtv-accelerator-cuda --features cuda
cargo run --release -p gtv-engine --features cuda --example bench_accelerator_gate
```

不得在普通 `build.rs` 中无条件调用 `nvcc`。

### 7.1 PTX 方案

首选：

- kernel 使用 CUDA C++；
- `cargo xtask cuda-ptx` 在有 Toolkit 的环境预编译 PTX；
- runtime 只加载内嵌 PTX；
- CUDA feature 默认关闭；
- PTX 变更必须与 `.cu` 源码同一提交；
- CI 校验 PTX 不是过期产物，可通过源码 hash manifest 实现。

workbuddy 不得自行引入新的 CUDA 框架，除非先提交依赖比较和最小探针。

---

## 8. 分阶段实施计划

## Phase A：CPU reference 与抽象，不写 CUDA kernel

### 任务

- 新增 accelerator API crate；
- 将已有 CPU flat KNN、window reduce、valid-time filter 通过 adapter 接入 trait；
- 新增 `ExecutionPolicy`；
- 新增统一错误和 metrics；
- 默认 engine 注册 CPU backend；
- 不改变现有公开 SQL 行为。

### 验收

- 现有测试全部通过；
- 新接口与直接调用现有 CPU kernel 结果逐项一致；
- 不启用任何 feature 时不新增 CUDA 依赖；
- benchmark 显示 adapter 开销，若对小查询有明显回归，保留原直接路径或内联优化；
- 不修改 WAL、GTVSeg 格式和 TemporalCSR 热循环。

---

## Phase B：CUDA 空后端、能力发现和回落

### 任务

- 新增 optional CUDA crate；
- 实现设备枚举、能力读取和错误映射；
- 实现 `CudaAccelerator`，所有未支持算子报告 unsupported 并由 policy 回落 CPU；
- 新增 `--features cuda` 构建；
- 无 GPU CI 使用 mock/stub 验证回落。

### 验收

- CPU-only CI 全绿；
- 启用 feature 但无 GPU 时查询仍正确；
- 日志/metrics 能分辨 `feature_disabled`、`device_unavailable`、`unsupported`；
- 不允许 panic、process abort 或存储错误。

---

## Phase C：驻留句柄、缓存、预算和失效

### 任务

- 实现 graph/vector resident handle；
- 实现可配置 `max_device_bytes`；
- 实现 LRU 驱逐；
- 接入 snapshot_id；
- 接入 Tag/ReadScope pin；
- 实现失效测试；
- 本阶段仍可只上传/释放，不实现业务 kernel。

### 验收

- 六类失效测试全部通过；
- 同一 snapshot 重复 load 命中缓存；
- 不同 `business_t` 不重复上传图；
- snapshot 变更不会返回旧结果；
- 释放后无悬空 pin；
- 显存预算可测试且不会变成查询失败。

---

## Phase D：CUDA flat batch KNN

### 范围

只做顺序 flat 扫描与批量 query。首轮不移植 IVF/HNSW。

### 任务

- 向量矩阵设备常驻；
- 支持 cosine/dot/L2 中仓库已经统一定义的 metric；
- 支持 batch query；
- 支持可选 selection/filter；
- GPU 生成候选，CPU 可执行 exact rerank；
- tie-break 确定化；
- 与 CPU exact reference 做 Top-K parity。

### 验收

- Top-K ID 集合一致；
- 距离符合浮点容差；
- 空输入、`k=0`、`k>row_count`、过滤后候选少于 k 均有定义结果；
- cold、resident、kernel-only 三组数字完整；
- 只有端到端达到项目设定门槛才允许默认 policy 选择 CUDA；否则保留实验 feature。

---

## Phase E：CUDA rolling/window reduce

### 任务

- 先支持一种已有 CPU reference 的数值窗口算子；
- 输入输出均使用定宽 numeric array；
- 不处理字符串或复杂 Arrow 类型；
- 支持空输入、窗口 0、窗口大于输入、NaN/Inf 策略；
- 与 CPU 逐值校验。

### 验收

- 结果逐值相等或符合浮点容差；
- 报告传输和回传字节；
- 若单独执行没有端到端收益，只允许作为“常驻链式计划”的内部算子，不得默认启用。

---

## Phase F：valid-time filter + GPU 链式执行

### 任务

- 在常驻图上计算 active-edge bitmap；
- kernel 只处理 `valid_from <= t < valid_to` 与 endpoint mask；
- system time 不进入 kernel 参数；
- bitmap 不立即回传时，可供后续 GPU reduce/KNN/filter 使用；
- 与 CPU active-edge reference 逐位比较。

### 验收

- bitmap 逐位一致；
- 半开区间边界测试通过；
- endpoint mask 测试通过；
- 单独 filter 不作为成功标准；
- `filter → aggregate` 或其他链式计划的端到端 p99 必须改善才进入生产 policy。

---

## Phase G：大规模 batch k-hop，可选

只有满足以下任一触发条件才开始：

```text
seed_count >= 10_000
或单层 frontier >= 1_000_000
```

### 范围

- 首版只实现 Push BFS；
- 不移植完整 direction-optimizing 策略；
- 不做单点 neighbors；
- 结果回传体积必须纳入成本模型；
- 必须支持 traversal budget。

### 验收

- 节点集合与 CPU reference 一致；
- 重复节点处理一致；
- 每 hop frontier 一致；
- cancellation/budget 行为清晰；
- 端到端优于 CPU 后才能启用。

---

## 9. AML 与 ext-lib 边界

CPU/CUDA 后端都只提供通用算子。以下内容不得进入 accelerator 或 core：

- `detect_smurfing()`；
- AML 监管阈值；
- 高风险国家名单；
- KYC、PEP、制裁数据适配；
- AML 风险公式；
- expected-business suppression；
- 案件工作流；
- SAR 文本；
- AML 专用 embedding 生成。

`gtvdb-ext-lib` 可以组合 core 算子：

```text
AML seed/filter
→ vector KNN
→ batch k-hop
→ motif
→ window/path aggregate
→ ext risk score
→ evidence package
```

ext 只通过稳定 API 使用 accelerator，不直接调用 CUDA kernel。

---

## 10. 测试矩阵

### 10.1 单元测试

- CPU trait adapter parity；
- policy 决策；
- cache key；
- LRU；
- snapshot invalidation；
- vector-space mismatch；
- deterministic tie-break；
- error-to-fallback mapping。

### 10.2 属性测试

随机生成小规模输入，验证：

- 时间半开区间；
- bitmap parity；
- KNN Top-K parity；
- window parity；
- snapshot 更替后无旧结果；
- 多次 load/release 不泄漏逻辑引用。

### 10.3 集成测试

```text
CPU-only build
CUDA feature + no device
CUDA feature + mock device
CUDA feature + real device
snapshot publish → stale handle invalidation
OOM → eviction/fallback
filtered KNN → graph seed handoff
```

### 10.4 Benchmark

新建统一 gate，例如：

```text
bench_accelerator_gate
  A. transfer probe
  B. flat KNN CPU vs CUDA
  C. window CPU vs CUDA
  D. valid-time filter CPU vs CUDA
  E. chained filter → reduce
  F. fallback overhead
```

每个数字必须伴随 correctness assertion。断言失败时不得输出性能结论。

---

## 11. 可观测性

建议记录以下 metrics：

```text
gtv_accel_query_total{backend,operator}
gtv_accel_fallback_total{operator,reason}
gtv_accel_resident_bytes{kind}
gtv_accel_cache_hit_total{kind}
gtv_accel_cache_miss_total{kind}
gtv_accel_eviction_total{kind}
gtv_accel_h2d_bytes_total
gtv_accel_d2h_bytes_total
gtv_accel_kernel_seconds{operator}
gtv_accel_end_to_end_seconds{backend,operator}
gtv_accel_snapshot_invalidation_total
```

不得把高基数的 `snapshot_id`、`entity_id`、`query_id` 放入 metric label。

---

## 12. workbuddy 每阶段输出格式

每个阶段完成后，workbuddy 必须按以下格式回复或写入 PR 描述：

```markdown
## Phase X 完成情况

### 更改文件
- `path`: 修改内容

### 设计决策
- 决策：
- 原因：
- 替代方案：
- 代价：

### 正确性
- CPU reference：
- CUDA parity：
- 边界输入：

### 性能
- 环境：
- CPU：
- kernel-only：
- H2D/D2H：
- cold end-to-end：
- resident end-to-end：
- p50/p95/p99：

### 测试命令
- `cargo ...`

### 测试结果
- passed / failed / ignored

### 未完成与风险
- ...
```

---

## 13. Definition of Done

一个阶段只有同时满足以下条件才算完成：

- 代码编译；
- CPU-only workspace 测试通过；
- 对应 parity 测试通过；
- 无 GPU 环境有明确回落测试；
- 新增公共 API 有文档；
- 没有改变存储正确性边界；
- 没有把 AML 领域逻辑塞入 core；
- benchmark 同时报告 CPU、传输、kernel-only 和端到端；
- 没有未解释的性能回归；
- 本文件执行记录已更新。

---

## 14. 明确禁止事项

workbuddy 不得：

1. 在默认构建中要求 CUDA；
2. 大规模重写现有 CPU kernel；
3. 为统一接口复制第三套 KNN、window 或 graph 算法；
4. 让 CUDA 成为事实源；
5. 在 kernel 中解析 system time；
6. 修改 WAL/manifest/publish 语义；
7. 仅用 kernel-only 加速比宣称成功；
8. 静默吞掉 parity 失败并回落；
9. 使用不确定排序导致 CPU/GPU 结果漂移；
10. 在没有规模门槛证据时实现 GPU k-hop；
11. 将 AML 规则、字段和阈值写入 core；
12. 一次提交同时完成多个 Phase；
13. 未运行测试就标记任务完成；
14. 生成大型 benchmark 数据并提交仓库。

---

## 15. 第一条给 workbuddy 的实际任务

只执行 **Phase A**，不要写 CUDA kernel。

具体要求：

1. 核查 workspace 与已有 CPU kernel；
2. 提出最小 crate/module 变更；
3. 实现 `Accelerator` 抽象和 `CpuAccelerator` adapter；
4. 接入一个 flat KNN、一个 window 算子、一个 valid-time filter；
5. 为三者增加直接调用 vs trait 调用 parity 测试；
6. 增加 `ExecutionPolicy` 的 CPU 决策骨架；
7. 保证默认 workspace 构建无 CUDA 依赖；
8. 运行测试并按第 12 节格式报告；
9. 不修改 AML ext、不修改存储格式、不修改 TemporalCSR 热循环。

开始编码前，先输出：

```text
A. 仓库事实核查
B. 拟修改文件
C. API 与现有类型的映射
D. 风险和不确定项
E. Phase A 实施顺序
```

得到代码事实后直接实施 Phase A；如果发现会触及 WAL、manifest、GTVSeg 格式或公开兼容性，停止该部分并记录阻塞，不要自行扩大范围。

---

## 16. 执行记录

| 日期 | Commit | Phase | CPU tests | CUDA tests | Benchmark | 结论 |
|---|---|---|---:|---:|---|---|
| 2026-11-10 | `7220bfd` `942e844` | A（CPU reference + 抽象） | 571 | N/A | 无回归（未改查询路径） | `Accelerator` / `SnapshotView` / `CpuAccelerator` / `ExecutionPolicy` / `AcceleratorRegistry` 落地；CPU 实现直接复用 `FlatIndex` 与 `gtv-array::window`，所以"参考实现"就是既有 kernel 本身 |
| 2026-11-10 | `7220bfd` | B（空后端 + 能力发现） | 571 | 4（无 GPU 时全部跳过） | `accel_state`：默认 `feature_disabled`、`--features cuda` 下 `device_ready` 且四算子均 `unsupported` | 驱动运行期加载（`LoadLibraryW` / `dlopen`），**无链接期 CUDA 依赖、无 build.rs、无 nvcc**；`FeatureDisabled` / `DeviceUnavailable` / `Unsupported` 三种原因可分辨，无 panic |
| 2026-11-10 | `7220bfd` | C（驻留 + 预算 + 失效） | 571 | 6 实机通过（RTX 4060 Laptop） | 1e7 边 257.5 MiB：冷 46.3 ms / 暖 35.7 ms（5.6–7.6 GB/s）；break-even 7.6 次 | 同快照命中 1 次上载；`business_t` 不入键；快照变更/显式失效不返回旧结果；驱逐跳过 pinned 且查询照常回答；延迟释放不悬空；2 MiB 显存粒度已实测记录 |
| 2026-11-10 | 本次 | E（阶段计时 + pinned A/B） | 583 | 8 实机通过 | `GraphLoadMetrics` 逐列：1e7 边窄列上载 29.68 ms / 7.75 GB/s，**97.3% 是 PCIe**；pack/transform/staging 三项恒 0。pinned 端到端 33.35 ms vs pageable 30.64 ms（慢 9%）。`graph_at` 1e7 边 **1 579 ms**，占冷路径 98% | 隐性复制在 materialize 而不在上载：`live_rows_at` 498 ms + `Vec<EdgeRecord>`（686 MiB AoS）折成 SoA 1 081 ms。端到端盈亏 **255 次**，上载内盈亏 4.8 次；C2 之前必须先修 `graph_at` |
| 2026-11-10 | 本次 | D（`dst` 宽度） | 577 | 7 实机通过 | 双宽度独立进程：u64 冷 45.9 ms / 稳态 34.8 ms（break-even 7.3 / 5.9 次），u32 冷 40.4 ms / 稳态 30.9 ms（6.4 / 4.9 次） | 22 B/edge 成立，但**只能由上游提供窄列**：上载时转换实测 9.5–11.2 ms 换 5.7 ms 传输，第一版实现因此删除。`ResidentGraph::dst_layout()` 是 kernel 必须读的宽度凭据 |

| 2026-11-10 | 本次 | F（列式 materialize） | 592 | 8 实机通过 | `graph_at` 1e7 边：旧实现 1 508–1 579 ms → **393 ms**（3 个存活段，k 路归并）/ **176 ms**（压实后单段，原样拷贝）；行路径 oracle 同步 1 455 ms。探针拆解：只读 102 ms、+写列 175 ms、完整归并 390 ms，**选择器约 215 ms**（3 条 run 在 src 上交错，分支预测必错） | 不再经 AoS：段内 CSR 已按 `(src, valid_from, valid_to, dst)` 有序，flush 时付过的排序不再付第二次。`tests/graph_at_columnar.rs` 用行路径逐列比对并直接断言排序不变量（9 项）。端到端盈亏 255 → 69 次（未压实）/ 34 次（压实）；压实后已贴住本机拷贝上限（同 260 MiB 的 `fs::read` 78 ms） |
**与规范的三处偏离，都有理由：**

1. **没有 `EndpointMask`。** 草案里 `valid_time_filter(…, policy: EndpointMask)` 需要"节点在 t 是否存活"，而 v1 视图只有出边，判断 `DST` 侧存活要转置图。C1 的签名是 `valid_time_filter(&ResidentGraph, t) -> Sel`；端点策略推迟到确实需要它的那一步（C4），届时视图要加节点寿命列，而不是在签名里先放一个算不出来的参数。
2. **`SelectionBitmap` 改为复用 `gtv_core::Sel`。** 草案要求"不得创建第三套选择向量"，`Sel` 就是 M1 立的那一套（带 `snapshot_id`、可 `to_mask()` 成 Arrow、即 H2D 输入）。边过滤器返回的是**边 id** 的 `Sel`（universe = edge_count），与节点候选集同一类型。
3. **cuda feature 挂在 `gtv-engine`，不在 `gtv-accelerator`。** 因为 `gtv-accelerator-cuda` 依赖 `gtv-accelerator`，若后者再 feature 依赖前者会构成 package 循环，Cargo 直接拒绝。注册表因此放在 engine（与 §3.1 的依赖方向一致）。

---

## 17. 修订：与 AML 就绪方案的衔接（2026-11-09）

评审《建议的总体边界》与本规范的交叉点集中在"候选集"上。据此修正本文件 **一处**，其余不变。

### 17.1 Phase A 增加一项：`Sel`

本文件 §4 把 KNN 过滤端口写成 `filter: Option<&SelectionBitmap>`，§8 Phase A 又只要求"抽象 + adapter"。按此执行，Phase D 会得到一个只服务 KNN 的私有位图类型，Phase F 的 `valid_time_filter` 会再造一个，恰好违反本文 §14 第 3 条。

修正：`gtv-core` 先提供一个带 `snapshot_id` 与 `universe` 的候选集值 `Sel`（三种内部表示：bitmap / dense / ranges），并把 `khop_bounded`、`execute_ann`、`find_bounded`、聚合入口四个端口统一到它。API 相应改为：

```rust
fn knn_flat(..., filter: Option<&Sel>) -> Result<Vec<NeighborList>, AcceleratorError>;
fn valid_time_filter(...) -> Result<Sel, AcceleratorError>;
```

`Sel::to_mask()` 一次转换即得到设备侧位图，Phase F 不需要新类型。

### 17.2 由此产生的顺序约束

`Sel` 必须先于 Phase B 落地，否则 CUDA 侧的驻留句柄会在 D 阶段定型后被 `Sel` 打回重做。即：

```text
A0 清理业务名词注释 → A1 Sel / edge_idx / budget 语义 → A2 抽象 → B → C → D …
```

A1 不在本文件原定的 Phase 列表里，它属于 `docs/aml-core-ext-enablement.md` 的 M1。

### 17.3 本文件未变的部分

- §1.1 CPU 边界（WAL / manifest / publish / GC / 快照选择 / RecordBatch 组装）不动；
- §2.1 parity 判据不动（整数逐位、浮点 `1e-5` 相对容差、`(distance, object_id)` 稳定 tie-break）；
- §2.3 冷路径与常驻路径分开测量不动；
- §6 驻留缓存键不动（`business_t` 明确不参与图句柄身份）；
- §7 feature 与 PTX 方案不动；
- §9 AML 边界不动，且与 `aml-core-ext-enablement.md` §1 的四问判定互为补充。
