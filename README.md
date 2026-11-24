# gtvdb_core — 时序图分析平台（核心仓）

> **这个仓库只包含核心。** 时序图内存数据库内核、SQL 表面、可 mmap 的 GTVSeg 存储、量子状态元数据，以及「扩展接口」本身。
> 银行监管、HFT/kdb+、量子计算求解、PQC/CBOM、q 前端等**工作负载不在这里**：它们已拆到独立扩展仓 [`jacalwu/gtvdb_ext`](https://github.com/jacalwu/gtvdb_ext)，通过 `GtvExtension` 注册接入。

定位与取舍见 [docs/positioning.md](docs/positioning.md)；路线图见 [ROADMAP.md](ROADMAP.md)。

---

## 1. 代码边界

| 仓库 | 内容 | 依赖方向 |
|:--|:--|:--|
| **`jacalwu/gtvdb_core`**（本仓） | 15 个内核 lib crate + 精简 `gtv` REPL + `gtvpy-local`（进程内引擎的 Python 绑定）+ 核心文档/测试 | 不依赖任何扩展 |
| [`jacalwu/gtvdb_ext`](https://github.com/jacalwu/gtvdb_ext) | 13 个领域扩展 crate、full-pack CLI、gRPC server、领域文档 | 以**固定 commit 的 git dependency** 单向依赖本仓，经 `GtvExtension` 接入 |

本仓是从完整 workspace **拆分出来的独立仓库**，拥有干净的 git 历史：不含扩展代码、不含领域脚本、不含任何被提交的数据集（CSV / Parquet / Segment / PDF）。扩展仓把 15 个内核 crate 钉在同一个本仓 commit SHA 上，因此「内核不依赖扩展」由 Cargo 依赖源保证，而不再只是目录约定。原 monorepo `jacalwu/gtvdb` 保留完整历史、监管脚本与历史数据，不再参与两仓构建。

---

## 2. 安装与运行

最短路径不需要 Rust 工具链，也不需要 clone 本仓。

**作为客户端（Python）：一个包，两种运行方式**

```bash
pip install pygtvdb        # 发行名 pygtvdb，导入名 gtvdb
```

```python
import gtvdb

# 远程：连一个跑着的 gtv-server（gRPC 协议 v1）
con = gtvdb.connect("gtv://127.0.0.1:50051")
print(con.cursor().execute("SELECT * FROM numbers WHERE n <= ?", (3,)).fetchall())
print(con.cursor().tri_state)      # 远程才有：结果 + 可证明的「无匹配/无结论」三态

# 本地：引擎编译进 Python 进程，不需要服务端、不需要 docker
con = gtvdb.connect("gtv://local")
con = gtvdb.connect("gtv://local", local_store="C:/data/gtv")   # 带本地数据目录
print(con.cursor().execute("SELECT n, n*n AS sq FROM range(1,4) t(n)").fetchall())
```

两种方式共用同一套 PEP 249 接口、同一套类型码与异常，区别只在引擎在哪跑：

| | 本地 `gtv://local` | 远程 `gtv://host:port` |
|:--|:--|:--|
| 引擎位置 | 本进程（wheel 内含编译产物） | 独立 `gtv-server` 进程 |
| 需要 docker / 服务端 | 否 | 是 |
| `tri_state` / `receipt` / `as_of` | 无（没有 wire protocol） | 有 |
| 适合 | notebook、CI、单机分析 | 生产、多客户端共享、风控判定 |

本地模式的 wheel 是平台包（`cp39-abi3`，一个平台一个包覆盖 3.9–3.13）。
完整用法见 [python/README.md](python/README.md)，发布流程见
[python/PUBLISHING.md](python/PUBLISHING.md)。

**作为引擎（容器 / 二进制）：**

```bash
# 容器：零依赖，同时提供 REPL 与协议 v1 的 gRPC 服务
docker run --rm -it ghcr.io/jacalwu/gtvdb-core:0.1.0                       # REPL
docker run --rm -p 50051:50051 -e GTV_ADDR=0.0.0.0:50051 \
  ghcr.io/jacalwu/gtvdb-core:0.1.0 gtv-demo-server                        # 供 Python 驱动以远程方式连接
```

Linux x86_64 静态产物见 GitHub Release 的 `v*` tag（`gtv` REPL + `gtv-demo-server`，
带 SHA256）。镜像里的 `gtv-demo-server` 服务一个不可变快照用于协议开发；带真实存储的
生产服务 `gtv-server` 在扩展仓 [`gtvdb_ext`](https://github.com/jacalwu/gtvdb_ext)。


### 从源码开发（需要 Rust 1.95+，见 [rust-toolchain.toml](rust-toolchain.toml)）

```bash
# 全量回归（无网络且依赖已缓存时）
cargo test --workspace --no-fail-fast

# 启动零扩展的核心 REPL
cargo run -p gtv-cli --bin gtv

# 跑一个核心示例
cargo run -p gtv-core --example quantum_state_demo

# 本地起一个协议 v1 服务（默认 127.0.0.1:50051）
cargo run -p gtv-engine --example serve_demo
```

REPL 内：`.help` 查看命令，`.quit` 退出，其余输入直接作为 SQL 执行。

```text
gtv> SELECT * FROM gtv_segments();
gtv> .segment g /path/to/graph.seg
gtv> SELECT * FROM gtv_segment_edges('g');
gtv> .extensions            -- 核心构建下为空表
```

---

## 3. 目录结构

```text
crates/
├── gtv-core        Temporal CSR（半开区间 [valid_from, valid_to)）、bitemporal、provenance 森林、量子状态元数据、mmap 列 Col<T>
├── gtv-array       向量化列运算（asof / mavg / msum）
├── gtv-index       ANN：Flat / IVF / HNSW，统一 Metric 契约
├── gtv-index-store .gtvidx 索引持久化
├── gtv-accelerator 装置加速层：弹性显存账本 + 热度准入常驻缓存 + 精确 KNN / as-of / IVF 卸载 + 主机回退（NVRTC 运行时编译 CUDA，`--features cuda` 选用）
├── gtv-storage     GTVSeg v1 段格式、WAL、MemSegment、SegmentStore、Parquet、SnapshotStore
├── gtv-delta       LSM delta + compaction
├── gtv-pattern     时序图 pattern matching
├── gtv-catalog     目录、lineage、DQ 规则、CBO 统计
├── gtv-ingest      批次入库 + 内容定址信封
├── gtv-udf         WASM 沙盒 UDF
├── gtv-engine      DataFusion SQL 表面 + ext.rs（扩展 ABI）
├── gtv-proto       gRPC / Arrow IPC 协议定义
└── gtv-cli         精简 REPL（本仓唯一 bin，零扩展依赖）

docs/               定位、存储设计、OceanBase 对照评估、量子状态图
tools/cuda-probe/   C0 加速准入探针（CUDA C++，不参与 cargo 构建）
testcase/data/      基准数据生成器 + SHA256SUMS（数据本身不入库）
examples/           量子状态示例（ghz3.jsonl）
```

`cargo tree -p gtv-engine` 不会拉进任何领域 crate —— CI 的 `core-boundary` job 会强制守住这条单向依赖。

---

## 4. 核心能力

- **时序图**：Temporal CSR 半开区间边、zone map 剪枝、run 内二分、k-hop / direction-optimizing 遍历；`as_of` / `neighbors` SQL table function。
- **双时间**：业务时间 vs 系统时间，`bitemporal_*` SQL 表面。
- **向量检索**：Flat（AVX2+FMA）/ IVF / HNSW，filter-aware 路由 + exact rerank。
- **状态来源图**：`gtv-core::provenance` + `gtv-core::quantum_state`，用于量子 `state_id / parent_state / gate / timestamp / probability` 的 state reconstruction；`quantum_state_summary / frontier / reconstruct` SQL。
- **可 mmap 存储**：GTVSeg v1 定宽、64B 对齐、LE、不可变段；`Col<T>` 同时支持 Owned / Mapped，算子代码零改动。

### GTVSeg 读写路径（P0 + P1 已落地）

| 阶段 | 内容 |
|:--|:--|
| P0 读路径 | `open_segment()` 直接 mmap；header checksum O(1)、data checksum 可选全量校验；`register_segment()` / `gtv_segments()` / `gtv_segment_edges()` |
| P1 写路径 | `wal.rs` append-only 定宽帧（42 B/边 + blake3 帧校验和）→ `mem_segment.rs` 原生 Vec delta → `store.rs` flush 生成段、JSON manifest 检查点、崩溃重放、历史段复用、`graph_at(T)` |

崩溃一致性：崩在 manifest 之前重放全部帧（孤儿段忽略）；崩在 manifest 之后按 `checkpoint_txn` 跳过旧帧，不重复计边。写路径**零编码零压缩**，热层不压缩的理由见 [docs/oceanbase-storage-lessons.md](docs/oceanbase-storage-lessons.md)。完整设计与验据见 [docs/storage-mmap-design.md](docs/storage-mmap-design.md) §8 / §9。

---

## 5. 扩展怎么接（不在本仓实现）

本仓提供 `gtv_engine::ext::GtvExtension` ABI —— 类似 PostgreSQL extension，只是当前为静态 Rust 注册：

```rust
use datafusion::error::Result;
use gtv_engine::{GtvContext, GtvExtension, ExtensionRegistry};

#[derive(Debug)]
struct MyExtension;

impl GtvExtension for MyExtension {
    fn name(&self) -> &'static str { "my_workload" }
    fn version(&self) -> &'static str { "0.1.0" }
    fn register(&self, ctx: &GtvContext) -> Result<()> {
        // 在 ctx.session() 上注册自定义 table function / UDF / 资源
        let _ = ctx;
        Ok(())
    }
}

let ctx = GtvContext::with_extensions(
    ExtensionRegistry::new().with(MyExtension)
)?;
```

`SELECT name, version, objects FROM gtv_extensions()` 可查看已启用的扩展；核心构建下返回空表。扩展获得的是**同一个** `GtvContext`（同一批 KNN collection、资源注册表、段注册表），因此可以随时卸载而不影响内核。

---

## 6. 硬规则

1. **不提交任何生成数据**：`*.csv` / `*.seg` / `*.parquet` / `*.pdf` 一律不入版本控制；测试数据在内存或系统临时目录构造，测试末尾巡检无 CSV 落盘。
2. **基准数据靠生成器重建**：`testcase/data/gen_datasets.sh` 会依据 `SHA256SUMS` 校验，跨平台字节一致（`.gitattributes` 钉死 LF）。
3. **单向依赖**：`crates/*` 不得引用任何扩展 crate；CI `core-boundary` 检查目录树、`Cargo.toml` 依赖与 `cargo tree`。

---

## 7. 文档索引

| 文档 | 内容 |
|:--|:--|
| [docs/user-guide.md](docs/user-guide.md) | **用户手册**：菜单式索引、5 分钟上手、图/向量/段/量子/DQ 的可跑示例（输出为实测）、REPL 指令表、参数调优、报错速查 |
| [docs/product-design.md](docs/product-design.md) | **产品设计文档**：已交付功能全清单（带代码落点）、关键设计决策与被否掉的方案、性能现状、路线图与冻结项、已知限制 |
| [docs/design-review-2026-11.md](docs/design-review-2026-11.md) | **第三方评审用设计文档**：架构、存储格式、读写路径、正确性策略、已知限制、评审问题清单 |
| [docs/cuda-acceleration-plan.md](docs/cuda-acceleration-plan.md) | CUDA 边界、C0 实测（传输/filter/点积）、常驻设计与准入门禁；探针见 `tools/cuda-probe/` |
| [docs/aml-core-ext-enablement.md](docs/aml-core-ext-enablement.md) | **AML 就绪方案**：边界四问、10 项评审逐条裁定（含 3 处事实修正）、五项 P0 通用原语、ext 分层、CPU/CUDA 两个发行物的落地顺序 |
| [docs/gtvdb-cpu-cuda-cursor-development-spec.md](docs/gtvdb-cpu-cuda-cursor-development-spec.md) | CPU + CUDA 双后端开发规范（Phase A–G、API 草案、测试矩阵、DoD）；执行前提见上一条 |
| [docs/positioning.md](docs/positioning.md) | 为什么定位为 Temporal Graph Analytics Platform |
| [docs/storage-mmap-design.md](docs/storage-mmap-design.md) | GTVSeg 格式、CPU cache 布局、P0 §8 / P1 §9 实现记录 |
| [docs/oceanbase-storage-lessons.md](docs/oceanbase-storage-lessons.md) | OceanBase 压缩与 LSM 的可借鉴点及不适用处 |
| [docs/quantum-state-graph.md](docs/quantum-state-graph.md) | 状态日志 schema、概率语义、SQL 表面与边界 |
| [docs/assessment-2026-11.md](docs/assessment-2026-11.md) | A/B/C 路线评估（本仓走 A） |
| [docs/storage_design.md](docs/storage_design.md) | 早期存储设计存档 |
| [ROADMAP.md](ROADMAP.md) | 核心主线里程碑 |
| [docs/api-compat.md](docs/api-compat.md) | 公开 API / 协议兼容、逐 crate MSRV、发布基线检查清单 |
| [Dockerfile](Dockerfile) / [publish-engine.yml](.github/workflows/publish-engine.yml) | 引擎镜像与 `v*` tag 二进制发行 |
| [python/PUBLISHING.md](python/PUBLISHING.md) | Python 包的 PyPI 发布流程（平台 wheel 矩阵 + trusted publishing） |

## License

`MIT OR Apache-2.0`，见 [LICENSE.md](LICENSE.md)。
