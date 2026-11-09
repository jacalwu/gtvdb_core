# cuda-probe — C0 准入门禁探针

这是 [`docs/cuda-acceleration-plan.md`](../../docs/cuda-acceleration-plan.md) §3 的实测来源。
**不是产品代码**，不参与 `cargo build` / `cargo test`，只用于回答一个问题：

> 在这台机器上，把数据搬到 GPU 再搬回来，能不能被省下的计算时间覆盖？

## 测什么

| 段落 | 内容 | 为什么测 |
|:--|:--|:--|
| `[1]` | H2D（pageable / pinned）、D2H 带宽，载荷 153 MiB（1e7 边 × 2 × i64） | 所有候选的固定成本上界 |
| `[2]` | valid-time filter：CPU 扫描 vs GPU kernel，1e7 边，含冷路径与 break-even | 判断「常驻复用几次才回本」 |
| `[3]` | 512 维点积：100k 向量 × 1000 query，合并访存 kernel | TC4 KNN 的算术核；同时验证是否撞带宽墙 |

## 怎么跑

```bash
nvcc -O3 --use_fast_math -Xcompiler "/O2 /arch:AVX2" -arch=sm_89 -o probe.exe probe.cu
./probe.exe
```

Windows 上 `nvcc` 找不到 `cl.exe` 时，先执行 `vcvars64.bat`；或显式指定：

```bash
nvcc ... -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
```

并设置 `INCLUDE` / `LIB` 指向该 MSVC 与 Windows SDK 的 `Include` / `Lib` 目录。
`-arch` 按目标卡改（本机为 `sm_89`）。

## 读数时注意

1. **`[3]` 的 CPU 基线不是手调实现**。它是 4 累加器循环 + `/arch:AVX2`，良好实现大约能快 4–8×，所以 15.5× 是**上界**，不是承诺。
2. **`[3]` 的 GPU 有效带宽（约 225 GB/s）接近 128-bit 总线峰值**，说明该 kernel 受带宽限制而非算力限制。加 FLOP 不会更快，只有减少读取字节才有用。
3. **`[2]` 的 break-even 依赖单次查询的 CPU 时间**。换 CPU、换数据分布、换有效边比例都会改变它，不要把它当常数。
4. 笔记本 GPU 与数据中心卡差一个数量级，本目录的数字**只对本机成立**。

## 不做什么

- 不生成、不读取任何项目数据集（图是合成构造，全部在内存里）；
- 不链接 CUB / cuGraph（只测硬件成本，不测库的性能）；
- 不产出二进制入库（`probe.exe` 应被 `.gitignore` 覆盖）。
