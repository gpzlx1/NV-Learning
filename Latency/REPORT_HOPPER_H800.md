# NVIDIA H800 存储层次与访问延迟：依赖路径的实证分析

## 摘要

本报告在 NVIDIA H800（Hopper，`sm_90a`）上研究单条依赖操作穿过 GPU 存储层次时的完成时间。实验以指针追逐隔离 load latency，以三种口径分解 store，以地址×SM 扫描识别 L2 位置差异，并通过双卡 P2P 测试观察 NVLink 远端访问的缓存与同步行为。

结果显示，shared、L1、DSMEM peer、L2、HBM、远端 HBM 与 host-pinned path 的依赖读延迟约为 23、32、181、272、682、1680 和 2510 cycle。global store 只需约 8 cycle 即可继续发射，但 L2 store→load 为 294 cycle、跨 SM 可见性约 958 cycle、跨卡 system-scope 标志传递约 4961 cycle。L2 延迟还与 SM 和物理地址共同相关，单地址的 global atomic 在本机可相差近 2 倍。

这些结果说明：CUDA 地址空间不能代替硬件命中层级；store issue 不能代替完成或可见性；峰值 NVLink 带宽也不能代表细粒度远端依赖的成本。对于指针追逐、锁、原子和细粒度通信，必须按实际路径预算延迟；对于规则数据并行，应转向吞吐、MLP 和数据复用分析。

## 1. 研究问题

本报告回答四个递进问题：

1. 一次依赖 load 命中不同层级时分别要等待多久？
2. store 的发射、读回和跨执行单元可见为何不能用同一数字描述？
3. L2 延迟是否仅由缓存命中决定，还是还与地址和 SM 位置相关？
4. 远端 GPU 数据经 NVLink 访问时，本地缓存和 system-scope 同步如何影响延迟？

这里研究的是 **latency**，而不是单位时间内大量独立请求的完成率。结论主要适用于无法用 warp/请求并发隐藏等待的关键路径。

## 2. 架构背景与方法

### 2.1 编程地址空间与物理层级

```text
Thread
 ├─ register
 ├─ local address space ───────────┐
 └─ SM                             │
     ├─ shared memory / L1         │
     ├─ constant cache             │
     └─ DSMEM (cluster peer)       │
          └──────────────┬─────────┘
                         L2 (all SMs)
                         ├─ local HBM
                         ├─ remote GPU memory via NVLink/NVSwitch
                         └─ mapped host memory
```

Shared/global/local/constant 是 CUDA 编程模型中的地址空间，L1/L2/HBM 是实际服务数据的硬件层级。local spill 仍可命中 L1/L2，global load 也可能停在 L1。CUDA Programming Guide 同样把每 SM 的 L1、全 GPU 共享的 L2 与编程存储空间区分描述 [R1]。

Hopper 的专有变化使这张通用层次图多出三个重要连接。第一，H100/H800 的 L1、texture 与 shared 使用 256 KiB unified on-chip resource，shared carveout 最高 228 KiB；L1 既是 cache，也是把 warp lane 请求聚合后送往下层的 coalescing buffer。第二，thread-block cluster 把协作范围从单 SM 扩展到同一 GPC 内的多个 SM，DSMEM 允许直接访问 peer block 的 shared memory，并可与普通 L2 访问同时使用。第三，TMA 把大块/多维 tensor 的地址生成与搬运交给异步硬件单元，而不是由整个 warp 用寄存器和普通 load/store 循环完成 [R7][R8]。

本机数据路径可据此画成：

```text
warp ─► LSU ─► 256 KiB unified L1/texture/shared capacity
                    ├─ local shared ports
                    ├─ cluster fabric ─► peer SM shared (DSMEM)
                    └─ 50 MiB distributed L2 ─► HBM

one elected thread ─ ─ TMA descriptor/address engine ─ ─► shared
128-thread warp-group ─ ─ wgmma/Tensor Core ─ ─► accumulator registers
```

图中只表示 NVIDIA 公开的逻辑单元。L2 slice 数、地址哈希和 cluster fabric 路由没有公开到足以直接画出本机物理 floorplan；这些部分必须由后文的地址×SM 实验约束。

### 2.2 依赖链测量

每个链表节点直接保存下一个节点的地址，SASS 核心形式为 `LDG... [R6]`、`LDS [R5]` 或 `LDL`。下一次地址必须等待上一次 load 返回，因此 memory-level parallelism 不能隐藏被测延迟。随机单环跨 cache line 分布，双点斜率消去计时与循环固定开销。

NCU 归属探针验证：24 KiB `.ca` 的被计时访问稳定命中 L1；32 MiB `.cg` 绕过 L1，计时区间稳定命中 L2；2 GiB `.cg` 持续产生 DRAM sector。另一个独立 kernel 的 shared/L1/L2/HBM 结果与主实现分别相同或相差不超过 0.06%。

### 2.3 写延迟的三个定义

GPU store 通常把请求交给后续队列后即可继续执行，所以必须分开测：

1. **issue interval**：独立 store 多快进入流水线；
2. **store→load round trip**：本线程多久能读到所写数据；
3. **visibility**：数据何时按指定 memory scope 对其他执行单元可见。

把三者混为“写延迟”，会把数百乃至数千周期的完成成本误写成 4–8 cycle。

## 3. 读延迟：容量层次形成清晰阶梯

| 路径 | cycle | ns（该轮频率） | 证据与含义 |
|---|---:|---:|---|
| shared memory | 23.00 | 11.62 | 纯 `LDS` 地址依赖 |
| L1 hit | 32.00 | 16.16 | 24 KiB `.ca` |
| DSMEM self | 32.19 | 16.26 | 经 DSM 接口访问本 CTA |
| local memory | 34.02 | 17.18 | spill 地址空间，背后命中 L1 |
| constant | 58.64 | 29.62 | 数据相关下标，非 uniform broadcast |
| DSMEM peer | 180.84 | 91.34 | cluster 内另一 SM 的 shared |
| L2 hit | 271.94 | 137.34 | 32 MiB `.cg` |
| local HBM | 683.80 | 345.35 | 2 GiB footprint，容量 miss |
| host pinned | 2513.40 | 1269.40 | 映射 host path |

CUDA 从 compute capability 9.0 起提供 thread-block cluster；cluster 内 block 被协同调度并可读写彼此的 distributed shared memory [R2]。DSMEM peer 需要穿过 cluster 内互连和远端 shared 服务路径，因此比本地 shared/L1 慢，但仍显著快于本机 L2 hit。

Footprint sweep 给出相同层次：32–256 KiB 约 32 cycle，1–32 MiB 约 272 cycle，64 MiB 进入部分 miss 过渡，128 MiB–2 GiB 约 640–676 cycle。第一个拐点与 Hopper 公开的 256 KiB unified data cache 相符，第二个拐点与本机 50 MiB L2 相符。由于该 benchmark 几乎不占 shared carveout，unified resource 的大部分可用于 L1；真实 kernel 若申请大量 shared memory，可用 L1 容量会随 carveout 改变。容量边界、公开资源大小和 counter 共同支持层级归属；仅凭延迟数值相近不足以命名硬件路径。

### 3.1 Cache operator 与访问宽度

32 MiB footprint 下，`.ca/.cg/.cs/.cv/.nc/relaxed` 基本为 272–277 cycle，`.lu` 在两轮正式采集中为 333–344 cycle。cache operator 主要表达缓存与淘汰语义，不保证改变已经稳定命中 L2 的 service latency；`.lu` 的 evict-first/last-use 倾向会破坏重复命中，因此该访问模式下 miss 增多。

64-bit 与 128-bit load 都约 272 cycle。向量化没有降低单次依赖命中时间；它的主要收益是每条指令承载更多数据、减少事务或循环开销，属于带宽问题。

## 4. 写路径：发射、读回与可见性逐级变贵

### 4.1 Issue interval

| 目标 | cycle/store |
|---|---:|
| shared / local | 4.14 / 4.12 |
| DSMEM peer | 4.69 |
| L1 / L2 / HBM | 8.11 |
| host pinned / remote GPU | 8.11 |

global store 的 issue interval 几乎不随最终目标变化。这不说明 L2、HBM、PCIe/NVLink 具有相同写成本，只说明执行前端把它们交给异步后端队列的速率相近。

### 4.2 Store→load round trip

| 路径 | round trip (cycle) | 对比纯读 |
|---|---:|---:|
| shared | 28.00 | +5 |
| L1 | 41.05 | +9 |
| DSMEM peer | 176.50 | 与 peer read 同量级 |
| L2 | 293.89 | +22 |
| HBM | 737.62–739.46 | 约 +56 |
| host pinned | 2169.83–2479.98 | 跨轮次波动明显 |

往返总体由“读回来”支配。HBM 差分测试每轮写新的随机 cache line，并把 B 区结果重新串回 A 区地址链，避免两条访问链并行后只测到 `max()`。

### 4.3 跨 SM 可见性

`st.release.gpu→ld.acquire.gpu` 为 957.64 cycle，`st.cg+membar.gl→ld.cg` 为 927.90 cycle，约为普通 L2 往返的 3.2 倍。memory fence/order 需要等待先前访问达到相应作用域的有序与可见条件，所以成本取决于未完成访存和 scope，不能视为脱离上下文的一条 ALU 指令。

## 5. L2 位置效应：从双峰到 `(SM, address)` 模型

### 5.1 分布现象

8 MiB footprint 的单次访问主要形成约 266/297 cycle 两峰；2 GiB 容量 miss 形成约 550–587 与 746–843 cycle 两组。均值会把这些不同路径压成一个数，掩盖尾延迟和地址位置效应。

### 5.2 地址×SM 扫描

每个地址保存自身指针，针对 132 个 SM 和 8 个相距 8 MiB 的地址采样：

| 地址偏移 | near center | far center | 差值 | near/far SM 数 |
|---:|---:|---:|---:|---:|
| 0 MiB | 259.0 | 298.5 | 39.5 | 62 / 70 |
| 16 MiB | 261.2 | 299.1 | 37.9 | 62 / 70 |
| 32 MiB | 260.1 | 301.0 | 40.8 | 77 / 55 |
| 48 MiB | 266.0 | 304.2 | 38.3 | 68 / 64 |

所有 132 个 SM 在不同地址上都出现过 near 与 far；相邻 SM 对的分类在 528/528 次比较中一致，提示该映射在本机呈 TPC 粒度。预先占用 64 MiB 后，虚拟 offset 分类与原分配仅 38.6% 一致，说明物理页分配参与映射。

证据支持的最小模型是：物理地址被哈希到不同 L2 位置，SM 到这些位置的互连距离或服务路径不同。公开资料不足以从延迟矩阵唯一重建 die 的 cache slice 数和路由，因此示意模型不能写成已确认的物理布局。

这一现象对原子尤其重要：八个地址上的 `atom.global.add` 为 263–510 cycle，相差 1.94×。单地址、单 SM 的 L2 级结果不能代表整卡，应至少报告地址样本分布。

## 6. Hopper 专有执行单元如何改变“延迟”的含义

Hopper 的 TMA、异步 barrier 与 `wgmma` 不是简单增加几条更快指令，而是把数据移动、同步和矩阵计算变成可与普通 warp 指令重叠的独立进度域。

| Hopper 机制 | 本机测量 | 架构含义 |
|---|---:|---|
| DSMEM peer load | 180.84 cycle | direct cluster path 比 L2 272 cycle 更短，但远慢于 local shared 23 cycle |
| `cp.async` L2→shared，立即 wait | 305.87 cycle | 与同步等价路径约 302 cycle 接近；LDGSTS 只有重叠时获益 |
| TMA L2→shared，立即 wait | 323.51 cycle | descriptor、transaction barrier 与完成通知形成固定成本 |
| TMA shared→global issue / complete | 6.16 / 45.20 cycle | 单线程可快速描述 128 B，数据完成在异步后端继续 |
| `mma.sync m16n8k16` | 24.17 cycle | 单 warp 同步 MMA 的依赖结果 |
| `wgmma m64n8k16` | 54.23–54.61 cycle | 128-thread warp-group 异步 MMA；当前 SASS 每条含 `WARPGROUP.DEPBAR` |

NVIDIA 对 TMA 的公开描述是：单线程发起 descriptor，硬件完成 stride、offset、boundary calculation 和数据搬运，其他线程直到真正消费数据时才等待 [R8]。因此 TMA 立即 wait 比同步 load 慢并不否定 TMA；它说明 benchmark 刻意关闭了这项硬件的核心收益——地址指令卸载和传输/计算重叠。实际 kernel 应让 elected producer warp/thread 负责搬运，让 consumer warps 在 transaction barrier 完成前执行上一 tile 的 MMA。

`wgmma` 同理。它把 128 个线程组成 warp-group 并异步驱动第四代 Tensor Core；commit/wait/dependency barrier 决定 accumulator 何时可安全复用。当前 SASS 每条后都插入 `WARPGROUP.DEPBAR`，所以“连续发射”“每条 wait”和“四组 accumulator”得到相近 54-cycle，测到的是该编译结果下的保守依赖边界，而不是 Hopper Tensor Core 的所有可实现吞吐。要研究峰值 Tensor Core，应像 Bandwidth 套件那样增加足够独立 tile 并核对 SASS，而不是把这个 latency 数倒数成 TFLOP/s。

DPX 也是 Hopper 专有的执行增强。官方 Tuning Guide 将其定义为把 add+min/max、三输入 min/max 等动态规划模式融合成原生指令 [R7]；本机 `VIADDMNMX/VIMNMX3` 依赖 latency 约 4.11 cycle、issue 约 2.04 cycle。其收益来自减少指令序列和中间依赖，不代表任意 Smith–Waterman/Floyd–Warshall kernel 会自动达到官方应用级加速比。

## 7. 跨设备访问：峰值互连带宽不能代替依赖延迟

本机 P2P 使用 GPU 6 计算、GPU 7 放置内存，经 NVSwitch 连接。`cudaPointerGetAttributes` 验证 allocation 属于 GPU 7，测试期间 NVLink Rx 增长约 213 MiB。

### 7.1 远端数据的本地缓存行为

| Footprint | local read | remote read |
|---:|---:|---:|
| 24 KiB | 32.0 | 32.0 |
| 32 MiB | 276.1 | 1680.3 |
| 2 GiB | 684.4 | 1704.6 |

24 KiB 重复访问表现为本地 L1 hit；32 MiB 虽小于本机 50 MiB L2，远端延迟却与 2 GiB 接近。结合 NVLink counter，数据支持远端行在该测试中可被 L1 重用，却没有表现出本地 L2 容量缓存。这里描述的是本机路径行为，不外推到所有 Hopper 拓扑或 cache operator。

### 7.2 细粒度远端同步

| 操作 | local (cycle) | remote (cycle) |
|---|---:|---:|
| 32 MiB dependent read | 276.1 | 1680.3 |
| store issue | 8.11 | 8.11 |
| store→load | 319.2 | 1786.98 |
| atomic add | 341.5 | 1833.31 |
| system-scope flag handoff | — | 4961.19 |

远端 store 仍可快速发射，但任何需要结果返回或 system-scope 可见性的路径都要支付完整互连往返。NVLink 的高流式带宽依赖许多并行 packet；单条依赖链无法用这些并发隐藏延迟。跨卡算法应批量化通信、在本地聚合原子、减少细粒度锁和 flag ping-pong。

Hopper Tuning Guide 说明第四代 NVLink 每 link 双向 50 GB/s、H100 家族最多 18 links；本机拓扑报告 `NV18` 并由 Rx counter 确认流量 [R7]。这些是链路吞吐能力，不是一次依赖访问的 latency。1680-cycle 远端读说明 packet routing、远端 memory service、返回与本地 fill 的串行关键路径仍然很长；只有大量独立请求才能逼近链路 GB/s。

## 8. 方法学审计

微基准最危险的失败模式不是报错，而是编译器与硬件给出“看起来合理”的错误数字。本套件曾实际遇到：依赖加法链被强度削减、uniform shuffle 被删除、幂等操作对被折叠、store-to-load forwarding、循环不变量外提、死 store 删除，以及两条链意外并行。

因此接受一个数据点至少需要：

- SASS 中目标指令数量和依赖关系成立；
- 被测结果由 checksum/哨兵消费；
- 往返不短于对应单程；
- footprint 与 cache operator 能隔离目标层级；
- 关键层级有独立 kernel 或 NCU counter 交叉验证；
- L2/atomic 等位置敏感项目采样多个地址。

在 1.980 与 1.590 GHz 两个 SM 频率下，SHF/IMAD/FFMA、shared、DSMEM self 和 L1 的 cycle 数逐位相同，而 L2/HBM cycle 随 SM 频率变化。这既验证了片上依赖链的稳定性，也提醒 L2/DRAM 跨时钟域结果必须同时记录频率，不能只报 cycle 或只报 ns。

## 9. 工程含义与边界

1. Global/local 是地址空间，不是固定 latency；先定位实际命中层级。
2. 对写路径先确定需要 issue、read-back 还是 visibility；三者可相差两个数量级。
3. L2 级延迟应采样地址和 SM；均值不能代表尾延迟。
4. DSMEM 适合 cluster 内共享，但 peer access 不是本地 shared 的同义词。
5. NVLink 更适合批量传输而非细粒度依赖；system-scope 同步应按微秒级预算。
6. 本报告是低并发 latency 研究。规则数组 kernel 的最终速度还需结合 [`Bandwidth/`](../Bandwidth/) 中的 MLP、coalescing 与吞吐结果。

本报告未覆盖 host launch latency、满载下 latency inflation、H800 的 L2 persisting policy，以及更多 TMA tensor-map 形式。

## 10. 参考与复现

- [R1] NVIDIA, [CUDA Programming Guide — GPU Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#gpu-memory).
- [R2] NVIDIA, [CUDA Programming Guide — Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-block-clusters).
- [R3] NVIDIA, [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/).
- [R4] Luo et al., [Benchmarking and Dissecting the NVIDIA Hopper GPU Architecture](https://arxiv.org/abs/2402.13499), IPDPS 2024.
- [R5] Luo et al., [Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis](https://arxiv.org/abs/2501.12084).
- [R6] [HPMLL/NVIDIA-Hopper-Benchmark](https://github.com/HPMLL/NVIDIA-Hopper-Benchmark).
- [R7] NVIDIA, [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/).
- [R8] NVIDIA, [Hopper Architecture In-Depth — TMA and Asynchronous Execution](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).

单卡原始输出、P2P 输出和 NCU/SASS 存档见 [`results/`](results/)。复现命令与编译器风险清单见 [`README.md`](README.md)。
