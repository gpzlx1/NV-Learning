# NVIDIA Thor 延迟微架构：从存储层次到 tcgen05/TMEM 的实证分析

## 摘要

本报告在 NVIDIA Thor（Blackwell，`sm_110`）上测量存储、指令、同步和异步搬运的依赖完成时间。研究重点不是列出“某条指令多少周期”，而是区分三类经常被混淆的量：指令进入流水线的间隔、数据对依赖消费者可用的时间，以及数据达到指定 memory scope 的可见时间。

Thor 上 shared、L1、DSMEM peer、L2 与 device-memory path 的依赖读延迟约为 23、32、178、248 和 821 cycle。global store 约 8 cycle 即可继续发射，但 L2 store→load 为 267 cycle，GPU-scope 可见性约 617–646 cycle。第五代 Tensor Core 的 `tcgen05.mma` 稳态间隔为 44 cycle，每条 commit/barrier 后的软件可见完成为 156 cycle；TMEM store 的对应数字为 8 和 40 cycle。异步 copy 若立即等待，不比同步路径更低延迟，其价值必须来自与独立工作重叠。

由于 Thor 使用 ATS/统一内存形态，报告不把脚本遗留的 `HBM`/`PCIe` 标签直接解释成离散 GPU 物理路径；由于 L2 地址×SM 数据尚未形成稳定拓扑，报告也不沿用 Hopper 的双分区结论。所有机制解释均限制在原始输出、SASS 和公开 CUDA/PTX 语义能够支持的范围内。

## 1. 研究问题

1. Thor 的依赖访问如何跨越 shared、L1、DSMEM、L2 与系统内存路径？
2. Store、barrier、fence 和 atomic 的 issue 与 completion 有何差别？
3. Blackwell `tcgen05` 与 Tensor Memory 应如何分解稳态吞吐和软件可见完成时间？
4. `cp.async`/TMA 何时能隐藏延迟，何时只是增加控制成本？

这些问题共同服务于一个目标：为无法靠大量独立 warp 隐藏等待的关键路径建立可审计的延迟模型。

## 2. 平台与实验设计

### 2.1 环境

| 项目 | 配置 |
|---|---|
| 采集时间 | 2026-08-09（UTC） |
| GPU | NVIDIA Thor，Blackwell，compute capability 11.0 |
| SM / L2 | 20 SM / 32 MiB |
| Driver / CUDA | 595.78 / 13.3 (`V13.3.73`) |
| 内存形态 | ATS；`nvidia-smi` 不报告独立显存容量 |

CUDA 文档说明，ATS 系统可使用 host page tables 和硬件一致性，地址空间背后的迁移/访问路径由平台能力决定 [R1]。因此下文使用 device-memory path、host-pinned path 等操作性名称；源码中的旧标签只用于定位 benchmark。

### 2.2 测量与验证

- 依赖 load：随机单环指针追逐，使下一条地址依赖上一条返回；
- instruction latency：目标指令组成真实寄存器依赖链；
- issue interval：多条相互独立操作连续发射；
- completion：在操作后加入数据依赖或规定的 commit/wait；
- 时间：双点斜率消除固定开销，同轮采集 `%clock64` 与 `%globaltimer`；
- 审计：哨兵验证结果被消费，关键循环保存 SASS。

Thor 的频率在不同测试轮次约为 0.93–1.56 GHz，因此跨轮次比较以 cycle 为主。L2/device path 跨越多个时钟域，解释 cycle 时仍需结合该轮频率。

### 2.3 覆盖与跳过

| 项目 | 状态 | 原因 |
|---|---|---|
| `01/02/03/03b/05/06b/06c/07/08` | 通过 | 哨兵与运行退出码通过，关键项保存 SASS |
| `04_mem_p2p` | 跳过（77） | 当前只有一张 GPU |
| `06_tensor` | 跳过（77） | 原测试为 Hopper `wgmma`，不支持 `sm_110` |
| cluster size 16 | 单项跳过 | 当前资源组合不可启动，occupancy API 预检为 N/A |
| NCU hierarchy counter | 未取得 | 驱动返回 `ERR_NVGPUCTRPERM`；诊断原样留档 |

“NCU 无权限”降低的是路径 counter 的证据等级，不会被写成测试通过。软件 kernel 本身仍可执行，报告以 footprint、独立 probe、SASS 和重复结果建立现阶段结论。

### 2.4 Thor/CC 11.0 的公开硬件画像

NVIDIA 的 Jetson AGX Thor `deviceQuery` 示例与本机关键属性一致：20 SM、每 SM 128 CUDA cores、32 MiB L2，并明确标记 integrated GPU sharing host memory [R6]。CUDA 的 compute-capability 表为 CC 11.0 补充了 SM 内部资源上限 [R7]：

| 资源 | CC 11.0 / Thor 公开值 | 对本报告的意义 |
|---|---:|---|
| Unified data cache | 256 KiB/SM | L1/texture/shared 的总片上容量边界 |
| Shared memory | 最高 228 KiB/SM，32 banks | 容量由 carveout 配置；bank mapping 与吞吐需分开测 |
| Registers | 64K×32-bit/SM | ILP、accumulator 与 occupancy 竞争的通用资源 |
| Resident warps/threads | 最多 48 warps / 1536 threads/SM | Thor 的 latency hiding 并发预算 |
| L2 | 32 MiB/device | footprint 超过 32 MiB 后应出现 miss 台阶 |
| Tensor types | TF32/BF16/FP16/FP8/FP6/FP4/INT8/INT4 | CC 11.0 支持范围；本报告只验证 F16→F32 |

Thor 是 Blackwell 家族的集成 GPU，并不是 B100/B200。B200 的 dual-die、HBM3e、126 MiB L2 或 NVLink 配置不能迁移到这里。当前能够与 `sm_110` SASS 直接相连的 Blackwell 专有结构，是第五代 Tensor Core 与 Tensor Memory：

```text
                    ordinary CUDA path
warp ─► LSU ─► unified L1/texture/shared ─► 32 MiB L2 ─► SoC memory fabric
                  │
                  └─ cluster fabric ─► peer-SM shared (DSMEM)

                    Blackwell tensor path
global ──TMA──► SMEM(A/B) ──tcgen05.mma──► TMEM(accumulator)
                              ▲                 │
                       one issuing thread       └─ LDTM/STTM/TCGEN05.CP
```

CUTLASS 将 TMEM 定义为专用于 accumulator、并可选存放 A operand 的片上空间；`tcgen05.mma` 由单线程发起，CTA 内线程共同遵守其 TMEM 与同步协议 [R8]。这与 Hopper `wgmma` 主要以 128-thread warp-group 和寄存器 accumulator 组织控制的方式不同。

## 3. 存储层次：依赖读形成约 36 倍延迟跨度

![NVIDIA Thor memory latency ladder](figures/blackwell_thor_memory_ladder.png)

| 路径 | cycle | ns（1.126 GHz 该轮） | 机制解释 |
|---|---:|---:|---|
| shared memory | 23.00 | 20.44 | 本 SM 显式片上存储 |
| L1 hit | 32.00 | 28.43 | 每 SM unified data cache |
| DSMEM self | 32.14 | 28.56 | 经 DSM 接口访问本 CTA |
| local memory | 34.01 | 30.22 | local address space，实际命中片上 cache |
| constant memory | 39.92 | 35.47 | 数据相关索引路径 |
| DSMEM peer | 178.00 | 158.15 | cluster 内跨 SM shared 访问 |
| L2 hit | 248.35 | 220.66 | 全设备共享 cache 路径 |
| device-memory path | 821.22 | 729.65 | 超 32 MiB 后的容量 miss 路径 |
| host-pinned path | 889.89 | 790.66 | ATS/系统内存映射路径 |

CUDA 将每 SM 的 L1、所有 SM 共享的 L2 与 shared/global/local 地址空间分开定义 [R2]。因此“global memory latency”不是常量；它取决于实际命中位置。DSMEM peer 位于 L1 与 L2 之间也符合其语义：thread-block cluster 被协同调度到同一 GPC，可访问其他 block 的 distributed shared memory [R3]，但 peer 数据仍需跨 SM/cluster 互连。

### 3.1 Footprint sweep 与容量边界

![NVIDIA Thor footprint sweep](figures/blackwell_thor_footprint_sweep.png)

| Footprint | 代表延迟 | 解释 |
|---|---:|---|
| 32–256 KiB | ~32 cycle | 低延迟 cache 区 |
| 512 KiB | ~174 cycle | 过渡区 |
| 1–32 MiB | ~248 cycle | L2 容量范围 |
| 64 MiB | ~578 cycle | 超过 L2 后的部分 miss |
| 128 MiB–2 GiB | ~792–835 cycle | 稳定容量 miss 路径 |

台阶位置与 32 MiB L2 报告容量一致，独立 probe 又分别得到 L1 32.00、L2 248.38、device path 808.08 cycle。两条证据链相互支持；在获得 NCU 权限前，报告不进一步断言物理 DRAM 类型或每级 transaction 细节。

32–256 KiB 的低延迟平台还与 CC 11.0 的 256 KiB unified data cache 精确对齐。由于测试几乎不申请 shared memory，可将更多 unified capacity 用作 L1；真实 kernel 把 carveout 提高到 228 KiB 时，不应仍假定拥有相同的 L1 工作集容量。这里揭示的是 **容量共享**，不是说 `LDS` 与 `LDG` 走同一端口：本机 shared/L1 依赖延迟分别为 23/32 cycle，说明它们在 unified resource 内仍有不同的寻址与服务路径。

### 3.2 Cache operator 与访问宽度

32 MiB footprint 下，`.ca/.cg/.lu/.nc` 均约 248 cycle，`.cs` 为 230 cycle，`.cv/relaxed.sys` 为 246 cycle。单轮差异可能包含 cache state；这些 operator 表达缓存、作用域或淘汰语义，并不保证改变已命中层级的 service latency。

64-bit 与 128-bit 访问分别为 248.34/248.39 cycle。访问加宽没有降低依赖完成时间；Bandwidth 报告中 128-bit 的持续 GB/s 更高，是每条指令携带更多有效数据，而不是单次返回更快。

### 3.3 L2 地址×SM 分布不支持照搬 Hopper 拓扑

20 个 SM×8 个地址的延迟总体约 240–269 cycle，两组中心差约 9–12.5 cycle；只有 4/20 个 SM 在八地址之间同时出现分类翻转，相邻 SM 对一致率为 69%。

H800 上同类实验可形成稳定、可复现的 near/far 分类，但 Thor 样本既没有相同峰距，也没有相同翻转结构。现阶段只能确认地址/SM 之间存在离散，不能宣称 Thor 拥有与 Hopper 相同的“TPC 粒度双分区”。要识别 Thor 物理拓扑，还需更多地址、重复物理分配、聚类稳定性和硬件 counter。

L2 被所有 SM 共享并不表示它是一块等距离 SRAM；现代 GPU 通常需要把大容量 L2 与 memory-controller 接口物理分布。地址哈希、SM 所在位置与 NoC 路由都可能形成延迟差异，但公开 Thor 文档没有给出 slice/fabric floorplan。因而“物理分布是合理候选机制”，“具体双分区/TPC 映射已证实”则不是当前证据允许的结论。

## 4. 写路径：后端解耦使 issue 远快于 completion

### 4.1 发射与本线程读回

| 路径 | store issue (cycle) | store→load (cycle) |
|---|---:|---:|
| shared | 4.03 | 28.00 |
| local | 4.03 | — |
| DSMEM peer | 4.17 | 176.00 |
| L1 | 7.95 | 41.05 |
| L2 | 7.95 | 266.59 |
| device-memory path | 8.03 | 911.85（差分） |
| host-pinned path | — | 1012.25 |

global store 可在约 8 cycle 后继续发射，说明前端与目标路径解耦；数据真正可由依赖 load 使用时，仍支付目标层级的传输与排序成本。性能模型若只采用 issue interval，会严重低估 producer-consumer 链、原子或同步通信的关键路径。

### 4.2 Memory scope 的可见性成本

| 语义 | cycle |
|---|---:|
| 普通 L2 store→load | 266.59 |
| `st.cg + membar.gl → ld.cg` | 616.97 |
| `st.release.gpu → ld.acquire.gpu` | 646.22 |

Fence/acquire-release 需要先前访问按 GPU scope 有序和可见，成本取决于尚未完成的访存，而不是一条固定流水线延迟。因此，应只在真正需要的作用域上同步，并把数据交换批量化。

## 5. 标量、同步与原子：依赖链和吞吐仍须分开

### 5.1 标量指令

| 指令 | dependency latency | issue interval |
|---|---:|---:|
| SHF / LOP3 / IMAD | 4.11 | 2.02 |
| FADD / FMUL / FFMA | 4.11 | 1.02 |
| FP64 DADD / DFMA | ~63.8 | 64.0 |
| POPC | 18.00 | 8.01 |
| SHFL | 26.00 | 4.39 |
| REDUX | 44.05 | 11.29 |
| DPX `viaddmax` | 8.11 | 4.07 |
| DPX `vimax3` | 11.73 | 5.41 |

FADD/FFMA 的 4-cycle dependency 与约 1-cycle issue 可以同时成立：流水线可每周期接收独立操作，但同一结果链必须等四周期。原 `vimax3(x,const,const)` 会快速达到固定点，ptxas 删除后续链；正式实现改用依赖前值的输入，SASS 保留连续 `IMNMX`。

### 5.2 同步与原子

| 操作 | cycle |
|---|---:|
| `barrier.sync` | 14.15 |
| `membar.cta` / `membar.gl` / `membar.sys` | 8.23 / 267.12 / 460.51 |
| mbarrier arrive+wait | 46.28 |
| cluster barrier relaxed/release | 73.67 / 340.77 |
| shared atomic add | 34.35 |
| global atomic add / CAS | 267.55 / 274.58 |
| global/shared RED issue | 6.11 / 4.14 |

同一 global atomic add 在八地址上为 247.62–270.63 cycle，最大/最小 1.09×；Thor 的位置离散明显小于本仓库 H800 的近 2×，再次说明不能跨架构照搬 L2 地址模型。

## 6. tcgen05 与 Tensor Memory：稳态间隔不是单条完成时间

### 6.1 第五代 Tensor Core

`06b_tcgen05` 使用 `sm_110f`，测试 `tcgen05.mma.cta_group::1.kind::f16` 的 M64N{8,16,32}K16，F16 输入、TMEM 中 F32 累加。SASS 核心为 `UTCHMMA`，提交/完成路径为 `UTCBAR`。

![Blackwell tcgen05 and TMEM costs](figures/blackwell_thor_tcgen05_tmem.png)

| 测量口径 | cycle/MMA |
|---|---:|
| 连续 overwrite，末尾一次 commit/wait | 44.23 |
| 同一 accumulator 累加，末尾一次 commit/wait | 44.23 |
| 每条 MMA 后 commit+mbarrier wait | 156.13 |

N=8/16/32 三种 shape 在这三种口径下相同。44 cycle 是当前 SASS 和 shape 下的稳态可见间隔，不能简单等同于数学结果完成；156 cycle 才包含每条操作的软件完成通知。PTX ISA 将 `tcgen05.mma` 与 `tcgen05.commit` 分开定义，也要求通过异步完成机制观察结果 [R4]。

从硬件数据流看，这三个结果相同具有两层含义。第一，N=8→32 时一条 UTCHMMA 承载的 FMA 数增加，而 issuing thread 的可见间隔不变，说明这些 shape 的更宽数学工作由 Tensor Core 内部并行资源承接，没有线性增加 front-end issue cost。第二，overwrite 与同一 TMEM accumulator 累加相同，只能说明 accumulator hazard 没有在 44-cycle 稳态上产生额外反压；依赖仍由异步 Tensor Core/TMEM pipeline 维护，不能据此声称 MMA 已在 44 cycle 完成。

这正是 Blackwell TMEM 的架构目的：把大 accumulator tile 从普通 register file 分离，避免 Tensor Core 结果长期占用每线程寄存器，并让单一 issuing thread 用异步完成协议管理矩阵操作 [R8]。收益首先体现在寄存器压力、warp specialization 和可组合 pipeline，而不只是某个单指令 latency。

### 6.2 Tensor Memory

| 操作 | cycle |
|---|---:|
| alloc+dealloc，32 columns | 395–396 |
| TMEM→register load dependency | 35.19 |
| TMEM store issue / store+wait | 8.17 / 40.09 |
| TMEM store→load | 58.47 |
| empty wait::ld / wait::st | 1.27 / 11.08 |
| SMEM→TMEM copy issue / 每条完成 | 64.01 / 159.10 |

第一版纯 load 被 ptxas 跨循环消除；正式实现让上一次 load 结果参与下一次地址，SASS 保留 `LDTM/STTM`。数据表明 TMEM 同样是异步/解耦资源：store 发射快，但安全复用或读取必须等待完成。

`alloc+dealloc` 约 396 cycle，远大于单次 MMA 的 44-cycle 稳态间隔，说明 TMEM 更像 CTA 生命周期内分配的专用 tile store，而不是每次内积临时申请的 scratch。合理 kernel 应在 CTA 启动时分配、跨许多 K-loop MMA 重用，最后统一释放。SMEM→TMEM copy 每条完成约 159 cycle，也应与后续/前一 tile 的 Tensor Core 工作交叠。

## 7. 异步搬运：等待位置决定它是 latency 成本还是 hiding 工具

| Source | 同步基线 | `cp.async` | TMA global→shared |
|---|---:|---:|---:|
| L2 | 278.31 | 282.36–282.37 | 298.23 |
| device miss | 826.55 | 862.99 | 879.28 |

这些都是发射后立即等待数据可用的端到端 cycle。异步路径略慢，原因是多了 commit/barrier/completion 控制；它并没有让 L2 或系统内存介质变快。

CUDA 文档把 LDGSTS 定义为 global→shared 的 element-wise 异步搬运，把 TMA 定义为 bulk/多维数据搬运，并明确其目的在于允许发起线程继续计算、通过 barrier 或 async group 接收完成信号 [R5]。因此只有在 wait 前安排独立计算或积累多个在飞 transfer，异步机制才可能隐藏延迟。Bandwidth 报告中 `cp.async` stage 1→4 和 TMA tile/batch 扫描验证了这一条件。

TMA 是 Hopper 引入、Blackwell 延续的独立地址生成/数据搬运单元；它让单线程描述 bulk transfer，避免整个 warp 消耗 register 和普通 SM issue slots [R9]。因此本表衡量的是“同步消费时的固定开销”，Bandwidth 报告的大 tile/batch 扫描衡量的才是该硬件在持续 pipeline 中的价值。

TMA shared→global 的 issue、`wait.read` 和完全完成分别为 8.29、34.58、46.39 cycle。若只需复用 shared buffer，`wait.read` 足够；若消费者依赖 global 中的新值，则需要更强的完成条件。等待语义应与数据 hazard 对齐，过强同步会缩短本可重叠的窗口。

## 8. 结论与适用边界

1. Thor 依赖访问从 shared 的 23 cycle 到 device path 的约 821 cycle；命中层级比“global/local”地址空间名称更能解释延迟。
2. Store、FP pipeline、tcgen05 与 TMEM 都表现出 issue/completion 解耦；性能模型必须明确采用哪一口径。
3. DSMEM peer 是跨 SM 的 cluster 访问，不能按本地 shared 估算；cluster barrier release 又显著贵于 relaxed barrier。
4. 异步搬运只有在 wait 前存在独立工作或足够在飞深度时才能隐藏延迟。
5. 当前 L2 数据不足以重建 Thor 物理拓扑；单卡环境也无法给出 Thor P2P 结论。
6. Thor 官方平台是 integrated-memory SoC，host-pinned 与 device path 接近是拓扑信号；不能把它解释成离散 PCIe 性能，也不能把 B200 HBM 规格套到 Thor。

本报告是低并发 latency 研究。持续吞吐、coalescing、MLP、TMA batching 和多 stream 竞争见 [Thor Bandwidth 报告](../Bandwidth/REPORT_BLACKWELL_THOR_20260809.md)；跨主题因果关系见 [综合报告](../REPORT.md)。

## 9. 参考资料

- [R1] NVIDIA, [CUDA Programming Guide — Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html).
- [R2] NVIDIA, [CUDA Programming Guide — GPU Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#gpu-memory).
- [R3] NVIDIA, [CUDA Programming Guide — Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-block-clusters).
- [R4] NVIDIA, [PTX ISA — TensorCore 5th Generation Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html).
- [R5] NVIDIA, [CUDA Programming Guide — Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html).
- [R6] NVIDIA, [Jetson AGX Thor Developer Kit — CUDA deviceQuery](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_cuda.html).
- [R7] NVIDIA, [CUDA Programming Guide — Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
- [R8] NVIDIA, [CUTLASS tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html).
- [R9] NVIDIA, [Hopper Architecture In-Depth — TMA](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).

## 10. 原始数据与复现

正式回归位于 [`results/blackwell-thor-20260809-124123/`](results/blackwell-thor-20260809-124123/)。目录包含 `environment.txt`、每项 `.txt/.exit`、标量/tcgen05/TMEM SASS，以及 footprint/直方图 CSV 输出。退出码 0 表示通过，77 表示硬件或架构跳过；NCU 权限失败原始诊断位于 `03_mem_levels-ncu.txt`。

```bash
./run_blackwell_thor.sh 0
```

每次运行都会创建新的时间戳目录，不覆盖正式结果。
