# 从访问延迟到系统吞吐：NVIDIA Thor GPU 微架构实证研究

## 摘要

GPU 性能常被简化为“算力”和“带宽”两个峰值，但真实 kernel 还取决于访问依赖、在飞请求数、缓存驻留、异步搬运以及 CTA 调度。本研究在 NVIDIA Thor（Blackwell，`sm_110`）上构造三组互补微基准：以依赖链测完成延迟，以高并发流式访问测持续吞吐，以双 stream 干扰测调度行为；另以 H800（Hopper，`sm_90a`）结果检验哪些现象具有跨平台共性。

实验得到一条一致的因果链。Thor 上 shared memory、L1、DSMEM peer、L2 和 device-memory path 的依赖读延迟依次约为 23、32、178、248 和 821 个 SM 周期；但 device-memory read 在足够并发下仍可达到约 261 GB/s。二者并不矛盾：延迟描述单条依赖链的等待时间，带宽描述大量独立请求重叠后的完成率。1 block/SM、ILP=1 时只有 133 GB/s；将 ILP 提高到 4 或将 blocks/SM 提高到 4，带宽均升至约 256–260 GB/s，说明 instruction-level parallelism 与 occupancy 是提供 memory-level parallelism（MLP）的两种可替代来源。缓存与显式片上复用进一步把聚合吞吐从 device path 的 261 GB/s 提高到 L2 的 1.41 TB/s、L1 的 3.85 TB/s 和 shared memory 的 6.73 TB/s。

异步搬运并未降低介质本身的延迟或提高其物理上限。`cp.async` 和 TMA 的收益来自扩大 bytes-in-flight、减少地址与搬运指令，并让数据传输与计算重叠；若发射后立刻等待，端到端延迟反而略高于同步路径。并发 stream 也不会复制硬件资源：read+read 与 copy+copy 最终仍共享约 234–262 GB/s 的 fabric ceiling。stream priority 实测能优先选择尚未驻留的 CTA；CUDA 公开语义另行规定它不抢占 already-running work，而当前 resident-control 尚未强制耗尽 SM 资源，不能把二者写成同一项实验证明。

这些结果共同表明，GPU 优化不应从单个峰值数字出发，而应先定位受限环节：依赖延迟、事务利用率、在飞深度、片上复用、数据搬运控制成本，或 CTA admission。报告据此给出一套从测量到机理、再到工程决策的分析框架。

## 1. 研究问题与贡献

本研究围绕三个相互衔接的问题展开。

**RQ1：单条依赖操作的等待由什么决定？** 这要求区分 L1/L2/device memory 等物理层级，也要区分 store 发射、store→load 往返和跨执行单元可见性。

**RQ2：高延迟为何仍能形成高吞吐？** 这要求研究访问宽度、coalescing、ILP、occupancy、cache reuse 与异步 pipeline，而不能把带宽下降笼统归因于“内存慢”。

**RQ3：多个工作负载同时提交时，资源如何共享？** 这要求把 memory fabric 竞争与 CTA 调度分开，并检验 stream priority 的能力边界。

相较于逐项罗列 benchmark，本报告的贡献是把三类结果连接为一个层次模型：

```text
单操作完成时间
    ↓ 需要足够独立请求来覆盖等待
MLP / 事务利用率
    ↓ 决定持续吞吐能否接近层级上限
片上复用 / 异步搬运
    ↓ 改变流量去向与在飞深度
并发资源竞争 / CTA admission
    ↓ 决定多 workload 下谁能获得这些上限
应用可见性能
```

## 2. 平台、方法与证据边界

### 2.1 测试平台

Thor 主实验采集于 2026-08-09，设备报告 20 个 SM、32 MiB L2、compute capability 11.0，Driver 595.78，CUDA 13.3。该系统使用 ATS/统一内存形态；因此报告使用 **device-memory path** 和 **host-pinned path** 描述软件可见路径，不仅凭 API 名称把它们解释成离散 GPU 的 HBM 或 PCIe。CUDA 文档同样指出，地址空间的实现与数据交换方式取决于系统硬件；ATS 系统可使用 CPU 页表与硬件一致性 [R1]。

H800 对照实验来自 132-SM、50 MiB L2 的 Hopper SXM 系统。它用于验证方法与观察跨架构差异，不与 Thor 的纳秒值、带宽或物理互连作等价比较。

### 2.2 三类测量为何互补

| 目标 | 核心方法 | 实际回答的问题 |
|---|---|---|
| 延迟 | 单线程/单 warp 依赖链、指针追逐、`clock64` 差分 | 后一条操作依赖前一条时要等多久 |
| 吞吐 | 多 SM、多 warp、独立地址流、CUDA event | 大量操作重叠后每秒完成多少工作 |
| 调度 | 双 stream、受控提交顺序、短前台与长后台 | 工作何时获得 CTA 驻留和执行机会 |

读延迟使用随机单环指针追逐：节点保存下一个节点的地址，使第 (i+1) 次 load 必须等待第 (i) 次返回。吞吐测试则故意构造多个独立 accumulator 或多个 active warp，让硬件拥有足够多的 outstanding request。两者方向相反，正好分别暴露“无法隐藏的等待”和“能够隐藏等待后的完成率”。

### 2.3 数据口径

带宽至少有三种常见定义：

- requested bandwidth：程序显式请求的有效字节数除以时间；
- payload bandwidth：一次 copy 的逻辑数据量除以时间；
- fabric bandwidth：copy 同时发生读和写，按两侧总流量计数。

例如同一个 121 GB/s payload copy，也可描述为约 243 GB/s read+write fabric traffic。报告在表头明确口径，避免人为制造两倍差异。

### 2.4 证据等级

本研究按以下顺序约束结论强度：

1. 原始时间或周期数据；
2. SASS 确认目标指令确实存在且位于循环内；
3. 哨兵、独立 kernel 或重复进程复测；
4. Nsight Compute transaction/pipe counter；
5. CUDA/PTX 公开语义；
6. 在以上证据仍不能唯一确定内部结构时，只保留最小架构推断。

这个区分尤其重要。Nsight Compute 默认会在 replay pass 前清空缓存，并可能序列化 kernel；官方文档明确说明这会让被测 kernel 更像在隔离、冷缓存状态下执行 [R2]。因此，依赖前序 kernel 建立状态的 L2 persistence 实验以普通执行为主证据，replay 下的近全 miss 只记录为工具边界。

### 2.5 Hopper 与 Thor 的公开硬件基线

为了避免把测量曲线直接“画成”并不存在的芯片框图，本报告先固定公开可验证的架构事实，再解释实验。

| 层面 | H800 / Hopper（本机与 CC 9.0 公开模型） | Thor / Blackwell（本机与 CC 11.0 公开模型） |
|---|---|---|
| SM 组织 | 本机 132 SM；Hopper 每 SM 最多 64 resident warps | 官方 Thor 示例为 20 SM、每 SM 128 CUDA cores；CC 11.0 最多 48 resident warps、1536 threads/SM |
| SM 片上容量 | 256 KiB unified data cache；shared carveout 最高 228 KiB | 256 KiB unified data cache；shared carveout 最高 228 KiB |
| L2 | 本机 50 MiB | 本机与官方 Thor 示例均为 32 MiB |
| 跨 SM 协作 | Hopper 引入 thread-block cluster、DSMEM 与 TMA | CC 11.0 保留 cluster、DSMEM、TMA，并支持 `tcgen05`/TMEM |
| Tensor 数据路径 | `wgmma` 以 warp-group 组织异步 MMA，累加器占用寄存器 | `tcgen05.mma` 使用 TMEM 保存 accumulator，可由单线程发起 CTA collective MMA |
| 系统内存形态 | 离散 H800 SXM，HBM 与 NVLink/NVSwitch | integrated GPU sharing host memory；ATS/统一内存系统 |

Hopper 的 256 KiB unified resource、228 KiB shared carveout、50 MiB L2、TMA 和 DSMEM 语义来自 NVIDIA Hopper Tuning Guide [R12]；Thor 的具体 SM/L2/integrated-memory 信息来自官方 Jetson AGX Thor `deviceQuery` 示例 [R13]，CC 11.0 的 256 KiB unified cache、228 KiB shared、32 banks 与 occupancy 上限来自 CUDA compute-capability 表 [R14]。Blackwell B200 的 HBM、dual-die 或 126 MiB L2 **不适用于 Thor**，所以本报告不把通用 Blackwell 数据中心产品框图套到这颗 SoC。

这些公开事实与本仓库可观测路径可以组成下述逻辑数据流；虚线表示异步发起后不要求普通 warp 指令在原地等待：

```text
                         ┌──────────── SM ────────────┐
register / warp issue ──►│ LSU ─► unified L1/texture │
                         │       shared-memory ports  │
                         │ Tensor Core ◄──► TMEM      │  Blackwell only
                         └──────┬───────────┬──────────┘
                                │           │ cluster fabric
                         global │           └────► peer-SM shared (DSMEM)
                                ▼
                         distributed/shared L2
                                │
                         memory controller / system fabric

global memory ─ ─ TMA/async proxy ─ ─► shared memory ─► Tensor Core
                   （硬件地址生成、transaction barrier 完成通知）
```

这张图只表达公开的软件可见单元与数据依赖，不声称知道 Thor 的 L2 slice 数、NoC 路由或 shared read 端口数。

## 3. RQ1：从片上存储到系统内存，延迟如何形成

### 3.1 延迟阶梯不是地址空间列表，而是实际数据路径

Thor 的依赖读结果如下。周期是主口径；由于测试轮次的 SM 频率在约 0.93–1.56 GHz 之间变化，跨轮次直接比较纳秒会混入动态频率影响。

| 路径 | 依赖延迟（cycle） | 相对 shared | 解释 |
|---|---:|---:|---|
| shared memory | 23 | 1.0× | SM 内显式管理的片上存储 |
| L1 hit | 32 | 1.4× | 每 SM 私有 unified data cache |
| DSMEM peer | 178 | 7.7× | 经 cluster 访问另一 CTA 的 shared memory |
| L2 hit | 248 | 10.8× | 全 GPU 共享、跨 SM/片上网络路径 |
| device-memory path | 821 | 35.7× | 超过 32 MiB L2 后的容量 miss 路径 |
| host-pinned path | 890 | 38.7× | 系统/统一内存形态下的映射访问 |

CUDA 的编程模型把 shared、global、local、constant 定义为地址空间；硬件则以 register、shared/L1、L2 和系统内存构成实际路径。官方文档也说明 L1 属于每个 SM，而更大的 L2 被所有 SM 共享 [R3]。因此“global memory latency”没有唯一值：global load 可能命中 L1、命中 L2，也可能离开片上缓存。

Thor footprint 在 32–256 KiB 保持约 32 cycle，与 CC 11.0 公布的 256 KiB unified data cache 上限吻合；H800 的台阶同样落在 256 KiB。这里的“一致”比单个 32-cycle 数字更有架构意义：它把容量拐点、公开资源大小和 load latency 三类证据连接起来。由于 L1/texture 与 shared 共享可配置的片上容量，benchmark 几乎不分配 shared memory 时，更多 unified resource 可用于 L1；真实 kernel 若把 carveout 大量分给 shared，L1 可用容量与 miss 行为可能改变 [R12][R14]。

DSMEM 位于 L1 与 L2 之间不是偶然。thread-block cluster 保证相关 block 在同一 GPC 内协同驻留，并允许它们访问彼此的 shared memory [R4]。peer 访问无需绕到普通 global-memory 容量层级，但仍需跨 SM/cluster 互连和远端 shared 端口，所以显著慢于本地 shared/L1，又快于本机 L2 依赖命中。

### 3.2 Store 的“快”只代表发射快

Thor 上 local/shared store 约 4 cycle、global store 约 8 cycle 即可继续发射，但这不是数据完成时间：

| 口径 | shared | L1 | L2 | device/host path |
|---|---:|---:|---:|---:|
| store issue interval | 4.03 | 7.95 | 7.95 | 8.03 |
| store→load round trip | 28.00 | 41.05 | 266.59 | 911.85 / 1012.25 |

这一差异说明 store 是 decoupled/posted 的：执行前端只需把请求交给后续队列，目标层级的传输与一致性工作可在后台继续。若后续计算不依赖该值，短 issue interval 有利于吞吐；若马上读回或跨 SM 同步，真正的关键路径仍是数百周期。

因此，分析写路径必须先问“测的是哪一种完成”：

1. 指令能否继续发射；
2. 本线程何时能读回；
3. 其他执行单元何时按指定 memory scope 观察到。

Thor 的 `release.gpu→acquire.gpu` 和 `store.cg+membar.gl→load.cg` 分别约为 646 和 617 cycle，明显高于普通 L2 store→load 的 267 cycle。同步不是一条固定成本指令，而是要求先前访存达到相应作用域的有序与可见状态。

### 3.3 地址、SM 与物理放置会改变 L2 延迟

H800 的 8 个地址×132 个 SM 测试出现稳定的近/远两簇：同一物理地址对不同 SM 可表现为约 259 或 299 cycle，改变物理分配后分类也会改变。这支持“延迟由 `(SM, 物理地址)` 共同决定”的模型，而不支持给每个虚拟 buffer 偏移硬编码 near/far 标签。

Thor 的同类测试只出现约 9–12.5 cycle 的两组中心差，且只有 4/20 个 SM 在八个地址间发生分类翻转；它不支持直接照搬 Hopper 的稳定双分区/TPC 粒度结论。合理写法是：**H800 已观察到位置相关 L2 延迟；Thor 也存在地址/SM 离散，但当前样本不足以识别其物理拓扑。** 这比用一张示意图反推 die 结构更符合证据边界。

### 3.4 H800 的跨设备结果说明“缓存层级”还取决于访问来源

H800 经 NVSwitch 读取另一张 GPU 时，24 KiB 工作集可稳定达到 32 cycle，与本地 L1 命中相同；32 MiB 与 2 GiB 工作集却分别约为 1680 和 1705 cycle。结合指针属性与 NVLink Rx counter，数据支持“远端行可被本地 L1 复用，但在该测试中未表现出本地 L2 容量复用”。

远端普通读约 1680 cycle，远端原子约 1833 cycle，而 system-scope 标志传递约 4961 cycle。工程含义不是“NVLink 很慢”这一表面结论，而是：跨卡细粒度依赖同时支付远端往返与系统作用域有序性，难以靠峰值链路带宽解释。应优先批量传输、局部聚合并降低跨卡同步频率。

Thor 则是另一种系统边界。官方 `deviceQuery` 明确报告 integrated GPU sharing host memory，故 device allocation 与 host-pinned path 都落在 SoC/统一内存体系中，而不是“一侧 HBM、一侧经 PCIe 的 CPU DRAM”这种离散 GPU 二分。Thor 上两者依赖读只差约 8%，而 H800 host-pinned 约为本地 HBM 的 3.7×；这项跨平台差异首先反映内存拓扑，而不是 Blackwell cache 本身突然把 PCIe 变快 [R13]。

## 4. RQ2：高延迟如何被转换为持续吞吐

### 4.1 关键不是降低一次等待，而是增加在飞工作

若单个请求延迟为 (L)，希望维持完成率 (R)，系统平均需要约 (N=R\times L) 个在飞工作。这是 Little's Law 在访存 pipeline 上的直观形式。GPU 通过不同 warp、同一线程的独立 load、异步 copy group 和多个 CTA 提供这些在飞工作。

Thor 的控制实验直接展示了这一点：

| Blocks/SM | ILP 1 | ILP 2 | ILP 4 | ILP 8 |
|---:|---:|---:|---:|---:|
| 1 | 133 | 209 | **256** | 213 |
| 2 | 226 | 259 | **263** | 259 |
| 4 | 260 | 262 | 263 | 262 |
| 8 | 253 | 263 | 263 | 264 |

单位为 requested GB/s。NCU 对 `1 block×ILP1`、`1×ILP4`、`4×ILP1` 记录了相同 request 与 sector 数，因此性能差异不是“做了更少工作”。`1×ILP4` 在 active warps 仍约 16% 时已接近峰值，证明高 occupancy 不是目的本身；它只是与 ILP 一样，为调度器提供更多相互独立的请求。ILP=8 在低 occupancy 下反而回落，说明额外寄存器、指令与调度开销在覆盖延迟后不再免费。

### 4.2 访问宽度改变每条指令携带的有效工作

相同数据量与并发下，32/64/128-bit load 分别达到 153、221 和 257 GB/s，而延迟报告中 64/128-bit L2 load 都约 248 cycle。由此可排除“128-bit load 单次返回更快”的解释；更符合数据的机制是，宽 load 在相近的依赖完成时间和指令开销下携带更多字节，减少 load、地址更新和循环控制指令的占比。

因此向量化的收益属于吞吐域，而不是延迟域。它仍受对齐、cache-line 边界和寄存器压力约束，不能从一个对齐微基准外推到任意结构体访问。

### 4.3 Coalescing 决定事务中的有效字节比例

当 warp 的 32 个 lane 各读取 4 B 连续数据时，128 B useful data 由 4 个 32 B sector 服务；lane stride 从 1 增至 8 时，sector/request 从 4、8、16 增至 32，有效利用率从 100% 降至 12.5%。CUDA Best Practices Guide 对 compute capability 6.0+ 的通用规则也是相邻 4 B word 可由四个 32 B transaction 服务 [R5]。

这里的深层限制不是“地址不连续所以慢”，而是 transaction amplification：硬件移动的 sector 数增长，而程序真正消费的 requested bytes 不变。stride≥8 后已达到每 lane 一个 sector 的上限，继续增大 stride 不再增加 sector/request。原始 benchmark 在 stride=32 时 wall time 下降，是为了不越过 allocation 而减少了 load 总数；NCU 计数器避免把这一实现细节误判为 coalescing 恢复。

### 4.4 数据复用把请求移到更高吞吐的层级

在足够 ILP 下，Thor 各层聚合 requested bandwidth 为：

| 数据来源 | 工作集组织 | GB/s |
|---|---|---:|
| device-memory path | 512 MiB 单次流式读 | 261 |
| L2 | 16 MiB 共享工作集反复读 | 1411 |
| L1 | 每 SM 64 KiB 私有工作集 | 3849 |
| shared memory | conflict-free | 6727 |

从 device path 到 L2/L1/shared 的提升不是同一物理总线“变快”，而是越来越多访问在靠近 SM 的复制数据上完成，避开下层共享资源。最初只有一个 accumulator 依赖链时，L1/L2 只有 1435/419 GB/s；改为 8 路独立 accumulator 后才升至 3861/1410 GB/s。这也说明 cache 命中并不自动等于 cache pipe 饱和：命中解决“去哪里取”，MLP 解决“同时取多少”。

“L1 与 shared 使用统一容量”也不等于它们使用完全相同的数据端口。Thor 的纯 LDS latency/带宽为 23 cycle/6.73 TB/s，L1 load 为 32 cycle/3.85 TB/s，说明 shared addressing/service path 与 L1TEX/coalescing path 在软件可见行为上仍不同。公开文档只确认容量与 L1/texture coalescing buffer 的统一关系 [R14]；现有数据可以证明服务特性不同，却不足以给出内部端口数量。

### 4.5 L2 persistence 管的是驱逐优先级，不是 pipe 峰值

Thor 报告 32 MiB L2、24 MiB persisting reserve。先预热 hot set、再用 128 MiB 流污染 L2 后，persisting access-policy 对 4/16/32 MiB hot set 的 reload 加速范围分别为 1.45–2.56×、1.63–3.99×、1.25–1.35×。16 MiB 完整落入 reserve 时收益最高；32 MiB 超出 reserve 后收益回落。

CUDA 官方语义是为 persisting access 提供 L2 set-aside 的优先使用权，`hitRatio` 控制 window 中约有多少访问取得 persisting 属性；多个并发 window 仍共享这部分容量 [R6]。因此该机制减少的是污染后的 miss 概率，不会把 1.41 TB/s 的 L2 service ceiling 再凭空提高。它适合跨 kernel 重用的权重、查表或常驻状态，而不是单次流式数据。

### 4.6 Shared bank conflict 在 Thor 上表现出读写非对称

传统 CUDA 模型中，同一 bank 的不同地址访问可能被串行化 [R7]。Thor 的 write 符合这一规律：conflict degree 1→32 时，requested bandwidth 从约 3.53 TB/s 降到 126 GB/s，近似反比下降；MIO throttle 从 43% 升至 84%。

但本测试的 read 从 1-way 到 32-way 始终约 6.76 TB/s。SASS 保留循环内 LDS，NCU 又确实报告每指令 0/1/3/…/31 个 conflict，而 short-scoreboard 与 MIO throttle 接近零。最小、可验证的解释是：**counter 仍按 bank 映射识别冲突，但 Thor 在这些静态 32-bit 读模式下的服务/合并能力使冲突没有转化为可见 stall。** 现有证据不能唯一确定 bank 数、端口数或 crossbar 拓扑，也不能外推到动态地址和所有 Blackwell 芯片。

### 4.7 DSMEM 的上限由网络注入与驻留共同决定

cluster=2 的全 CTA ring read 约 6.5–6.7 TB/s，cluster=4 约 3.36 TB/s，cluster=8 约 2.69 TB/s。cluster 增大不只改变逻辑 rank 距离，也减少可同时驻留的 cluster 数与注入 CTA 数。更细的 rank-delta 扫描没有稳定的“距离越远越慢”，而在不同进程中出现离散档位。

因此报告不把 rank 号解释为物理 mesh 坐标。CUDA 只保证 cluster block 同时位于一个 GPC，并允许访问 distributed shared memory [R4]；它没有承诺逻辑 rank 与物理距离的单调对应。Hopper/Blackwell Tuning Guide 还明确说明 DSMEM 可与 L2 同时使用 [R12][R15]，这支持它是 cluster 内直接 SM-to-SM 数据路径而非“把 peer shared 请求翻译成普通 L2 load”。本机 DSMEM peer latency 低于 L2，也与此公开结构一致。现有数据更支持 placement state 与活跃注入点数量共同决定吞吐。

## 5. 异步搬运：改变重叠方式，而不是改变介质

### 5.1 立即等待时，异步路径没有延迟优势

Thor L2 同步 global→shared 基线约 278 cycle，`cp.async` 约 282 cycle，TMA 约 298 cycle；device miss 路径分别约 827、863 和 879 cycle。异步形式若发射后立刻 wait，会额外承担 commit/barrier/completion bookkeeping，不能降低一次依赖访问的端到端时间。

CUDA 对 LDGSTS/`cp.async` 的定义也是允许发起线程在硬件搬运时继续计算；TMA 则把一维或多维 bulk copy 的地址工作交给专门路径，并通过 barrier 或 async-group 报告完成 [R8]。Hopper 官方说明进一步指出，TMA 由单线程发起，硬件负责 stride、offset、边界与数据移动，避免让整个 warp 用普通 SM 指令和寄存器循环搬运 [R16]。这解释了小 tile 为什么被固定控制成本支配，也解释了大 tile/多 stage 为什么能以更少 issue slot 覆盖更多字节。它们的价值条件是等待窗口内存在独立计算或其他 transfer。

### 5.2 Pipeline 深度把延迟转化为 bytes-in-flight

`cp.async` 在飞 group 从 1 增至 4 时，吞吐从 260 增至 267 GB/s；8 group 不再提升。TMA 的 128 B×batch1 只有约 19 GB/s，而 1024 B×batch8 可达约 201–268 GB/s。前者每 128 B 都支付一次描述、barrier accounting 与完成通知，固定控制成本占主导；更大 tile 摊薄固定成本，更深 batch 增加在飞字节。

这与 CUDA 的建议一致：bulk copy 通常应以尽可能少、尽可能大的传输发起，并可用多 stage 预取把未来数据搬运与当前计算重叠 [R8]。最佳 stage 数仍是平台与 kernel 资源的共同结果；超过覆盖延迟所需的深度后，shared buffer、寄存器和 barrier 状态都会成为成本。

### 5.3 Multicast 去重 source request，但不保证提高 delivered bandwidth

fanout=2 时，TMA unicast 发射 524,288 条 load，multicast 仅 262,144 条，正好减半；二者 delivered bytes counter 都是 512 MiB。由此可知 multicast 确实减少了上游发射/取数次数，但 counter 的 byte 位置在 fan-out 之后。

两者 delivered bandwidth 仍约为 29–31 GB/s，因为测试每次 transfer 后立即等待，瓶颈落在交付与完成控制，而非 source fetch 数。multicast 的正确收益场景是多个 cluster CTA 复用同一 tile，同时上游读流量或地址指令构成瓶颈；它不是把下游 shared/cluster 交付管线成倍扩宽。

## 6. 计算与内存的连接：微基准 Roofline

Thor 的 F16→F32 `tcgen05.mma` 在 M64N8/16/32K16 三个 shape 上耗时近似相同，但每条指令的数学工作量随 N 翻倍，持续性能相应为 11.62、23.25、46.50 TFLOP/s。Latency 测试中三种 shape 的稳态间隔也都约 44.23 cycle，说明当前范围内指令承载量增长而可见 issue interval 基本不变。

TMEM 进一步体现“发射”和“完成”的区别：store 约 8.17 cycle 可连续发射，每条 store 后等待则为 40.09 cycle，store→load 往返为 58.47 cycle。`tcgen05.mma` 每条 commit+barrier wait 的软件可见完成时间约 156 cycle，不能把 44-cycle 稳态间隔误写成单条数学结果完成延迟。PTX ISA 将 Tensor Memory 的分配、load/store/copy、MMA 与 commit 分成不同指令族 [R9]，与这一测量分解一致。

Blackwell 在 Hopper 的异步 `wgmma` 基础上把 accumulator 数据路径进一步从普通寄存器文件中拆出：CUTLASS 的 `tcgen05` 指南把 TMEM 描述为专用于 accumulator（并可选存放 A operand）的片上存储，MMA 由单线程发起，并支持相邻 CTA 协同 [R17]。其直接架构后果有三点：

1. 大矩阵 accumulator 不再持续占用大量通用寄存器，给地址、pipeline state 和 occupancy 留出空间；
2. issuing thread 的普通寄存器 scoreboard 不等同于 Tensor Core/TMEM 内部完成状态，必须用 `commit`/mbarrier 建立软件可见依赖；
3. TMEM alloc/dealloc 是 CTA 级资源管理，约 396-cycle 固定成本应由很多次 MMA 摊销，不能按每个小 tile 反复分配。

本机 overwrite 与同一 TMEM accumulator 累加都表现为 44-cycle 稳态间隔，不代表硬件“没有 accumulator dependency”；更准确的解释是该依赖由异步 Tensor Core/TMEM pipeline 管理，尚未反压当前形状的可见发射间隔。156-cycle completion 才暴露结果真正可由软件继续消费的边界。

以 46.5 TFLOP/s compute ceiling 和约 0.258 TB/s device-read ceiling 构造教学性 Roofline，交点约为：

\[
I^* = \frac{46.5\ \text{TFLOP/s}}{0.258\ \text{TB/s}} \approx 180\ \text{FLOP/byte}
\]

当 arithmetic intensity 低于该值时，理想上限随可用带宽线性增长；高于该值后才逐渐受 tensor compute ceiling 约束。Roofline 的 arithmetic intensity 定义与这一判断方式见 NVIDIA/NERSC 的方法说明 [R10]。这里的两个 ceiling 来自分离微基准，不能当作真实 GEMM 保证值；它的用途是先判断优化 FLOP、byte 还是数据复用。

## 7. RQ3：并发 workload 如何共享带宽与调度资源

### 7.1 两个 stream 不等于两套 memory engine

`serial/wall=2` 才表示两项工作完全重叠。Thor 的并发结果为：

| 组合 | serial/wall | 总 fabric GB/s | 主要限制 |
|---|---:|---:|---|
| read+read | 1.19× | 262 | 共享 read ceiling |
| read+write | 1.39× | 238 | 部分独立服务能力与共享 fabric 并存 |
| write+write | 1.78× | 249 | 单 workload 未饱和，多 stream 补足并发 |
| copy+read | 1.06× | 245 | copy 已同时占用读写两侧 |
| copy+copy | 0.99× | 234 | 近似共享同一 read+write ceiling |

“能并发”只表示调度上允许重叠，不表示物理资源被复制。read+write 比 read+read 更易重叠，说明路径上存在部分独立队列或服务能力；但总 fabric 仍停在约 238–262 GB/s，说明它们最终共享系统瓶颈。现有 counter 无法唯一定位共享点，报告不进一步命名未公开的 crossbar 或 memory-controller 结构。

### 7.2 Stream priority 的实测边界是 pending CTA 选择

短前台 kernel 单独需 0.176 ms。后台拥有很多尚未调度的 CTA wave 时，同优先级前台延迟为 15.53 ms；提高前台优先级后降至 1.10 ms。若后台只启动每 SM 一个长 CTA，同优先级与高优先级前台分别为 0.333 和 0.330 ms，差异不可分辨。

第一组数据直接支持：高优先级 pending work 可越过低优先级 pending CTA。第二组却不能独立证明“无法抢占”：后台 CTA 为 256 threads/8 warps，而 CC 11.0 每 SM 最多 48 resident warps，且 kernel 没有用 shared memory 强制 occupancy=1，前台可能直接共驻留。CUDA Programming Guide 另行明确规定 priority 不抢占 already-running work，也不保证严格顺序 [R11]；本机数据与该语义相容，但若要做硬件验证，需用 occupancy API 和约 228 KiB dynamic shared memory 构造真正 resource-saturating resident CTA。

工程上，高优先级 stream 不是实时保证。若服务有严格尾延迟目标，应缩短 CTA 执行区间、减少单 CTA 资源占用、保留共驻留余量，或在更高层对长任务切片；仅调整 priority 无法消除所有 head-of-line blocking。

## 8. 综合讨论：如何把结果用于真实 kernel

从上述证据可以形成一个按顺序执行的诊断框架。

1. **先确认关键路径是 latency 还是 throughput。** 指针追逐、锁、链式原子和细粒度同步关注完成延迟；规则数组计算通常更关注持续吞吐。
2. **若是 latency，确认实际命中层级与 memory scope。** 不要用“global memory”代替 L1/L2/device/remote 路径，也不要用 store issue interval 代替可见时间。
3. **若是 throughput，先检查事务利用率。** 宽访问与 coalescing 解决每条指令、每个 sector 中有多少有效字节。
4. **再检查 MLP 是否足够。** ILP、active warps、多个 CTA 与 async stage 都可增加在飞工作；occupancy 是手段，不是目标。
5. **通过数据复用改变流量层级。** shared/L1/L2 reuse 的收益通常大于在 device-memory ceiling 上微调几个百分点；跨 kernel reuse 可考虑 persisting policy。
6. **为异步机制安排真正可重叠的工作。** `cp.async`/TMA 后立即等待只增加控制成本；tile 大小、stage 数和同步粒度必须一起设计。
7. **多 workload 下同时看 fabric 与 admission。** stream 可以提高利用率，也会竞争共享上限；priority 只能影响尚未执行的工作选择。

## 9. 有效性威胁与未决问题

- Thor 的动态频率变化明显；片上 cycle 适合跨轮次比较，跨时钟域路径仍应同时报告频率与时间。
- ATS/统一内存平台的 `cudaMalloc`、host-pinned 与物理内存控制器关系不能从 API 标签直接推断；本报告只描述软件可见路径。
- Jetson Thor 的 EMC/MC 与 DRAM 会根据 bandwidth QoS 和利用率动态调频；官方文档说明 BPMP bandwidth manager 可在多个 EMC 档位间选择 [R18]。这为 201/268 GB/s、97/131 GB/s 等离散平台状态提供了架构上合理的候选原因，但当前结果未同步记录 EMC 档位，因此仍标为解释而非已证实归因。
- 部分 NCU 采集受权限或 replay/cache-control 限制。没有 counter 的结论使用较低证据等级，不以推断补齐。
- Shared read conflict 的反常行为只覆盖当前 `sm_110`、静态地址和已验证 SASS；需要动态依赖、不同宽度和地址置换继续检验。
- DSMEM 需要更多跨进程 placement、双向混合流与系统 trace，才能区分网络拓扑和驻留效应。
- Thor 当前只有单卡，不能复现实机 P2P/NVLink；H800 结果不应代替 Thor 多 GPU 结论。
- tcgen05 仅覆盖 F16→F32 的若干 shape；FP8/FP4、block scaling、稀疏与双 CTA 在完成数值正确性、SASS 和 pipe counter 三重验证前不报告性能。

## 10. 结论

本研究的核心结论不是某个孤立峰值，而是三个尺度之间的关系：

1. **延迟由路径与依赖决定。** Thor 从 shared 的 23 cycle 到 device-memory path 的约 821 cycle 跨越 36 倍；store issue、read-back 与可见性又是不同指标。
2. **吞吐由有效事务与在飞深度决定。** ILP 和 occupancy 均可提供 MLP；cache/shared reuse 则把请求转移到更高吞吐的层级。
3. **异步与并发只是在共享硬件上重排工作。** 它们可以隐藏等待、摊薄控制成本和提高利用率，但不能突破介质与 fabric 的共同上限。
4. **调度决定优化是否能及时生效。** priority 实测可改善 pending CTA 的选择；API 不承诺抢占已运行工作，本机抢占粒度仍需资源饱和 CTA 与 timeline 验证。

因此，可靠的 GPU 性能分析必须把时间、流量和调度放进同一个因果模型，并让每项架构解释与其证据强度匹配。

## 11. 参考资料

- [R1] NVIDIA, [CUDA Programming Guide — Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html).
- [R2] NVIDIA, [Nsight Compute Profiling Guide — Replay and Cache Control](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
- [R3] NVIDIA, [CUDA Programming Guide — GPU Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#gpu-memory).
- [R4] NVIDIA, [CUDA Programming Guide — Thread Block Clusters and Distributed Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-block-clusters).
- [R5] NVIDIA, [CUDA C++ Best Practices Guide — Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory).
- [R6] NVIDIA, [CUDA C++ Programming Guide — L2 Access Management](https://docs.nvidia.com/cuda/archive/11.5.2/cuda-c-programming-guide/index.html#device-memory-l2-access-management).
- [R7] NVIDIA, [CUDA Programming Guide — Shared Memory Bank Conflicts](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).
- [R8] NVIDIA, [CUDA Programming Guide — Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html).
- [R9] NVIDIA, [Parallel Thread Execution ISA — TensorCore 5th Generation Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html).
- [R10] NVIDIA, [Accelerating HPC Applications with Nsight Compute Roofline Analysis](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/).
- [R11] NVIDIA, [CUDA Programming Guide — Stream Priorities](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#stream-priorities).
- [R12] NVIDIA, [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/).
- [R13] NVIDIA, [Jetson AGX Thor Developer Kit — CUDA deviceQuery](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_cuda.html).
- [R14] NVIDIA, [CUDA Programming Guide — Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
- [R15] NVIDIA, [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/).
- [R16] NVIDIA, [Hopper Architecture In-Depth — TMA and Asynchronous Execution](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).
- [R17] NVIDIA, [CUTLASS tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html).
- [R18] NVIDIA, [Jetson Thor Platform Power and Performance — EMC Dynamic Frequency Scaling](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html#emc-dynamic-frequency-scaling).

## 12. 数据与复现索引

- Thor latency：[`Latency/results/blackwell-thor-20260809-124123/`](Latency/results/blackwell-thor-20260809-124123/)
- Thor bandwidth：[`Bandwidth/results/`](Bandwidth/results/)
- Thor scheduling：[`Scheduling/results/thor-stream-priority-20260809-155136.txt`](Scheduling/results/thor-stream-priority-20260809-155136.txt)
- H800 latency/P2P：[`Latency/results/`](Latency/results/)

完整表格、失败边界、SASS 文件和单项复现命令见各专项报告与目录 README。
