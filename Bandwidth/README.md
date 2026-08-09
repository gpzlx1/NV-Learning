# GPU Bandwidth Architecture Exploration

本目录与 `Latency/` 并列，专门研究“单位时间能搬运或完成多少数据/操作”。Latency 使用依赖链、单线程或单 warp；Bandwidth 则需要足够并发，并明确区分 requested bytes、实际内存事务和读写合计流量。

当前测试设备是 NVIDIA Thor (`sm_110`, 20 SM, 32 MB L2)。Thor 使用 ATS/系统协同地址转换，本文优先使用 `device memory path`，不预设它与离散式 H800/B200 的 HBM 路径相同。

## 值得探索的矩阵

| 编号 | 主题 | 能揭示的架构问题 | 关键变量 |
|---|---|---|---|
| 01 | Global memory | 峰值读/写/copy 带宽，需要多少 SM 饱和 | blocks/SM、读写比例、向量宽度 |
| 02 | Cache working set | L1/L2 的吞吐台阶与容量边界 | footprint、reuse、`.ca/.cg/.cs` |
| 03 | Coalescing | 一个 warp 的地址如何映射为内存事务 | stride、offset、32/64/128-bit |
| 04 | Shared memory | shared pipe 峰值与 bank conflict 代价 | bank stride、广播、读写比例 |
| 05 | DSMEM | cluster 网络的单向/双向和扩展带宽 | cluster size、peer 数、方向 |
| 06 | Async copy | `cp.async`/TMA 相对同步搬运的吞吐与在飞深度 | tile、stage、producer 数 |
| 07 | Atomics | shared/L2 原子单元吞吐及热点退化 | 地址数、竞争度、操作类型 |
| 08 | Tensor Core | tcgen05 的 FLOP/s、shape 和数据类型扩展性 | F16/FP8/FP4、M/N/K、CTA group |
| 09 | Host/P2P | ATS host path、PCIe/NVLink 的持续带宽 | pinned/pageable、方向、双向 |

## 测量规则

1. 使用 CUDA event 测完整 kernel 的 wall time；吞吐测试不使用单线程 `clock64` 依赖链。
2. 每个配置预热，重复多次取最短稳定时间；原始输出保留 bytes、毫秒与 GB/s。
3. read/write 报告单向有效字节；copy 同时报告 payload GB/s 和 read+write fabric GB/s。
4. 访问必须有可观察结果，且用 SASS/反汇编确认 load/store 没被消除。
5. cache 测试与 device-memory 测试分开，不能把重复命中 L2 的数据称作 DRAM 带宽。
6. 当前只有一张 GPU，P2P/NVLink 留作硬件条件满足后执行。

## 当前实现

- `01_global_memory.cu`
  - 512 MB device allocation 的 read、write、copy 吞吐
  - grid 从 1 block 扩展到 8 blocks/SM，观察饱和点
  - 32/64/128-bit 向量访问宽度
  - stride 1–32 的 requested bandwidth，观察 coalescing/transaction amplification
- `02_cache_working_set.cu`：每 SM 私有 L1 工作集、共享 L2 工作集与 cache operator
- `03_shared_memory.cu`：shared read/write 峰值与 1–32-way bank conflict
- `03b_shared_patterns.cu`：broadcast/same-bank distinct 与 32/64/128-bit shared read 交叉验证
- `04_dsmem.cu`：cluster 单发送者、全 CTA ring read/write aggregate bandwidth
- `05_atomics.cu`：global/shared 原子吞吐随热点地址数变化
- `06_copy_engines.cu`：D2D 与 ATS pinned-host copy engine 带宽
- `07_tcgen05_throughput.cu`：M64N8/N16/N32K16 F16→F32 持续 Tensor Core TFLOP/s
- `08_async_copy.cu`：`cp.async` global→shared 带宽随在飞 group 数变化
- `09_tma.cu`：TMA global→shared 带宽随 transfer size 与 batch depth 变化

```bash
make
./01_global_memory --dev 0
```

结果保存到 `results/`，后续测试和报告均只引用本目录的数据。
