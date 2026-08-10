# Bandwidth / Throughput 微基准套件

本目录用高并发 kernel 与 CUDA event 测量持续吞吐，回答“足够多独立工作重叠后，每秒能完成多少数据或操作”。它与依赖链 latency 测试互补，不用带宽倒数推断单次完成时间。

- [Thor Bandwidth 综合报告](REPORT_BLACKWELL_THOR_20260809.md)
- [仓库综合研究报告](../REPORT.md)
- [Thor Latency 报告](../Latency/REPORT_BLACKWELL_THOR_20260809.md)

原 Phase 2/3 已合并到综合报告；旧文件只保留迁移说明，避免外部链接失效。

## 测试矩阵

| 程序 | 主题 |
|---|---|
| `01_global_memory` | blocks/SM 扩展、访问宽度、stride/coalescing |
| `02_cache_working_set` | device/L2/L1 层次吞吐 |
| `03_shared_memory` / `03b_shared_patterns` | shared read/write bank pattern |
| `04_dsmem` / `12_dsmem_topology` | DSMEM ring、rank delta、placement state |
| `05_atomics` | global/shared atomic contention |
| `06_copy_engines` | D2D 与 host-pinned copy path |
| `07_tcgen05_throughput` | F16→F32 Tensor Core 持续吞吐 |
| `08_async_copy` | `cp.async` 在飞 group 深度 |
| `09_tma` / `13_tma_multicast` | TMA tile、batch、cluster multicast |
| `10_hardware_probe` | coalescing/shared counter 交叉验证 |
| `11_l2_residency` | L2 persisting access-policy window |
| `14_concurrent_traffic` | 双 stream 共享 fabric |
| `15_mlp_occupancy` | ILP、occupancy 与 MLP |

## 复现

```bash
make
./run.sh
./run_multicast.sh
./profile_ncu.sh
./profile_l2_ncu.sh
./profile_ncu_pipelines.sh
```

每个配置预热后重复运行，主报告采用明确标注的最短值或跨进程范围。执行时请避免其他 GPU/SoC workload 干扰；Thor 的系统内存状态会使部分绝对带宽形成离散档位。

## 口径

- read/write：程序请求的单向 requested bytes；
- copy payload：逻辑复制的数据量；
- copy fabric：读与写两侧相加；
- atomic：logical operations/s，不将一次 read-modify-write 简化成 4 B；
- Tensor Core：一次 FMA 计 2 FLOP，并明确矩阵 shape。

图表脚本在 `figures/`，原始输出、SASS 与 NCU CSV 在 [`results/`](results/)。
