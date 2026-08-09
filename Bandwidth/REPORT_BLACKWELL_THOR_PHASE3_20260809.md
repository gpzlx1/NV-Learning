# NVIDIA Thor Bandwidth 体系结构探索：Phase 3

Phase 3 研究多工作负载共享硬件时的干扰，以及 bandwidth 所需的 memory-level parallelism 来自哪里。它仍属于 `Bandwidth/`；由结果引出的 stream priority/CTA 调度问题另放 `Scheduling/`。

## 1. 两条 stream 并不意味着两倍硬件

![Concurrent overlap](figures/thor_concurrent_overlap.png)

| 组合 | Serial/wall | 总 fabric GB/s | 解释 |
|---|---:|---:|---|
| read+read | 1.19× | 262 | 两个读共享相同 read ceiling |
| read+write | 1.39× | 238 | 读写有部分独立服务能力，但仍共享 fabric |
| write+write | 1.78× | 249 | 单个 write 未充分占满路径，多 stream 更易饱和 |
| copy+read | 1.06× | 245 | copy 已占 read+write 两侧，与额外 read 强竞争 |
| copy+copy | 0.99× | 234 | 几乎等价串行，共享同一个 read+write ceiling |

`serial/wall=2` 才表示两个任务完全重叠。copy+copy 小于 1 的轻微偏差来自并发调度开销。两个 stream 的完成时间持续不对称，说明 CTA launch order 和资源占用会影响公平性；这不是两套独立 memory engines。

## 2. ILP 与 occupancy 是两种 MLP 来源

![MLP and occupancy](figures/thor_mlp_occupancy.png)

| Blocks/SM | ILP 1 | ILP 2 | ILP 4 | ILP 8 |
|---:|---:|---:|---:|---:|
| 1 | 133 | 209 | **256** | 213 |
| 2 | 226 | 259 | **263** | 259 |
| 4 | 260 | 262 | 263 | 262 |
| 8 | 253 | 263 | 263 | 264 |

单位为 GB/s。只有 1 block/SM 时，ILP 1 无法覆盖 device-memory latency；ILP 4 已达到峰值。增加到 4 blocks/SM 后，即便 ILP 1 也能靠更多 warps 提供相近的 outstanding requests。ILP 8 在低 occupancy 下回落，说明更多独立寄存器/指令并非免费，超过覆盖 latency 所需深度后会增加调度和寄存器压力。

NCU 对 1 block/SM×ILP1、1×ILP4、4×ILP1 都记录完全相同的 1,048,576 requests 与 16,777,216 sectors，排除了 transaction 数变化。1×ILP4 occupancy 仍约 16%，但 long-scoreboard 从 95.24% 降到 92.28%；4×ILP1 occupancy 约 65%。因此真正条件是足够的 memory-level parallelism，不是某个固定 occupancy 百分比。

## 原始证据

- `results/thor-concurrent-traffic-20260809-154828.txt`
- `results/thor-mlp-occupancy-20260809-154828.txt`
- `results/ncu-mlp-20260809-160000/`
- `results/14_concurrent_traffic.sass.txt`
- `results/15_mlp_occupancy.sass.txt`
