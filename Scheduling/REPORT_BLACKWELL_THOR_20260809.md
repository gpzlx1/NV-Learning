# NVIDIA Thor Stream Priority 与 CTA 调度探索

日期：2026-08-09

平台：NVIDIA Thor，`sm_110`

## 为什么独立成 `Scheduling`

这里测的不是一次访问要多少周期，也不是单位时间搬多少字节，而是并发工作**何时获得 SM 驻留资格**。因此它既不属于 `Latency/`，也不应混入 `Bandwidth/`。

## 核心结果

前台短 kernel 单独运行需 0.1763 ms。先启动后台长 kernel，再延迟 500 μs 提交前台，结果如下：

| 后台形态 | 前后台优先级 | 前台时间 | 相对单独运行 |
|---|---:|---:|---:|
| 无后台 | — | 0.1763 ms | 1.000× |
| 多 CTA waves | 相同 | 15.5314 ms | 88.118× |
| 多 CTA waves | 前台高 | 1.0979 ms | 6.229× |
| 每 SM 1 个长 CTA | 相同 | 0.3334 ms | 1.892× |
| 每 SM 1 个长 CTA | 前台高 | 0.3303 ms | 1.874× |

![stream priority 与 CTA 调度](figures/thor_stream_priority.png)

## 硬件解释

当后台拥有很多尚未调度的 CTA wave 时，同优先级前台只能排在大量后台 CTA 后面；高优先级 stream 可以越过这些**尚未驻留**的低优先级 CTA，因此等待从 15.53 ms 降至 1.10 ms。

但当后台恰好用一个长 CTA 占住每个 SM 时，提高前台优先级没有可测收益：1.892× 与 1.874× 基本相同。这说明本机的 CUDA stream priority 主要作用于 **CTA admission / pending work selection**，并不把已经驻留的长 CTA 从 SM 上指令级抢占下来。前台仍要等某个驻留 CTA 释放执行资源。

这也解释了为什么“高优先级”不是实时保证：它改善排队选择，却无法消除已驻留工作造成的阻塞。若需要低尾延迟，kernel 应保留可调度余量，或缩短 CTA 的不可让出执行区间。

## 测试与审计

- 前后台分别使用 CUDA non-blocking stream；查询到的优先级范围为 low=0、high=-5。
- 在提交前台前由 host 等待 500 μs，确保后台已经进入执行；该等待发生在前台 event 记录之前，不计入前台时间。
- 后台循环按 round 改变访问位置，避免编译器把 global load 提出循环。SASS 留档确认循环内存在 `LDG`。
- 正式原始输出：`results/thor-stream-priority-20260809-155136.txt`
- SASS：`results/01_stream_priority.sass.txt`

## 当前边界

该实验覆盖单进程、单 CUDA context 内的 stream 调度。当前机器只有一张可用 GPU，且没有可用的 MIG/MPS 多租户配置，因此无法可靠外推到跨进程公平性、MPS active-thread percentage 或 MIG 隔离。更细的 instruction-level preemption 也需要系统级 trace/CUPTI 事件，而不能只靠 CUDA event 推断。
