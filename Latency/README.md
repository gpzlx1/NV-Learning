# Latency 微基准套件

本目录使用依赖链和低并发 kernel 测量“后一条操作必须等待前一条完成时”的延迟。研究结论与运行说明已经分开：

- [NVIDIA Thor / Blackwell 延迟报告](REPORT_BLACKWELL_THOR_20260809.md)
- [NVIDIA H800 / Hopper 延迟报告](REPORT_HOPPER_H800.md)
- [仓库综合研究报告](../REPORT.md)

## 测试矩阵

| 程序 | 研究对象 | 关键口径 |
|---|---|---|
| `01_mem_read` | shared、local、constant、L1/L2、device/host path | 指针追逐依赖读 |
| `02_mem_write` | local/shared/global/remote store | issue、store→load、visibility 分开 |
| `03_mem_levels` | footprint 台阶与单次延迟分布 | 容量 sweep、直方图、独立探针 |
| `03b_l2_partition` | 地址×SM 的 L2 延迟差异 | 多 SM、多物理地址 |
| `04_mem_p2p` | NVLink/P2P | 本地与远端读、写、原子、system scope |
| `05_inst` | 标量与 DPX | 依赖链 latency、独立指令 issue interval |
| `06_tensor` | Hopper `mma.sync` / `wgmma` | Hopper-only |
| `06b_tcgen05` | Blackwell `tcgen05.mma` | steady interval 与 commit/wait completion |
| `06c_tmem` | Tensor Memory | alloc、load/store、copy、wait |
| `07_sync` | barrier、fence、atomic | completion/visibility |
| `08_async_copy` | LDGSTS/`cp.async`、TMA | 发射、立即等待的端到端 latency |

## 复现

Thor：

```bash
./run_blackwell_thor.sh 0
```

脚本会构建 `sm_110`/`sm_110f` 目标、逐项运行并保存环境、stdout、退出码和关键 SASS 到新的时间戳目录。正式结果见 `results/blackwell-thor-20260809-124123/`。

H800：

```bash
make
./run.sh            # 默认 DEV=3，可用 DEV=N 覆盖
./p2p.sh            # 默认 GPU 6 -> 7
./ncu.sh            # 锁频与层级 counter 交叉验证
make sass
```

## 方法约束

1. 随机单环指针追逐使相邻 load 存在真实数据依赖，并抑制规则预取。
2. 双点斜率消去 `clock64`、循环控制和 kernel 启动的固定开销。
3. 周期和 `%globaltimer` 同时采集；片上结果以 cycle 为主，跨时钟域结果同时记录实测频率。
4. `.ca`/`.cg` 等 cache operator 只建立候选路径，层级归属应再由工作集容量和 NCU counter 验证。
5. 所有“往返”必须不短于对应单程，否则优先怀疑两条链并行或编译器优化。
6. 源码通过不代表微基准成立；关键循环必须检查 SASS，结果必须通过哨兵。

## 编译器风险清单

这些问题都曾真实产生“数值合理但含义错误”的结果：

| 风险 | 典型症状 | 防护 |
|---|---|---|
| 强度削减/重结合 | 依赖链被改写成更少指令 | 使用不可重结合操作并检查 SASS |
| warp-uniform 消除 | shuffle 链完全消失 | 让初值依赖 lane id |
| 对合/幂等折叠 | 两次操作被删成恒等 | 让每轮输入数据相关 |
| store-to-load forwarding | 往返低于单程 | 让地址运行时相同、编译期不可证明 |
| loop invariant hoisting | 循环内只剩简化操作 | 让所有操作数进入依赖链 |
| dead-store elimination | store interval 接近 0 | inline PTX `volatile` 与可观察结果 |
| 两条链意外并行 | 差分测到 `max()` 而不是和 | 把第二条链结果依赖传回第一条 |
| 单地址采样偏差 | L2/atomic 结果跨地址差异很大 | 采样多个地址和 SM |

## 结果与图表

- `results/`：原始输出、退出码、SASS 与 NCU CSV；
- `figures/data/`：从原始输出提取的绘图数据；
- `figures/`：专项报告使用的 PNG 与生成脚本。

图表的生成方法见 [`figures/README.md`](figures/README.md)，结果存档规则见 [`results/README.md`](results/README.md)。
