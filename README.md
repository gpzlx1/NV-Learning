# GPUProfiling

本仓库用可复现的 CUDA 微基准研究 GPU 的延迟、吞吐与调度。主平台是 NVIDIA Thor（Blackwell，`sm_110`）；H800（Hopper，`sm_90a`）作为独立对照平台。两台机器的内存形态、频率与互连不同，因此报告不会把它们的绝对数值直接混合比较。

## 阅读顺序

如果只读一份文档，请从 **[综合研究报告](REPORT.md)** 开始。它按论文结构串联全部实验，回答三个问题：

1. 一次依赖访问要等多久，代价来自哪一层；
2. 并发如何隐藏延迟并形成持续吞吐；
3. 多工作负载共享 GPU 时，带宽与调度资源如何竞争。

专项报告用于查阅完整数据、实验边界和复现入口：

| 文档 | 平台 | 关注问题 |
|---|---|---|
| [Thor 延迟报告](Latency/REPORT_BLACKWELL_THOR_20260809.md) | Thor / Blackwell | 存储层次、写完成、同步、tcgen05、TMEM |
| [H800 延迟报告](Latency/REPORT_HOPPER_H800.md) | H800 / Hopper | 本地与跨卡延迟、L2 地址/SM 差异、微基准方法 |
| [Thor 吞吐报告](Bandwidth/REPORT_BLACKWELL_THOR_20260809.md) | Thor / Blackwell | MLP、coalescing、cache、DSMEM、TMA、并发与 Roofline |
| [Thor 调度报告](Scheduling/REPORT_BLACKWELL_THOR_20260809.md) | Thor / Blackwell | stream priority、CTA admission 与抢占边界 |

各目录的 `README.md` 只负责代码结构、运行方法和结果索引，不再兼任研究报告。

## 仓库结构

```text
Latency/      依赖链与低并发测试：测完成延迟
Bandwidth/    高并发与 CUDA event 测试：测持续吞吐
Scheduling/   多 stream 干扰测试：测工作何时获得执行资源
```

三类指标不可互换：一条 load 的依赖延迟不等于 load pipe 的倒数吞吐；store 的发射间隔不等于数据可见时间；两个 stream 能并发提交也不等于拥有两套独立硬件。

## 复现入口

```bash
cd Latency
./run_blackwell_thor.sh 0   # Thor 完整延迟回归
./run.sh                    # H800 单卡回归；DEV=N 可覆盖设备号
./p2p.sh                    # H800 P2P，默认 GPU 6 -> 7

cd ../Bandwidth
make
./run.sh                    # 吞吐主套件

cd ../Scheduling
make
```

原始输出保存在各目录的 `results/`。报告中的关键结论均应至少能回溯到原始输出与生成该输出的 kernel；涉及编译器优化风险的项目还保存 SASS，涉及数据路径判断的项目优先使用 Nsight Compute counter 交叉验证。

## 解释约定

- **测量事实**：由本仓库原始数据直接支持；
- **交叉验证**：SASS、独立实现或硬件计数器与测量事实一致；
- **机制解释**：结合 CUDA/PTX 官方语义对现象作出的解释；
- **架构推断**：公开资料不足时的最小假设，不写成已经证实的物理结构。

所有带宽默认说明 requested bytes、payload bytes 或 read+write fabric bytes 中的哪一种；Thor 是 ATS/统一内存平台，脚本遗留的 `HBM`、`PCIe` 标签不能自动等同于离散 GPU 的物理链路。
