# 结果存档

顶层是**当前代码布局下的最新一份**结果，每次 `./run.sh` / `./p2p.sh` / `./ncu.sh` 都会新增带时间戳的文件。

| 文件 | 产生方式 | 内容 |
|---|---|---|
| `all-gpu3-*.txt` | `./run.sh` | 单卡全量：四大类共 6 个 benchmark，含 nvidia-smi 状态与 nvcc 版本 |
| `p2p-gpu6to7-*.txt` | `./p2p.sh` | 跨卡 NVLink，附 NVLink 硬件计数器前后增量作为流量验证 |
| `ncu-gpu3-clk_*.txt` | `./ncu.sh` | NCU 交叉验证：锁频复测 + 硬件计数器层级归属 |

每份结果都自带运行环境（卡号、pstate、SM/显存频率、温度、显存占用、nvcc 版本），便于回溯比较。

`archive/` 是历史结果，包括**重构前**的旧文件命名（`lat-gpu3-*`，那时是 `lat_hopper.cu` / `lat_mem.cu` 等按开发顺序命名的布局）。数值与当前一致，保留用于交叉对照。
