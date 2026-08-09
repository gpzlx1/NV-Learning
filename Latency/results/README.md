# 结果存档

顶层是**当前代码布局下的最新一份**结果，每次 `./run.sh` / `./p2p.sh` / `./ncu.sh` 都会新增带时间戳的文件。

`blackwell-thor-20260809-124123/` 是 NVIDIA Thor (`sm_110`) 的独立正式回归结果。
每项均有单独输出和退出码；0 表示通过，77 表示由于硬件或架构不适用而跳过。
该目录对应 `../REPORT_BLACKWELL_THOR_20260809.md`，不与 H800 结果混用。

| 文件 | 产生方式 | 内容 |
|---|---|---|
| `all-gpu3-*.txt` | `./run.sh` | 单卡全量：四大类共 6 个 benchmark，含 nvidia-smi 状态与 nvcc 版本 |
| `p2p-gpu6to7-*.txt` | `./p2p.sh` | 跨卡 NVLink，附 NVLink 硬件计数器前后增量作为流量验证 |
| `ncu-gpu3-clk_*.txt` | `./ncu.sh` | NCU 交叉验证：锁频复测 + 硬件计数器层级归属 |

每份结果都自带运行环境（卡号、pstate、SM/显存频率、温度、显存占用、nvcc 版本），便于回溯比较。

`archive/` 是历史结果，包括**重构前**的旧文件命名（`lat-gpu3-*`，那时是 `lat_hopper.cu` / `lat_mem.cu` 等按开发顺序命名的布局）。数值与当前一致，保留用于交叉对照。
