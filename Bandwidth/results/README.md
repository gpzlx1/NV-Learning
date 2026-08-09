# Bandwidth results

每次正式测试使用带 UTC 时间戳的文件保存。原始 CSV 风格输出包含设备信息、配置、毫秒和 GB/s，报告与绘图脚本直接解析这些文件。

运行 `../run.sh 0` 会构建并执行当前全部测试，同时生成对应 `.exit` 和关键 SASS。吞吐统一使用十进制 GB/s（`bytes / 1e9 / seconds`）；原子使用 logical Gop/s；Tensor Core 使用 FMA=2 FLOP 的 TFLOP/s。
