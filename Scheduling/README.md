# Scheduling 微基准套件

本目录研究并发工作何时获得 GPU 驻留与执行机会。它不测单条指令完成时间，也不把并发 wall time 当成单 workload 峰值带宽。

- [Thor Stream Priority 与 CTA 调度报告](REPORT_BLACKWELL_THOR_20260809.md)
- [仓库综合研究报告](../REPORT.md)

```bash
make
./01_stream_priority
make sass
```

正式输出见 [`results/thor-stream-priority-20260809-155136.txt`](results/thor-stream-priority-20260809-155136.txt)，关键循环 SASS 见 `results/01_stream_priority.sass.txt`。
