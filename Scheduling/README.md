# GPU Scheduling Architecture Exploration

本目录研究 stream priority、CTA wave 调度、并发 kernel 公平性和抢占边界。吞吐归 `Bandwidth/`，依赖链归 `Latency/`；当问题变成“谁先获得 SM、何时让出”时归本目录。

```bash
make
./01_stream_priority 0
```
