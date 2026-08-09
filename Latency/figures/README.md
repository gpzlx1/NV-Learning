# 图表

全部由 `plot.py` 从 `data/` 里的实测数据生成，不是手工画的。

## NVIDIA Thor 独立报告图

`plot_blackwell_thor.py` 从最终回归目录
`results/blackwell-thor-20260809-124123/` 直接解析数据，生成：

| 图 | 内容 |
|---|---|
| `blackwell_thor_memory_ladder.png` | shared、L1、DSMEM、L2、DRAM 与 host pinned 的依赖读 latency |
| `blackwell_thor_footprint_sweep.png` | 32 KB–2 GB cache-capacity 台阶 |
| `blackwell_thor_tcgen05_tmem.png` | tcgen05 MMA 与 TMEM issue/completion latency |

```bash
python3 figures/plot_blackwell_thor.py
```

Thor 图片只供独立报告使用，不覆盖下面的 H800 图片与数据。

```bash
# 1. 采集数据 (卡 7)
./03_mem_levels --dev 7 --csv > /tmp/csv.txt
grep '^#CSV sweep' /tmp/csv.txt | sed 's/^#CSV sweep //' > figures/data/sweep.csv
grep '^#CSV hist '  /tmp/csv.txt | sed 's/^#CSV hist //'  > figures/data/hist.csv
# 2. 阶梯图数据: 从全量结果里提取
python3 figures/make_ladder.py results/all-gpu7-*.txt results/p2p-*.txt
# 3. 出图
# 2b. L2 分区映射热力图数据
./03b_l2_partition --dev 3 --csv | grep '^#CSV map' | sed 's/^#CSV map //' > figures/data/l2map.csv
# 3. 出图
python3 figures/plot.py
```

| 图 | 对应章节 | 数据源 |
|---|---|---|
| `fig1_footprint_sweep.png` | 1.3 容量台阶 | `data/sweep.csv`（34 个 footprint 点，√2 步长） |
| `fig2_l2_partition_histogram.png` | 1.3 L2 近/远分区 | `data/hist.csv`（3 档 × 16384 次单次访问采样） |
| `fig3_latency_ladder.png` | 总览 | `data/ladder.csv`（35 项，从全量结果里提取） |
| `fig4_l2_sm_partition_map.png` | 1.3 L2 双分区映射 | `data/l2map.csv`（132 SM × 8 地址，由 `03b_l2_partition --csv` 生成） |

## 依赖

- `matplotlib` + `numpy`
- 中文字体：脚本会在 `~/.local/share/fonts/SourceHanSansSC-Regular.otf` 找思源黑体。缺字体时中文会渲染成方框，装一个即可：
  ```bash
  curl -L -o ~/.local/share/fonts/SourceHanSansSC-Regular.otf \
    https://mirrors.tuna.tsinghua.edu.cn/adobe-fonts/source-han-sans/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf
  ```

## 配色

用的是一套经过校验的分类配色（相邻对 CVD 色盲可分辨性 ΔE 9.1、普通视觉 ΔE 19.6，均超过阈值）。三条约束：

- 单序列的图（曲线、直方图）只用一个蓝色，不需要图例。
- 热力图是"量的大小"，所以用**单色相顺序色阶**（蓝色浅→深），不用彩虹色；色条上直接标出了近/远两簇的中心值。
- 阶梯图的 5 个分类**按块空间分离**（不是穿插排列），所以只需满足相邻对可分辨。
- 其中 3 个色相在浅色背景上对比度低于 3:1 → 按规则**每行都配了可见的文字标签和数值**，颜色从不单独承载信息。

坐标轴与网格都用实线（虚线会增加视觉噪声），延迟跨 1200 倍所以阶梯图用对数轴 + 点状标记（而不是柱状——对数轴上柱长比例会误导）。
