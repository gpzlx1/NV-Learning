#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Latency/ 的测量数据画成图。数据全部来自 figures/data/*.csv, 由
   ./03_mem_levels --dev 7 --csv   生成 (sweep.csv / hist.csv)
   ladder.csv 由 make_ladder.sh 从 results/ 的全量结果里提取
用法: python3 figures/plot.py     (在 Latency/ 目录下运行)
输出: figures/*.png
"""
import csv, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ── 中文字体 ────────────────────────────────────────────────────────────
for p in [os.path.expanduser("~/.local/share/fonts/SourceHanSansSC-Regular.otf"),
          "/usr/share/fonts/SourceHanSansSC-Regular.otf"]:
    if os.path.exists(p):
        fm.fontManager.addfont(p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# ── 配色 (validate_palette.js 校验通过: 相邻对 CVD ΔE 9.1 / 普通视觉 19.6) ──
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK2      = "#52514e"
INK3      = "#8a8983"
GRID      = "#e8e7e3"
SPINE     = "#d5d4cf"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
BAND      = "#f0efec"          # 区域底纹(中性灰, 不携带含义)

GHZ = 1.980                    # 卡 7 实测 SM 频率

def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)          # 实线, 不用虚线
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9.5, length=3, width=0.8)

def savefig(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"  已生成 {name}  ({os.path.getsize(out)/1024:.0f} KB)")

def bytes_label(b):
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if b >= div:
            v = b / div
            return f"{v:.0f}{unit}" if abs(v - round(v)) < 0.05 else f"{v:.1f}{unit}"
    return f"{b}B"

# ════════════════════════════════════════════════════════════════════════
# 图 1: 延迟 vs footprint —— 容量台阶
# ════════════════════════════════════════════════════════════════════════
def fig_sweep():
    xs, ys = [], []
    with open(os.path.join(DATA, "sweep.csv")) as f:
        for r in csv.DictReader(f):
            xs.append(int(r["bytes"])); ys.append(float(r["cycles"]))
    xs, ys = np.array(xs, dtype=float), np.array(ys)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    style(ax)
    ax.set_xscale("log", base=2)

    L1_CAP, L2_CAP = 256 * 1024, 50 * (1 << 20)
    # 三个容量区间用中性底纹分开(灰色, 不携带类别含义), 文字说明层级
    for lo, hi, name, detail in [
        (xs[0] * 0.85, L1_CAP, "L1 命中区", "256KB unified L1/smem"),
        (L1_CAP, L2_CAP, "L2 命中区", "50MB L2"),
        (L2_CAP, xs[-1] * 1.18, "HBM 区", "超出 L2 容量"),
    ]:
        ax.axvspan(lo, hi, color=BAND, alpha=0.55, lw=0, zorder=0)
        xm = math.sqrt(lo * hi)
        ax.text(xm, 838, name, ha="center", va="top", fontsize=11, color=INK, weight="bold")
        ax.text(xm, 806, detail, ha="center", va="top", fontsize=8.5, color=INK3)

    # 容量边界: 细实线(不用虚线)
    for cap in (L1_CAP, L2_CAP):          # 容量边界线; 刻度已经标了数值, 不再重复标
        ax.axvline(cap, color=SPINE, lw=1.2, zorder=1)

    ax.plot(xs, ys, color=S1, lw=2.0, marker="o", ms=5.0,
            mfc=SURFACE, mec=S1, mew=1.6, zorder=3, clip_on=False)

    # 三个平台的实测值直接标在图上(不是每点都标)
    for lo, hi, dy in [(xs[0], L1_CAP, 30), (1 << 20, 32 << 20, 30), (256 << 20, xs[-1], -40)]:
        m = (xs >= lo) & (xs <= hi)
        if not m.any():
            continue
        v = float(np.median(ys[m])); xm = math.sqrt(max(lo, xs[0]) * min(hi, xs[-1]))
        ax.annotate(f"≈ {v:.0f} 周期 / {v/GHZ:.0f} 纳秒", (xm, v), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=10.5, color=INK, weight="bold")

    ax.set_xlim(xs[0] * 0.85, xs[-1] * 1.18)
    ax.set_ylim(0, 880)
    ax.set_xticks([1 << k for k in range(14, 31, 2)])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: bytes_label(int(v))))
    ax.minorticks_off()
    ax.set_xlabel("footprint（随机指针追逐的工作集大小，对数轴）", fontsize=10.5, color=INK2, labelpad=9)
    ax.set_ylabel("依赖链延迟（周期）", fontsize=10.5, color=INK2)
    sec = ax.secondary_yaxis("right", functions=(lambda c: c / GHZ, lambda n: n * GHZ))
    sec.set_ylabel("延迟（纳秒）", fontsize=10.5, color=INK2)   # 同一数据换单位, 不是双轴
    sec.tick_params(colors=INK2, labelsize=9.5, length=3, width=0.8)
    sec.spines["right"].set_color(SPINE); sec.spines["right"].set_linewidth(0.8)

    ax.set_title("H800 存储层次的容量台阶", fontsize=14, color=INK, weight="bold", loc="left", pad=30)
    ax.text(0, 1.035, "单线程随机指针追逐 · ld.global.ca · 256B stride · 卡 7 · SM 1.980 GHz",
            transform=ax.transAxes, fontsize=9.5, color=INK3, va="bottom")
    savefig(fig, "fig1_footprint_sweep.png")

# ════════════════════════════════════════════════════════════════════════
# 图 2: 单次访问延迟直方图 —— L2 近/远分区双峰
# ════════════════════════════════════════════════════════════════════════
def split_two(c, p):
    """在两个主峰之间最深的谷处切开 -> 返回 (切点下标, 左组, 右组)。
    比逐峰检测稳: DRAM 那档的分布很宽, 逐峰会把同一簇拆成两个。"""
    n = len(p)
    sm = np.array([p[max(0, i-2):i+3].mean() for i in range(n)])
    i1 = int(np.argmax(sm))
    far = [i for i in range(n) if abs(i - i1) >= 8]
    if not far:
        return None, None, None
    i2 = max(far, key=lambda i: sm[i])
    a, b = min(i1, i2), max(i1, i2)
    cut = a + int(np.argmin(sm[a:b+1]))
    return cut, slice(0, cut+1), slice(cut+1, n)

def fig_hist():
    data = {}
    with open(os.path.join(DATA, "hist.csv")) as f:
        for r in csv.DictReader(f):
            data.setdefault(r["tag"], []).append((int(r["bin_lo_cycles"]), float(r["pct"])))

    # 显式 x 范围: 长尾会把主体挤成一小坨, 所以按主体设范围, 被裁掉的比例写进副标题
    panels = [
        ("L2_8MB",  (248, 334), "footprint 8MB —— 全部命中 L2",
         "左右两簇 = L2 近分区命中 / 远分区命中",
         [(258, "论文 258")], ("近分区命中", "远分区命中")),
        ("L2_40MB", (248, 334), "footprint 40MB —— 接近 50MB L2 容量",
         "主体仍是命中的两簇", [], ("近分区命中", "远分区命中")),
        ("HBM_2GB", (500, 900), "footprint 2GB —— 全部未命中，下到 DRAM",
         "左右两簇 = 近分区未命中 / 远分区未命中",
         [(555.5, "论文 555.5"), (743.7, "论文 743.7")], ("近分区未命中", "远分区未命中")),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.4))
    for ax, (tag, xlim, title, sub, refs, gname) in zip(axes, panels):
        style(ax)
        rows = sorted(data.get(tag, []))
        c = np.array([r[0] for r in rows], dtype=float)
        p = np.array([r[1] for r in rows])
        vis = (c >= xlim[0]) & (c <= xlim[1])
        clipped = float(p[~vis].sum())
        cv, pv = c[vis], p[vis]
        ax.bar(cv, pv, width=3.4, color=S1, lw=0, zorder=3)   # 宽3.4/间距4 -> 2px 表面缝隙
        ax.set_xlim(*xlim)
        ymax = pv.max() * 1.78
        ax.set_ylim(0, ymax)

        cut, L, R = split_two(cv, pv)
        if cut is not None:                    # 两簇各标一次; 上下错开, 避免互相压住
            for sl, nm, dy in ((L, gname[0], 52), (R, gname[1], 20)):
                w = pv[sl].sum()
                if w <= 1: continue
                m = float((cv[sl] * pv[sl]).sum() / w)
                yi = int(np.argmax(pv[sl])) + (sl.start or 0)
                ax.annotate(f"{nm}　均值 {m:.0f} 周期 · 占 {w:.0f}%", (cv[yi], pv[yi]),
                            textcoords="offset points", xytext=(0, dy), ha="center",
                            fontsize=10, color=INK, weight="bold",
                            arrowprops=dict(arrowstyle="-", color=INK3, lw=0.9,
                                            shrinkA=0, shrinkB=4))
            ax.plot([cv[cut], cv[cut]], [0, ymax * 0.52], color=SPINE, lw=1.0, zorder=2)

        for x, lab in refs:                    # 论文参考值: 细竖线 + 横排标签放在柱子上方空处
            if xlim[0] <= x <= xlim[1]:
                ax.plot([x, x], [0, ymax * 0.60], color=INK3, lw=1.0, zorder=2)
                ax.text(x, ymax * 0.62, lab, ha="center", va="bottom",
                        fontsize=8.5, color=INK3, zorder=5)

        mean = float((c * p).sum() / p.sum())
        note = sub + f"　·　整档均值 {mean:.0f} 周期"
        if clipped > 0.5:
            note += f"　·　另有 {clipped:.0f}% 的样本在视图右侧之外（更慢的未命中长尾）"
        ax.set_title(title, fontsize=11.5, color=INK, weight="bold", loc="left", pad=22)
        ax.text(0, 1.025, note, transform=ax.transAxes, fontsize=9, color=INK3, va="bottom")
        ax.set_ylabel("占比（%）", fontsize=10, color=INK2)
    axes[-1].set_xlabel("单次访问延迟（周期）", fontsize=10.5, color=INK2)
    fig.suptitle("L2 是双分区的：单次访问延迟呈双峰", fontsize=14, color=INK,
                 weight="bold", x=0.088, ha="left", y=0.998)
    fig.text(0.088, 0.966, "每档 16384 次采样 · 计时框架开销已扣除 · 卡 7　|　"
             "论文 far hit 414 在本机复现不出来：实测远分区命中只比近分区高约 33 周期",
             fontsize=9.5, color=INK3, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.952], h_pad=3.8)
    savefig(fig, "fig2_l2_partition_histogram.png")

# ════════════════════════════════════════════════════════════════════════
# 图 3: 延迟阶梯 —— 跨 1200 倍的全景
# ════════════════════════════════════════════════════════════════════════
def fig_ladder():
    path = os.path.join(DATA, "ladder.csv")
    if not os.path.exists(path):
        print("  跳过图 3: 缺 ladder.csv"); return
    groups, order = {}, []
    with open(path) as f:
        for r in csv.DictReader(f):
            g = r["category"]
            if g not in groups: groups[g] = []; order.append(g)
            groups[g].append((r["name"], float(r["cycles"])))
    colors = {g: c for g, c in zip(order, [S1, S2, S3, S4, S5])}

    rows = []                          # 分类成块(空间分离 -> 只需相邻对配色安全)
    for g in order:
        rows.append(("head", g, 0.0, g))
        for name, cyc in sorted(groups[g], key=lambda t: t[1]):
            rows.append(("item", name, cyc, g))
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11.6, 0.315 * n + 2.0))
    fig.subplots_adjust(left=0.265, right=0.985, top=0.952, bottom=0.052)
    style(ax)
    ax.set_xscale("log")
    ax.grid(True, axis="y", color=SURFACE)          # 只保留竖向网格

    from matplotlib.transforms import blended_transform_factory
    LT = blended_transform_factory(ax.transAxes, ax.transData)   # x 用坐标区比例, y 用数据

    ymax = n - 0.4
    for i, (kind, name, cyc, g) in enumerate(rows):
        y = ymax - i
        if kind == "head":
            ax.text(-0.012, y, name, transform=LT, va="center", ha="right",
                    fontsize=11.5, color=INK, weight="bold")     # 文字用墨色
            ax.plot([1.06], [y], marker="s", ms=9, color=colors[g],
                    clip_on=False, zorder=4)                     # 颜色只在色块上
            continue
        col = colors[g]
        ax.plot([1.0, cyc], [y, y], color=col, lw=2.0, alpha=0.45, zorder=2,
                solid_capstyle="round")
        ax.plot([cyc], [y], marker="o", ms=8.5, mfc=col, mec=SURFACE, mew=2.0, zorder=4)
        ax.text(-0.012, y, name, transform=LT, va="center", ha="right",
                fontsize=10, color=INK2)
        ax.text(cyc * 1.13, y, f"{cyc:,.0f}", va="center", ha="left",
                fontsize=10, color=INK, weight="bold")           # 每行都有可见数值标签
    ax.set_ylim(-0.9, ymax + 1.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(1.0, 13000)
    ax.set_xticks([1, 4, 10, 40, 100, 400, 1000, 4000, 10000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.minorticks_off()
    ax.set_xlabel("延迟（周期，对数轴）　　1 周期 = 0.505 纳秒", fontsize=10.5, color=INK2)
    H = 0.315 * n + 2.0                       # 图高(英寸), 用来把标题定位在顶部
    fig.text(0.016, 1 - 0.30/H, "H800 延迟全景：从 4 周期到 4961 周期，跨 1200 倍",
             fontsize=15, color=INK, weight="bold", ha="left", va="top")
    fig.text(0.016, 1 - 0.58/H,
             "单卡部分测于卡 7，跨卡部分测于卡 6→7 · SM 1.980 GHz · "
             "与 README 表格（卡 3）的小幅差异属正常的卡间与分配位置波动",
             fontsize=9.5, color=INK3, ha="left", va="top")
    savefig(fig, "fig3_latency_ladder.png")

if __name__ == "__main__":
    print("生成图表:")
    fig_sweep()
    fig_hist()
    fig_ladder()
