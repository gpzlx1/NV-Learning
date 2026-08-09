#!/usr/bin/env python3
"""Generate the standalone NVIDIA Thor report figures from final raw results."""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "blackwell-thor-20260809-124123"
OUT = Path(__file__).resolve().parent

for font in [Path.home() / ".local/share/fonts/SourceHanSansSC-Regular.otf",
             Path("/usr/share/fonts/SourceHanSansSC-Regular.otf")]:
    if font.exists():
        fm.fontManager.addfont(font)
        plt.rcParams["font.family"] = fm.FontProperties(fname=font).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BG, INK, MUTED, GRID = "#fcfcfb", "#111111", "#5c5b57", "#deddd8"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def style(ax):
    ax.set_facecolor(BG)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(OUT / name)


def find_cycle(text, label):
    line = next(line for line in text.splitlines() if label in line)
    values = re.findall(r"\d+(?:\.\d+)?", line.split(label, 1)[1])
    return float(values[0])


def memory_ladder():
    read = (RAW / "01_mem_read.txt").read_text()
    items = [
        ("Shared memory", find_cycle(read, "shared memory (LDS)")),
        ("L1 hit", find_cycle(read, "L1 命中 (24KB, .ca)")),
        ("Local memory", find_cycle(read, "local memory (LDL, 2KB 栈溢出)")),
        ("Constant memory", find_cycle(read, "常量内存 (8KB, LDC + 寄存器偏移)")),
        ("DSMEM peer", find_cycle(read, "DSMEM 对端 (cluster=2, 跨 SM)")),
        ("L2 hit", find_cycle(read, "L2 命中 (32MB, .cg)")),
        ("Device DRAM", find_cycle(read, "HBM (2GB, .cg)")),
        ("Host pinned", find_cycle(read, "host pinned (32MB, zero-copy 经 PCIe)")),
    ]
    items.sort(key=lambda x: x[1])
    names, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    style(ax)
    ax.set_xscale("log")
    ax.barh(names, vals, color=BLUE, height=0.62)
    for y, v in enumerate(vals):
        ax.text(v * 1.06, y, f"{v:.0f} cycles", va="center", color=INK, fontsize=9.5)
    ax.set_xlim(10, 1500)
    ax.set_xlabel("Dependent-load latency (cycles, log scale)", color=MUTED)
    ax.set_title("NVIDIA Thor memory-latency ladder", loc="left", weight="bold", fontsize=15, pad=34)
    ax.text(0, 1.01, "Single-thread pointer chase · lower is better",
            transform=ax.transAxes, color=MUTED, fontsize=9.5)
    save(fig, "blackwell_thor_memory_ladder.png")


def footprint_sweep():
    rows = []
    for line in (RAW / "03_mem_levels-csv.txt").read_text().splitlines():
        m = re.match(r"#CSV sweep (\d+),([0-9.]+),", line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2))))
    xs, ys = zip(*rows)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    style(ax)
    ax.set_xscale("log", base=2)
    ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=4)
    ax.axvspan(xs[0], 256 << 10, color=GREEN, alpha=0.08)
    ax.axvspan(256 << 10, 32 << 20, color=BLUE, alpha=0.07)
    ax.axvspan(32 << 20, xs[-1], color=ORANGE, alpha=0.07)
    ax.text(64 << 10, 745, "L1 region", ha="center", weight="bold", color=GREEN)
    ax.text(4 << 20, 745, "L2 region", ha="center", weight="bold", color=BLUE)
    ax.text(256 << 20, 745, "Beyond L2", ha="center", weight="bold", color=ORANGE)
    ax.set_ylim(0, 880)
    ax.set_ylabel("Dependent-load latency (cycles)", color=MUTED)
    ax.set_xlabel("Random pointer-chase footprint (log₂ scale)", color=MUTED)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f"{x/(1<<30):g} GB" if x >= 1 << 30 else
                     f"{x/(1<<20):g} MB" if x >= 1 << 20 else f"{x/(1<<10):g} KB"))
    ax.set_title("Cache-capacity transitions on NVIDIA Thor", loc="left", weight="bold", fontsize=15, pad=34)
    ax.text(0, 1.01, "32 KB–2 GB sweep · ld.global.ca · 256-byte stride",
            transform=ax.transAxes, color=MUTED, fontsize=9.5)
    save(fig, "blackwell_thor_footprint_sweep.png")


def tensor_tmem():
    mma = (RAW / "06b_tcgen05.txt").read_text()
    tmem = (RAW / "06c_tmem.txt").read_text()
    items = [
        ("MMA steady interval\nN=8/16/32", find_cycle(mma, "N8K16 overwrite"), "MMA"),
        ("MMA completion\ncommit + barrier", find_cycle(mma, "N8K16 每条 commit"), "MMA"),
        ("TMEM load", find_cycle(tmem, "tcgen05.ld 16x256b.x1 (TMEM->register)"), "TMEM"),
        ("TMEM store issue", find_cycle(tmem, "tcgen05.st 16x256b.x1 连发"), "TMEM"),
        ("TMEM store completion", find_cycle(tmem, "tcgen05.st + wait::st"), "TMEM"),
        ("TMEM store→load", find_cycle(tmem, "TMEM store->load"), "TMEM"),
        ("SMEM→TMEM copy issue", find_cycle(tmem, "SMEM->TMEM) 连发"), "TMEM"),
        ("SMEM→TMEM completion", find_cycle(tmem, "每条 commit/wait"), "TMEM"),
        ("TMEM alloc + dealloc", find_cycle(tmem, "alloc + dealloc (32 columns)"), "TMEM"),
    ]
    items.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(10, 6.1))
    style(ax)
    names = [x[0] for x in items]
    vals = [x[1] for x in items]
    colors = [ORANGE if x[2] == "MMA" else GREEN for x in items]
    ax.barh(names, vals, color=colors, height=0.64)
    for y, v in enumerate(vals):
        ax.text(v + 7, y, f"{v:.2f}", va="center", fontsize=9.5, color=INK)
    ax.set_xlim(0, 440)
    ax.set_xlabel("Latency / steady-state interval (cycles)", color=MUTED)
    ax.set_title("Fifth-generation Tensor Core and TMEM costs", loc="left", weight="bold", fontsize=15, pad=34)
    ax.text(0, 1.01, "Orange: tcgen05 MMA · Green: Tensor Memory operations",
            transform=ax.transAxes, color=MUTED, fontsize=9.5)
    save(fig, "blackwell_thor_tcgen05_tmem.png")


if __name__ == "__main__":
    memory_ladder()
    footprint_sweep()
    tensor_tmem()
