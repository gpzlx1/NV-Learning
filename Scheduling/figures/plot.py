#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent / "thor_stream_priority.png"
labels = [
    "alone",
    "same priority\nmany CTAs",
    "high over low\nmany CTAs",
    "same priority\n1 CTA/SM",
    "high over low\n1 CTA/SM",
]
slowdown = [1.000, 88.118, 6.229, 1.892, 1.874]
colors = ["#4c78a8", "#e45756", "#72b7b2", "#f2cf5b", "#54a24b"]

fig, ax = plt.subplots(figsize=(9.2, 4.8))
bars = ax.bar(labels, slowdown, color=colors)
ax.set_yscale("log")
ax.set_ylim(0.8, 140)
ax.set_ylabel("Foreground slowdown (×, log scale)")
ax.set_title("Thor: stream priority acts at CTA admission, not resident-CTA preemption")
ax.grid(axis="y", which="both", alpha=0.25)
for bar, value in zip(bars, slowdown):
    ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:.2f}×",
            ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=180)
print(OUT)
