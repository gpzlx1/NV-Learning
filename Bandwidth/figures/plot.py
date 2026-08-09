#!/usr/bin/env python3
from pathlib import Path
import csv, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
raw = sorted((ROOT / "results").glob("thor-global-*.txt"))[-1].read_text()
BG, BLUE, ORANGE, GREEN, GRID = "#fcfcfb", "#2a78d6", "#eb6834", "#1baf7a", "#deddd8"

def table(title, next_title=None):
    s = raw.split(title, 1)[1]
    if next_title: s = s.split(next_title, 1)[0]
    lines = [x for x in s.strip().splitlines() if x and not x.startswith("===")]
    lines = lines[next(i for i, line in enumerate(lines) if "," in line):]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))

scale = table("=== Scaling", "=== Read access")
width = table("=== Read access", "=== Coalescing")
stride = table("=== Coalescing", "observable_sink")

fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=BG)
x = [int(r["blocks_per_sm"]) for r in scale]
for key, label, color in [("read_GBps", "Read", BLUE), ("write_GBps", "Write", ORANGE),
                          ("copy_fabric_GBps", "Copy read+write", GREEN)]:
    ax.plot(x, [float(r[key]) for r in scale], marker="o", lw=2.2, label=label, color=color)
ax.set(xticks=x, xlabel="Resident work offered (blocks/SM)", ylabel="GB/s",
       title="NVIDIA Thor device-memory bandwidth saturation")
ax.grid(color=GRID); ax.legend(frameon=False, ncol=3); ax.set_facecolor(BG)
fig.savefig(Path(__file__).parent / "thor_global_scaling.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), facecolor=BG)
axes[0].bar([r["access_bits"] for r in width], [float(r["requested_GBps"]) for r in width], color=BLUE)
axes[0].set(title="Access width", xlabel="Bits per load", ylabel="Requested GB/s")
axes[1].plot([int(r["stride_bytes"]) for r in stride],
             [float(r["requested_GBps"]) for r in stride], marker="o", lw=2.2, color=ORANGE)
axes[1].set_xscale("log", base=2); axes[1].set_xticks([4,8,16,32,64,128], labels=[4,8,16,32,64,128])
axes[1].set(title="Coalescing efficiency", xlabel="Stride between lanes (bytes)", ylabel="Requested GB/s")
for ax in axes: ax.grid(axis="y", color=GRID); ax.set_facecolor(BG)
fig.suptitle("Vector width and strided-access sensitivity", weight="bold", fontsize=14)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "thor_width_stride.png", dpi=180, bbox_inches="tight")
plt.close(fig)

def read_table(path, header, end=None):
    text = path.read_text().split(header, 1)[1]
    if end: text = text.split(end, 1)[0]
    lines = [x for x in text.strip().splitlines() if x and not x.startswith("===")]
    lines = lines[next(i for i,x in enumerate(lines) if "," in x):]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))

cache_path = sorted((ROOT / "results").glob("thor-cache-*.txt"))[-1]
cache = read_table(cache_path, "=== Working-set", "=== Cache operator")
shared_path = sorted(p for p in (ROOT / "results").glob("thor-shared-*.txt") if "patterns" not in p.name)[-1]
shared = read_table(shared_path, "=== Shared-memory", "observable_sink")
dsm_rw = sorted((ROOT / "results").glob("thor-dsmem-rw-*.txt"))
dsm_path = dsm_rw[-1] if dsm_rw else sorted((ROOT / "results").glob("thor-dsmem-*.txt"))[-1]
dsm = list(csv.DictReader(io.StringIO("\n".join(
    x for x in dsm_path.read_text().splitlines() if x and (x[0].isdigit() or x.startswith("cluster_size"))))))

levels = [("Device read", max(float(r["read_GBps"]) for r in scale)),
          ("L2 shared 16MB", float(cache[1]["requested_GBps"])),
          ("L1 private/SM", float(cache[0]["requested_GBps"])),
          ("Shared read", float(shared[0]["read_GBps"]))]
fig, ax = plt.subplots(figsize=(9, 4.7), facecolor=BG)
ax.barh([x[0] for x in levels], [x[1] for x in levels], color=[ORANGE,BLUE,GREEN,"#8c62bd"])
for i,(_,v) in enumerate(levels): ax.text(v+90,i,f"{v:.0f} GB/s",va="center")
ax.set(title="Bandwidth rises toward on-chip storage", xlabel="Aggregate requested GB/s", xlim=(0,7600))
ax.grid(axis="x",color=GRID); ax.set_facecolor(BG)
fig.tight_layout(); fig.savefig(Path(__file__).parent/"thor_bandwidth_hierarchy.png",dpi=180,bbox_inches="tight");plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(11,4.8),facecolor=BG)
x=[int(r["bank_conflict_degree"]) for r in shared]
axes[0].plot(x,[float(r["read_GBps"]) for r in shared],marker="o",label="Read",color=BLUE)
axes[0].plot(x,[float(r["write_GBps"]) for r in shared],marker="o",label="Write",color=ORANGE)
axes[0].set(xscale="log",yscale="log",xticks=x,xticklabels=x,title="Shared-memory bank conflicts",xlabel="Conflict degree",ylabel="Requested GB/s");axes[0].legend(frameon=False)
x2=[int(r["cluster_size"]) for r in dsm]
one_key="one_way_read_GBps" if "one_way_read_GBps" in dsm[0] else "one_way_GBps"
ring_key="ring_read_GBps" if "ring_read_GBps" in dsm[0] else "ring_aggregate_GBps"
axes[1].plot(x2,[float(r[one_key]) for r in dsm],marker="o",label="One reader/cluster",color=BLUE)
axes[1].plot(x2,[float(r[ring_key]) for r in dsm],marker="o",label="All CTAs read ring",color=GREEN)
if "ring_write_GBps" in dsm[0]: axes[1].plot(x2,[float(r["ring_write_GBps"]) for r in dsm],marker="o",label="All CTAs write ring",color=ORANGE)
axes[1].set(xticks=x2,title="DSMEM cluster-network scaling",xlabel="Cluster size",ylabel="Aggregate requested GB/s");axes[1].legend(frameon=False)
for ax in axes:ax.grid(color=GRID);ax.set_facecolor(BG)
fig.tight_layout();fig.savefig(Path(__file__).parent/"thor_shared_dsmem.png",dpi=180,bbox_inches="tight");plt.close(fig)

sp_path=sorted((ROOT/"results").glob("thor-shared-patterns-*.txt"))[-1]
sp=list(csv.DictReader(io.StringIO("\n".join(x for x in sp_path.read_text().splitlines()
    if x.startswith(("pattern,","broadcast_","conflict_","same_bank_"))))))
fig,ax=plt.subplots(figsize=(9,4.8),facecolor=BG)
labels=[f'{r["pattern"].replace("_"," ")}\n{r["access_bits"]} bit' for r in sp]
ax.bar(labels,[float(r["requested_GBps"]) for r in sp],color=[ORANGE if r["pattern"].startswith("broadcast") else BLUE if r["pattern"].startswith("conflict") else GREEN for r in sp])
ax.set(title="Independent shared-read pattern validation",ylabel="Logical requested GB/s");ax.tick_params(axis="x",labelrotation=25);ax.grid(axis="y",color=GRID);ax.set_facecolor(BG)
fig.tight_layout();fig.savefig(Path(__file__).parent/"thor_shared_patterns.png",dpi=180,bbox_inches="tight");plt.close(fig)

atom_path=sorted((ROOT/"results").glob("thor-atomics-*.txt"))[-1]
ga=read_table(atom_path,"=== Global atomic","=== Shared atomic")
sa=read_table(atom_path,"=== Shared atomic","observable_sink")
copy_path=sorted((ROOT/"results").glob("thor-copy-engines-*.txt"))[-1]
copy=list(csv.DictReader(io.StringIO("\n".join(x for x in copy_path.read_text().splitlines()
    if x.startswith(("direction,","D2D_","H2D_","D2H_"))))))
fig,axes=plt.subplots(1,2,figsize=(11,4.7),facecolor=BG)
axes[0].plot([int(r["independent_addresses"]) for r in ga],[float(r["Gops"]) for r in ga],marker="o",color=ORANGE)
axes[0].set(xscale="log",title="Global atomic contention",xlabel="Independent addresses",ylabel="Logical Gop/s")
axes[1].bar([r["direction"] for r in copy],[float(r["GBps"]) for r in copy],color=[BLUE,ORANGE,GREEN])
axes[1].set(title="CUDA copy engines",ylabel="Payload GB/s")
for ax in axes:ax.grid(axis="y",color=GRID);ax.set_facecolor(BG)
fig.tight_layout();fig.savefig(Path(__file__).parent/"thor_atomics_copy.png",dpi=180,bbox_inches="tight");plt.close(fig)

tc_path=sorted((ROOT/"results").glob("thor-tcgen05-*.txt"))[-1]
tc=list(csv.DictReader(io.StringIO("\n".join(x for x in tc_path.read_text().splitlines()
    if x.startswith(("shape,","M64"))))))
fig,ax=plt.subplots(figsize=(8.5,4.5),facecolor=BG)
ax.bar([r["shape"] for r in tc],[float(r["TFLOPs"]) for r in tc],color=ORANGE)
for i,r in enumerate(tc):ax.text(i,float(r["TFLOPs"])+1,f'{float(r["TFLOPs"]):.2f}',ha="center")
ax.set(title="tcgen05 F16→F32 sustained throughput",ylabel="TFLOP/s",ylim=(0,52));ax.grid(axis="y",color=GRID);ax.set_facecolor(BG)
fig.tight_layout();fig.savefig(Path(__file__).parent/"thor_tcgen05_throughput.png",dpi=180,bbox_inches="tight");plt.close(fig)

async_path=sorted((ROOT/"results").glob("thor-async-*.txt"))[-1]
ac=list(csv.DictReader(io.StringIO("\n".join(x for x in async_path.read_text().splitlines()
    if x.startswith(("inflight_groups,","1,","2,","4,","8,"))))))
fig,ax=plt.subplots(figsize=(8.5,4.5),facecolor=BG)
ax.plot([int(r["inflight_groups"]) for r in ac],[float(r["requested_GBps"]) for r in ac],marker="o",lw=2.2,color=BLUE)
ax.axhline(max(float(r["read_GBps"]) for r in scale),color=ORANGE,lw=1.5,label="Synchronous 128-bit read peak")
ax.set(xticks=[1,2,4,8],title="cp.async bandwidth versus in-flight groups",xlabel="Committed groups before wait",ylabel="Requested GB/s",ylim=(230,275));ax.grid(color=GRID);ax.legend(frameon=False);ax.set_facecolor(BG)
fig.tight_layout();fig.savefig(Path(__file__).parent/"thor_async_pipeline.png",dpi=180,bbox_inches="tight");plt.close(fig)

tma_path=sorted((ROOT/"results").glob("thor-tma-*.txt"))[-1]
tma=list(csv.DictReader(io.StringIO("\n".join(x for x in tma_path.read_text().splitlines()
    if x.startswith(("transfer_bytes,","128,","512,","1024,"))))))
fig,ax=plt.subplots(figsize=(8.5,4.7),facecolor=BG)
for size,color in [(128,ORANGE),(512,GREEN),(1024,BLUE)]:
    rows=[r for r in tma if int(r["transfer_bytes"])==size]
    ax.plot([int(r["batch_depth"]) for r in rows],[float(r["requested_GBps"]) for r in rows],marker="o",lw=2,label=f"{size} B",color=color)
ax.set(xticks=[1,4,8],title="TMA needs both large transfers and batching",xlabel="Transfers covered by one mbarrier wait",ylabel="Requested GB/s");ax.grid(color=GRID);ax.legend(frameon=False,title="Transfer size");ax.set_facecolor(BG)
fig.tight_layout();fig.savefig(Path(__file__).parent/"thor_tma_pipeline.png",dpi=180,bbox_inches="tight");plt.close(fig)

import numpy as np
intensity=np.logspace(-1,3,300);mem_peak=max(float(r["read_GBps"]) for r in scale)/1000.0
compute_peak=max(float(r["TFLOPs"]) for r in tc);perf=np.minimum(mem_peak*intensity,compute_peak)
fig,ax=plt.subplots(figsize=(8.5,4.8),facecolor=BG);ax.loglog(intensity,perf,color=BLUE,lw=2.4)
balance=compute_peak/mem_peak;ax.axvline(balance,color=ORANGE,lw=1.5);ax.text(balance*1.08,0.2,f"Balance ≈ {balance:.0f} FLOP/B",rotation=90,color=ORANGE)
ax.axhline(compute_peak,color=GREEN,lw=1.5,label=f"tcgen05 ceiling {compute_peak:.1f} TFLOP/s")
ax.set(title="Microbenchmark roofline for NVIDIA Thor",xlabel="Arithmetic intensity (FLOP/byte)",ylabel="Attainable TFLOP/s",ylim=(0.02,80));ax.grid(which="both",color=GRID);ax.legend(frameon=False);ax.set_facecolor(BG)
fig.tight_layout();fig.savefig(Path(__file__).parent/"thor_roofline.png",dpi=180,bbox_inches="tight");plt.close(fig)

# Nsight Compute counters from the deterministic one-launch hardware probe.
ncu=ROOT/"results"/"ncu-20260809-140740"/"summary.csv"
if ncu.exists():
    nr=list(csv.DictReader(ncu.open()))
    g=sorted((r for r in nr if r["mode"]=="global"),key=lambda r:int(r["stride"]))
    fig,ax=plt.subplots(figsize=(8.5,4.7),facecolor=BG);x=[int(r["stride"]) for r in g]
    ax.plot(x,[float(r["sectors_per_request"]) for r in g],"o-",lw=2.2,color=BLUE,label="32-B sectors / warp request")
    ax.set_xscale("log",base=2);ax.set_xticks(x,x);ax.set(xlabel="Lane stride (words)",ylabel="Sectors per request",title="NCU exposes transaction amplification")
    ax2=ax.twinx();ax2.plot(x,[float(r["sector_util_pct"]) for r in g],"s--",lw=2,color=ORANGE,label="Useful sector bytes");ax2.set_ylabel("Sector utilization (%)")
    lines=ax.lines+ax2.lines;ax.legend(lines,[l.get_label() for l in lines],frameon=False,loc="center right");ax.grid(color=GRID);ax.set_facecolor(BG)
    fig.tight_layout();fig.savefig(Path(__file__).parent/"thor_ncu_coalescing.png",dpi=180,bbox_inches="tight");plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),facecolor=BG)
    for mode,color in [("shared-read",GREEN),("shared-write",ORANGE)]:
        s=sorted((r for r in nr if r["mode"]==mode),key=lambda r:int(r["stride"]));xs=[int(r["stride"]) for r in s]
        conflict=[]
        for r in s:
            inst=float(r["shared_ld_inst"] if mode=="shared-read" else r["shared_st_inst"])
            raw=float(r["read_conflicts"] if mode=="shared-read" else r["write_conflicts"])
            conflict.append(raw/inst if inst else 0)
        axes[0].plot(xs,conflict,"o-",lw=2,label=mode,color=color)
        axes[1].plot(xs,[float(r["mio_throttle_pct"]) for r in s],"o-",lw=2,label=mode,color=color)
    axes[0].plot([1,2,4,8,16,32],[0,1,3,7,15,31],"k--",alpha=.4,label="degree - 1")
    for ax in axes: ax.set_xscale("log",base=2);ax.set_xticks([1,2,4,8,16,32],[1,2,4,8,16,32]);ax.set_xlabel("Conflict degree");ax.grid(color=GRID);ax.legend(frameon=False);ax.set_facecolor(BG)
    axes[0].set_ylabel("Conflicts / shared instruction");axes[0].set_title("Hardware detects both read and write conflicts")
    axes[1].set_ylabel("MIO throttle (% active warp cycles)");axes[1].set_title("Only write drives sustained pipe pressure")
    fig.tight_layout();fig.savefig(Path(__file__).parent/"thor_ncu_shared_counters.png",dpi=180,bbox_inches="tight");plt.close(fig)
