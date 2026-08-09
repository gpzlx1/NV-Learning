#!/usr/bin/env python3
from pathlib import Path
import csv,io
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];C=["#2a78d6","#eb6834","#1baf7a","#8055a5"]
def rows(path,header):
 l=Path(path).read_text().splitlines();i=next(i for i,x in enumerate(l) if x.startswith(header));return list(csv.DictReader(io.StringIO("\n".join(l[i:]))))
ct=[r for r in rows(R/"results/thor-concurrent-traffic-20260809-154828.txt","pair,") if r.get("serial_over_wall")]
fig,ax=plt.subplots(figsize=(9,4.8));names=[r["pair"] for r in ct];ax.bar(names,[float(r["serial_over_wall"]) for r in ct],color=C[0]);ax.axhline(1,color="gray",ls="--");ax.set(ylabel="Serial time / concurrent wall time",title="Two-stream overlap depends on traffic mix");ax.tick_params(axis="x",rotation=20);ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(Path(__file__).parent/"thor_concurrent_overlap.png",dpi=180);plt.close(fig)
mlp=[r for r in rows(R/"results/thor-mlp-occupancy-20260809-154828.txt","blocks_per_sm") if r.get("ILP")]
fig,ax=plt.subplots(figsize=(8.5,4.8));
for j,b in enumerate([1,2,4,8]):
 q=[r for r in mlp if int(r["blocks_per_sm"])==b];ax.plot([int(r["ILP"]) for r in q],[float(r["GBps"]) for r in q],"o-",lw=2,label=f"{b} blocks/SM",color=C[j])
ax.set(xlabel="Independent loads per thread (ILP)",ylabel="GB/s",title="ILP can substitute for occupancy",xticks=[1,2,4,8]);ax.grid(alpha=.25);ax.legend(frameon=False);fig.tight_layout();fig.savefig(Path(__file__).parent/"thor_mlp_occupancy.png",dpi=180);plt.close(fig)
