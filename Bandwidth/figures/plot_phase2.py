#!/usr/bin/env python3
from pathlib import Path
import csv,io,re
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BG="#fcfcfb";C=["#2a78d6","#eb6834","#1baf7a"]
def csvblock(path,header):
 lines=Path(path).read_text().splitlines();i=next(i for i,x in enumerate(lines) if x.startswith(header));return list(csv.DictReader(io.StringIO("\n".join(lines[i:i+4]))))
l2=csvblock(ROOT/"results/thor-l2-residency-20260809-153516.txt","hot_MiB")
fig,ax=plt.subplots(figsize=(8.5,4.7),facecolor=BG);x=[int(r["hot_MiB"]) for r in l2]
ax.plot(x,[float(r["normal_reload_GBps"]) for r in l2],"o-",lw=2,label="Normal",color=C[0]);ax.plot(x,[float(r["persist_reload_GBps"]) for r in l2],"o-",lw=2,label="Persisting window",color=C[1]);ax.axvline(24,color="gray",ls="--",label="24 MiB persisting reserve");ax.set(xlabel="Hot working set (MiB)",ylabel="Reload GB/s",title="L2 access-policy protects reuse from 128 MiB pollution",xticks=x);ax.grid(alpha=.25);ax.legend(frameon=False);fig.tight_layout();fig.savefig(Path(__file__).parent/"thor_l2_persistence.png",dpi=180);plt.close(fig)

txt=(ROOT/"results/thor-dsmem-topology-20260809-152510.txt").read_text().splitlines();runs=[];cur=[]
for line in txt:
 if line.startswith("=== run"):
  if cur:runs.append(cur)
  cur=[]
 elif re.match(r'^[248],',line):cur.append(next(csv.reader([line])))
if cur:runs.append(cur)
fig,ax=plt.subplots(figsize=(9,4.8),facecolor=BG)
for cs,color in zip([2,4,8],C):
 vals=[]
 for run in runs:
  vals.extend(float(r[4]) for r in run if int(r[0])==cs)
 ax.scatter([cs]*len(vals),vals,s=28,alpha=.65,color=color,label=f"cluster {cs}")
ax.set(xlabel="Cluster size",ylabel="Aggregate DSMEM read GB/s",title="DSMEM has process-level placement states, not stable rank-distance cost",xticks=[2,4,8]);ax.grid(alpha=.25);ax.legend(frameon=False);fig.tight_layout();fig.savefig(Path(__file__).parent/"thor_dsmem_topology_states.png",dpi=180);plt.close(fig)

lines=sorted((ROOT/"results").glob("thor-tma-multicast-*.txt"))[-1].read_text().splitlines();mc=[next(csv.reader([x])) for x in lines if x.startswith(("unicast,","multicast,"))];by={(r[0],int(r[1])):r for r in mc};x=sorted({int(r[1]) for r in mc if ("unicast",int(r[1])) in by and ("multicast",int(r[1])) in by})
fig,ax=plt.subplots(figsize=(8.5,4.7),facecolor=BG)
ax.plot(x,[float(by[("unicast",i)][3]) for i in x],"o-",lw=2,label="Unicast delivered",color=C[0]);ax.plot(x,[float(by[("multicast",i)][3]) for i in x],"o-",lw=2,label="Multicast delivered",color=C[1]);ax.plot(x,[float(by[("multicast",i)][2]) for i in x],"s--",lw=2,label="Multicast source bytes",color=C[2]);ax.set(xlabel="Cluster size / fanout",ylabel="GB/s",title="Multicast preserves delivery while removing duplicate TMA issues",xticks=x);ax.grid(alpha=.25);ax.legend(frameon=False);fig.tight_layout();fig.savefig(Path(__file__).parent/"thor_tma_multicast.png",dpi=180);plt.close(fig)
