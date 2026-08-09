#!/usr/bin/env python3
"""Reduce Nsight Compute raw CSV files to architecture-facing counters."""
import csv, glob, os, re, sys

root=sys.argv[1] if len(sys.argv)>1 else sorted(glob.glob("results/ncu-*"))[-1]
paths=[p for p in sorted(glob.glob(os.path.join(root,"*.csv"))) if os.path.basename(p)!="summary.csv"]
if paths and not any(re.search(r'(global|shared-read|shared-write)-stride',p) for p in paths):
    out=[]
    for path in paths:
        lines=open(path,encoding="utf-8").readlines()
        start=next(i for i,x in enumerate(lines) if x.startswith('"ID"'))
        table=list(csv.reader(lines[start:])); hdr,units,val=table[0],table[1],table[2]
        data=dict(zip(hdr,val)); unit=dict(zip(hdr,units))
        for key in hdr:
            if key.startswith(("gpu__","l1tex__","lts__","smsp__","sm__")) and data[key] not in ("","n/a"):
                if key.endswith((".sum",".pct_of_peak_sustained_active","per_warp_active.pct")):
                    out.append((os.path.basename(path),data.get("Kernel Name",""),key,unit[key],data[key]))
    dest=os.path.join(root,"summary.csv")
    with open(dest,"w",newline="") as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(("probe","kernel","metric","unit","value"));w.writerows(out)
    print(dest);sys.exit(0)
rows=[]
for path in paths:
    lines=open(path,encoding="utf-8").readlines()
    start=next(i for i,x in enumerate(lines) if x.startswith('"ID"'))
    table=list(csv.reader(lines[start:]))
    hdr,unit,val=table[0],table[1],table[2]
    data=dict(zip(hdr,val)); units=dict(zip(hdr,unit))
    m=re.search(r'(global|shared-read|shared-write)-stride(\d+)',os.path.basename(path))
    mode,stride=m.group(1),int(m.group(2))
    def num(key):
        x=data.get(key,"0").replace(",","")
        return float(x) if x not in ("","n/a") else 0.0
    out={"mode":mode,"stride":stride,"duration_us":num("gpu__time_duration.sum")/1000}
    if mode=="global":
        req=num("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
        sec=num("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
        l2=num("lts__t_sectors_op_read.sum")
        out.update(requests=req,sectors=sec,sectors_per_request=sec/req if req else 0,
                   sector_util_pct=100*128/(sec/req*32) if sec else 0,
                   l1_hit_pct=num("l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct"),
                   l2_sectors=l2,l2_hit_pct=100*num("lts__t_sectors_op_read_lookup_hit.sum")/l2 if l2 else 0,
                   long_scoreboard_pct=num("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"),
                   lg_throttle_pct=num("smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct"))
    else:
        out.update(read_conflicts=num("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"),
                   write_conflicts=num("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"),
                   shared_ld_inst=num("smsp__inst_executed_op_shared_ld.sum"),
                   shared_st_inst=num("smsp__inst_executed_op_shared_st.sum"),
                   short_scoreboard_pct=num("smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"),
                   mio_throttle_pct=num("smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct"))
    rows.append(out)

keys=[]
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
dest=os.path.join(root,"summary.csv")
with open(dest,"w",newline="") as f:
    w=csv.DictWriter(f,keys,lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(dest)
