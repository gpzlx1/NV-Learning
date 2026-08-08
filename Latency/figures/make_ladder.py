#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 results/ 的实测结果里提取阶梯图数据 -> figures/data/ladder.csv
用法: python3 figures/make_ladder.py [单卡结果.txt] [跨卡结果.txt]"""
import re, sys, os, glob, csv
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

# (在结果里匹配的子串, 图上显示名, 分类)  —— 每项取该行第一个形如 123.45 的数
ITEMS = [
    ("FFMA   fma.rn.f32",              "FFMA 依赖链",              "计算指令"),
    ("DFMA   fma.rn.f64",              "FP64 DFMA 依赖链",         "计算指令"),
    ("IMAD64 mad.lo.s64",              "64 位整数乘加",            "计算指令"),
    ("SHFL   shfl.sync.idx",           "warp shuffle",             "计算指令"),
    ("mma.sync m16n8k16",              "mma.sync m16n8k16",        "计算指令"),
    ("MUFU   rcp.approx",              "MUFU rcp (SFU)",           "计算指令"),
    ("wgmma m64n8k16 RS 连发",         "wgmma m64n8k16 完成",      "计算指令"),
    ("shared memory (LDS)",            "shared memory",            "存储读"),
    ("L1 命中 (24KB",                  "L1 命中",                  "存储读"),
    ("local memory (LDL",              "local memory (栈溢出)",    "存储读"),
    ("常量内存 (8KB",                  "常量内存 (数据相关下标)",  "存储读"),
    ("DSMEM 对端 (cluster=2",          "DSMEM 对端 (跨 SM)",       "存储读"),
    ("L2 命中 (32MB",                  "L2 命中",                  "存储读"),
    ("HBM (2GB, .cg)",                 "HBM",                      "存储读"),
    ("host pinned (32MB",              "host pinned (经 PCIe)",    "存储读"),
    ("-> L2  (32MB, st.cg)",           "写发射间隔 (任意目标端)",  "存储写"),
    ("shared: STS -> LDS",             "shared 写→读往返",         "存储写"),
    ("L1:  st.wb -> ld.ca",            "L1 写→读往返",             "存储写"),
    ("L2:  st.cg -> ld.cg",            "L2 写→读往返",             "存储写"),
    ("相减 => 真 st.cg->ld.cg 往返",   "HBM 写→读往返",            "存储写"),
    ("st.release.gpu -> ld.acquire",   "release/acquire 往返",     "存储写"),
    ("red.global.add.u32",             "red.global.add (无返回值)","同步与原子"),
    ("membar.cta (__threadfence_block)", "membar.cta",             "同步与原子"),
    ("barrier.sync (__syncthreads",    "__syncthreads (1 warp)",   "同步与原子"),
    ("atom.shared.add.u32",            "atom.shared.add",          "同步与原子"),
    ("mbarrier arrive + try_wait",     "mbarrier arrive+wait",     "同步与原子"),
    ("barrier.cluster arrive.relaxed", "cluster barrier (relaxed)","同步与原子"),
    ("atom.global.add.u32",            "atom.global.add",          "同步与原子"),
    ("membar.gl (__threadfence)",      "membar.gl (__threadfence)","同步与原子"),
    ("barrier.cluster arrive.release", "cluster barrier (release)","同步与原子"),
    ("membar.sys (__threadfence_system)", "membar.sys",            "同步与原子"),
]
P2P = [
    ("远端 32MB .ca",                  "NVLink 远端读",            "跨卡 (NVLink)"),
    ("远端: st.cg -> ld.cg",           "NVLink 写→读往返",         "跨卡 (NVLink)"),
    ("远端 atom.global.add.u32",       "NVLink 原子",              "跨卡 (NVLink)"),
    ("远端: st.release.sys",           "跨卡传标志 (release.sys)", "跨卡 (NVLink)"),
]
NUM = re.compile(r'(\d+\.\d\d)(?!\d)')

def pick(lines, needle):
    for L in lines:
        if needle in L:
            m = NUM.search(L)
            if m: return float(m.group(1))
    return None

def main():
    single = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(ROOT,"results/all-gpu*.txt")))[-1]
    p2p    = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob(os.path.join(ROOT,"results/p2p-*.txt")))[-1]
    L1 = open(single).read().split("\n"); L2 = open(p2p).read().split("\n")
    rows, miss = [], []
    for src, items in ((L1, ITEMS), (L2, P2P)):
        for needle, name, cat in items:
            v = pick(src, needle)
            if v is None: miss.append(needle)
            else: rows.append({"name": name, "category": cat, "cycles": f"{v:.2f}"})
    os.makedirs(os.path.join(HERE,"data"), exist_ok=True)
    with open(os.path.join(HERE,"data","ladder.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["name","category","cycles"]); w.writeheader(); w.writerows(rows)
    print(f"  ladder.csv: {len(rows)} 行  (来源 {os.path.basename(single)} + {os.path.basename(p2p)})")
    if miss: print("  !! 没匹配到:", miss)

if __name__ == "__main__": main()
