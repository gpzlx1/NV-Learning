#!/usr/bin/env python3
"""从 run.sh / 03_mem_levels --hist 的输出里提取延迟直方图的峰(mode)。
用法: ./analyze.py results/all-gpu3-*.txt        (不给参数则读最新的)
峰检测: 8cyc 粗分箱 -> 3点平滑 -> 找有 prominence 的局部极大值。"""
import re, sys, glob

def peaks(bins, binw=4, prom=0.35):
    """bins: [(lo_cycle, pct)]  -> [(center, weight_pct)]"""
    if not bins: return []
    lo0 = min(b[0] for b in bins); hi = max(b[0] for b in bins)
    n = (hi - lo0) // binw + 1
    h = [0.0] * n
    for lo, p in bins: h[(lo - lo0) // binw] += p
    s = [ (h[max(0,i-1)] + h[i] + h[min(n-1,i+1)]) / 3 for i in range(n) ]
    out = []
    for i in range(n):
        if s[i] < prom: continue
        if s[i] >= max(s[max(0,i-2):i+3]):                    # 局部极大
            if out and i - out[-1][0] < 4: continue           # 太近, 认为是同一个峰
            out.append((i, s[i]))
    res = []
    idx = [i for i, _ in out]
    # 用相邻峰之间的谷做边界, 保证每个 bin 只归属一个峰(权重之和 = 100%)
    bnd = [0]
    for a, b in zip(idx, idx[1:]):
        bnd.append(a + min(range(b - a), key=lambda k: s[a + k]))
    bnd.append(n)
    for k, i in enumerate(idx):
        l, r = bnd[k], bnd[k+1]
        w = sum(h[l:r]); ctr = (sum((lo0+j*binw)*h[j] for j in range(l, r)) / w) if w else 0
        res.append((ctr + binw/2, w))
    return sorted(res, key=lambda x: -x[1])

def main(path):
    sec, data = None, {}
    for L in open(path):
        m = re.match(r'\s*--- (.+?) ---\s+(\d+) 个样本, 均值 = ([\d.]+)', L)
        if m: sec = m.group(1).strip(); data[sec] = [float(m.group(3)), []]; continue
        m = re.match(r'\s*(-?\d+)-\s*(-?\d+) cyc\s+([\d.]+)%', L)
        if m and sec: data[sec][1].append((int(m.group(1)), float(m.group(3))))
    if not data: print(f"{path}: 没找到直方图 (需要 --hist 的输出)"); return
    print(f"# {path}")
    for k, (mean, bins) in data.items():
        print(f"\n{k}\n  均值 = {mean:.1f} 周期")
        for ctr, w in peaks(bins):
            if w < 2.0: continue
            print(f"  峰 @ {ctr:6.0f} 周期   占 {w:5.1f}%")

if __name__ == '__main__':
    files = sys.argv[1:] or sorted(glob.glob('results/all-gpu*.txt'))[-1:]
    for f in files: main(f)
