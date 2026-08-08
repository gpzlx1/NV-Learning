#!/usr/bin/env bash
# NCU 交叉验证: (1) 锁频复测  (2) 硬件计数器验证"每一档到底命中在哪一级"
#  用法: ./ncu.sh          (固定卡 3, DEV=N 可换)
set -euo pipefail
cd "$(dirname "$0")"
DEV=${DEV:-3}
CLK=${CLK:-base}                   # base=锁到基频(默认) | none=不锁, 用于对比
make -s lat_hopper
mkdir -p results
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=results/ncu-gpu${DEV}-clk_${CLK}-${STAMP}.txt

M_ATTR=l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,\
lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum,\
dram__sectors_read.sum,gpc__cycles_elapsed.max

{
echo "############ (1) 锁频复测: ncu --clock-control ${CLK} 全程锁频 ############"
echo "# ncu 只 profile k_freq 一个 kernel(开销可忽略), 但锁频对整个进程生效。"
echo "# 脚本自己会用 %clock64 / %globaltimer 实测真实频率 -> 可直接看出锁频是否生效。"
ncu --clock-control "${CLK}" --kernel-name k_freq --launch-count 1 \
    --metrics gpc__cycles_elapsed.max --print-summary none \
    ./lat_hopper --dev "${DEV}" --core --dsmem 2>&1 | grep -vE '^==(PROF|WARN|ERROR)|^\s*$|Section:|Metric|^\s*-+\s*$|gpc__cycles'

echo
echo "############ (2) 层级归属验证: 每级一个独立 kernel, 看命中率 ############"
echo "# 期望: l1 -> L1 命中率高;  l2 -> L1 全 miss(.cg 绕过) 且 L2 命中率高;"
echo "#       hbm -> L2 几乎全 miss 且 dram 有实际读流量。"
ncu --clock-control "${CLK}" --kernel-name 'regex:(k_probe_|k_freq)' --cache-control none \
    --metrics "${M_ATTR}" ./lat_hopper --dev "${DEV}" --probe 2>&1 | grep -vE '^==(PROF|WARN)'
} 2>&1 | tee "${OUT}"
echo; echo "saved -> ${OUT}"
