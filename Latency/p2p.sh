#!/usr/bin/env bash
# P2P / NVLink 延迟: 固定用卡 6(计算) 和卡 7(内存)。
#   SRC=4 DST=5 ./p2p.sh   可换卡对
# 顺便用 NVLink 硬件计数器验证流量真的走了 NVLink (相当于 NCU 的归属验证)
set -euo pipefail
cd "$(dirname "$0")"
SRC=${SRC:-6}; DST=${DST:-7}
make -s 04_mem_p2p
mkdir -p results
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=results/p2p-gpu${SRC}to${DST}-${STAMP}.txt
rx() { nvidia-smi nvlink -gt d -i "$1" 2>/dev/null | awk "/Data Rx/{s+=\$(NF-1)} END{printf \"%.0f\", s+0}"; }
tx() { nvidia-smi nvlink -gt d -i "$1" 2>/dev/null | awk '/Tx/{s+=$NF} END{print s+0}'; }
{
  echo "# $(date -Is)  host=$(hostname)  src=${SRC} dst=${DST}"
  nvidia-smi --query-gpu=index,name,pstate,clocks.sm,memory.used --format=csv -i "${SRC}","${DST}"
  echo "--- 拓扑 ---"
  nvidia-smi topo -m 2>/dev/null | awk -v a="GPU${SRC}" 'NR==1||$1==a' | cut -c1-100
  R0=$(rx "$SRC"); T0=$(tx "$SRC")
  echo
  ./04_mem_p2p --src "${SRC}" --dst "${DST}"
  R1=$(rx "$SRC"); T1=$(tx "$SRC")
  echo
  echo "═══ NVLink 计数器验证 (卡 ${SRC} 的全部 link 合计, 单位 KiB) ═══"
  echo "  Rx 增量: $(( R1 - R0 )) KiB = $(( (R1-R0)/1024 )) MiB"
  echo "  Tx 增量: $(( T1 - T0 )) KiB = $(( (T1-T0)/1024 )) MiB"
  echo "  (Rx 增量应远大于 0 —— 证明远端读确实经 NVLink 取回了数据)"
} 2>&1 | tee "${OUT}"
echo; echo "saved -> ${OUT}"
