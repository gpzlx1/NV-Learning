#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
dev="${1:-0}"
scope="${2:-all}"
stamp="$(date -u +%Y%m%d-%H%M%S)"
out="results/ncu-${stamp}"
mkdir -p "$out"
make 10_hardware_probe
cuobjdump -sass 10_hardware_probe > results/10_hardware_probe.sass.txt

global_metrics="gpu__time_duration.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read_lookup_miss.sum,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,sm__warps_active.avg.pct_of_peak_sustained_active"
shared_metrics="gpu__time_duration.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_shared_st.sum,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,sm__warps_active.avg.pct_of_peak_sustained_active"

profile() {
  local mode="$1" stride="$2" metrics="$3"
  local file="$out/${mode}-stride${stride}.csv"
  echo "Profiling $mode stride=$stride"
  sudo -n ncu --csv --page raw --metrics "$metrics" --log-file "$file" \
    ./10_hardware_probe "$mode" "$stride" "$dev" >/dev/null
}

if [[ "$scope" == all || "$scope" == global ]]; then
  for stride in 1 2 4 8 16 32; do profile global "$stride" "$global_metrics"; done
fi
if [[ "$scope" == all || "$scope" == shared ]]; then
  for mode in shared-read shared-write; do
    for stride in 1 2 4 8 16 32; do profile "$mode" "$stride" "$shared_metrics"; done
  done
fi
python3 parse_ncu.py "$out"
printf '%s\n' "$out"
