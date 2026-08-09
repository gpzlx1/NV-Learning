#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
stamp="$(date -u +%Y%m%d-%H%M%S)"; out="results/ncu-pipelines-${stamp}"; mkdir -p "$out"
make 04_dsmem 07_tcgen05_throughput 08_async_copy 09_tma
common="gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active"
run() { local name="$1" metrics="$2"; shift 2; echo "Profiling $name"; sudo -n ncu --csv --page raw --launch-count 1 --metrics "$common,$metrics" --log-file "$out/$name.csv" "$@" >/dev/null; }
run ldgsts "l1tex__t_requests_pipe_lsu_mem_global_op_ldgsts_cache_access.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access.sum,smsp__inst_executed_op_ldgsts.sum,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct" ./08_async_copy
run tma "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,smsp__inst_executed_op_tma_ld.sum,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct" ./09_tma
run dsmem "l1tex__t_requests_pipe_lsu_mem_dshared_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_dshared_op_ld.sum,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct" ./04_dsmem
echo "Profiling tcgen05"
sudo -n ncu --csv --page raw --kernel-name 'regex:k_tcgen05' --launch-count 1 \
  --metrics "$common,smsp__inst_executed_pipe_tc.sum,smsp__inst_executed_pipe_tc_scope_1cta.sum,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct" \
  --log-file "$out/tcgen05.csv" ./07_tcgen05_throughput >/dev/null
python3 parse_ncu.py "$out"
printf '%s\n' "$out"
