#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"; stamp="$(date -u +%Y%m%d-%H%M%S)"; out="results/ncu-l2-${stamp}"; mkdir -p "$out"; make 11_l2_residency
metrics="gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read_lookup_miss.sum,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
for spec in 4:normal:2 4:persist:29 16:normal:56 16:persist:83 32:normal:110 32:persist:137; do IFS=: read -r size mode skip <<<"$spec"; echo "$size MiB $mode"; sudo -n ncu --csv --page raw --kernel-name 'regex:stream_read' --launch-skip "$skip" --launch-count 1 --metrics "$metrics" --log-file "$out/hot${size}-${mode}.csv" ./11_l2_residency 0 >/dev/null; done
python3 parse_ncu.py "$out"; printf '%s\n' "$out"
