#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")"
dev="${1:-0}"
stamp="$(date -u +%Y%m%d-%H%M%S)"
outdir="results/blackwell-thor-${stamp}"
mkdir -p "$outdir"

make blackwell-thor 2>&1 | tee "$outdir/build.txt"
build_rc=${PIPESTATUS[0]}
if (( build_rc != 0 )); then
  printf '%s\n' "$build_rc" > "$outdir/build.exit"
  exit "$build_rc"
fi
printf '0\n' > "$outdir/build.exit"

run_one() {
  local name="$1" limit="$2"
  shift 2
  printf '\n===== %s =====\n' "$name"
  timeout --signal=TERM "$limit" "$@" 2>&1 | tee "$outdir/${name}.txt"
  local rc=${PIPESTATUS[0]}
  printf '%s\n' "$rc" > "$outdir/${name}.exit"
  return 0
}

run_one 01_mem_read        10m ./01_mem_read --dev "$dev"
run_one 02_mem_write       10m ./02_mem_write --dev "$dev"
run_one 03_mem_levels      20m ./03_mem_levels --dev "$dev" --all
run_one 03_mem_levels-probe 10m ./03_mem_levels --dev "$dev" --probe
run_one 03_mem_levels-csv   20m ./03_mem_levels --dev "$dev" --csv
run_one 03b_l2_partition   10m ./03b_l2_partition --dev "$dev"

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
if (( gpu_count >= 2 )); then
  run_one 04_mem_p2p 10m ./04_mem_p2p
else
  printf 'SKIP: requires at least two GPUs; found %s\n' "$gpu_count" | tee "$outdir/04_mem_p2p.txt"
  printf '77\n' > "$outdir/04_mem_p2p.exit"
fi

run_one 05_inst             10m ./05_inst --dev "$dev"
printf 'SKIP: Hopper WGMMA is not executable on sm_110; replaced by 06b_tcgen05\n' | tee "$outdir/06_tensor.txt"
printf '77\n' > "$outdir/06_tensor.exit"
run_one 06b_tcgen05         10m ./06b_tcgen05 --dev "$dev"
run_one 06c_tmem            10m ./06c_tmem --dev "$dev"
run_one 07_sync             10m ./07_sync --dev "$dev"
run_one 08_async_copy       10m ./08_async_copy --dev "$dev"

for bin in 05_inst 06b_tcgen05 06c_tmem; do
  cuobjdump -sass "$bin" > "$outdir/${bin}.sass.txt"
done

printf '\nExit summary (0=pass, 77=hardware/architecture skip):\n'
for f in "$outdir"/*.exit; do printf '%-24s %s\n' "$(basename "$f" .exit)" "$(<"$f")"; done
printf 'Raw results: %s\n' "$outdir"
