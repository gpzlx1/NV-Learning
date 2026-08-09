#!/usr/bin/env bash
set -u -o pipefail
cd "$(dirname "$0")"
dev="${1:-0}"
stamp="$(date -u +%Y%m%d-%H%M%S)"
mkdir -p results
make all || exit $?

run_one() {
  local name="$1"
  local slug="$2"
  printf '\n===== %s =====\n' "$name"
  "./$name" --dev "$dev" 2>&1 | tee "results/thor-${slug}-${stamp}.txt"
  printf '%s\n' "${PIPESTATUS[0]}" > "results/thor-${slug}-${stamp}.exit"
}

run_one 01_global_memory global
run_one 02_cache_working_set cache
run_one 03_shared_memory shared
run_one 03b_shared_patterns shared-patterns
run_one 04_dsmem dsmem
run_one 05_atomics atomics
run_one 06_copy_engines copy-engines
run_one 07_tcgen05_throughput tcgen05
run_one 08_async_copy async
run_one 09_tma tma

run_positional() {
  local name="$1"
  local slug="$2"
  printf '\n===== %s =====\n' "$name"
  "./$name" "$dev" 2>&1 | tee "results/thor-${slug}-${stamp}.txt"
  printf '%s\n' "${PIPESTATUS[0]}" > "results/thor-${slug}-${stamp}.exit"
}
run_positional 11_l2_residency l2-residency
run_positional 12_dsmem_topology dsmem-topology
printf '\n===== 13_tma_multicast (isolated process per configuration) =====\n'
./run_multicast.sh "$dev" 2>&1 | tee "results/thor-tma-multicast-${stamp}.txt"
printf '%s\n' "${PIPESTATUS[0]}" > "results/thor-tma-multicast-${stamp}.exit"

for name in 01_global_memory 02_cache_working_set 03_shared_memory 03b_shared_patterns 04_dsmem \
            05_atomics 07_tcgen05_throughput 08_async_copy 09_tma 10_hardware_probe \
            11_l2_residency 12_dsmem_topology 13_tma_multicast; do
  cuobjdump -sass "$name" > "results/${name}.sass.txt"
done
python3 figures/plot.py
python3 figures/plot_phase2.py
printf '\nResults timestamp: %s\n' "$stamp"
