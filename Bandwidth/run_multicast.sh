#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"; dev="${1:-0}"; make 13_tma_multicast
run_retry(){ local mode="$1" cs="$2"; for attempt in 1 2 3; do if ./13_tma_multicast "$dev" "$mode" "$cs"; then return 0; fi; echo "retry mode=$mode cluster=$cs attempt=$attempt" >&2; done; return 1; }
for cs in 2 4; do for mode in unicast multicast; do run_retry "$mode" "$cs"; done; done
if [[ "${2:-}" == "--experimental-cs8" ]]; then for mode in unicast multicast; do run_retry "$mode" 8; done; fi
