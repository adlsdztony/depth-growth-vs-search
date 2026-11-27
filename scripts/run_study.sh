#!/usr/bin/env bash
# Every training run reported in the paper, one line each.
# usage: bash scripts/run_study.sh OUT_DIR GPU_UUID [FILTER]
#   OUT_DIR   where campaign directories are created (one per run, never reused)
#   GPU_UUID  from `nvidia-smi -L`; use `cpu` only together with --smoke for a quick test
#   FILTER    optional grep pattern to run a subset, e.g. "w400" or "rand"
# Runs are sequential; to parallelise, call this script several times with disjoint FILTERs.
# A 3M-step run takes roughly 35-65 minutes with 12 CPU cores and one GPU.
set -euo pipefail
OUT=$1; UUID=$2; FILTER=${3:-.}
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}

configs() {
  for s in 1 2 3 4 5; do
    for d in 1 2 3 4; do echo "fixed_w400_d${d}_s$s --method fixed --depth $d --width 400 --seed $s"; done
    echo "grow_w400_late_s$s     --method grow --depth 1 --width 400 --grow-at 1000000 2000000 --seed $s"
    echo "grow_w400_early_s$s    --method grow --depth 1 --width 400 --grow-at 500000 1000000 --seed $s"
    echo "grow_w400_vearly_s$s   --method grow --depth 1 --width 400 --grow-at 200000 400000 --seed $s"
    echo "grow_w400_early4_s$s   --method grow --depth 1 --width 400 --grow-at 500000 1000000 1500000 --seed $s"
    echo "grow_w400_rand_s$s     --method grow --depth 1 --width 400 --grow-at 500000 1000000 --insert random --seed $s"
  done
  for s in 1 2 3 4 5; do
    for d in 1 2 3 4; do echo "fixed_w128_d${d}_s$s --method fixed --depth $d --width 128 --seed $s"; done
    echo "grow_w128_late_s$s     --method grow --depth 1 --width 128 --grow-at 1000000 2000000 --seed $s"
    echo "grow_w128_early_s$s    --method grow --depth 1 --width 128 --grow-at 500000 1000000 --seed $s"
    echo "grow_w128_early4_s$s   --method grow --depth 1 --width 128 --grow-at 500000 1000000 1500000 --seed $s"
  done
}

mkdir -p "$OUT"
configs | grep -E "$FILTER" | while read -r name args; do
  [ -e "$OUT/$name" ] && { echo "skip $name (exists)"; continue; }
  echo "$(date +%T) start $name"
  $PY "$HERE/runner/run_one.py" --out "$OUT/$name" --uuid "$UUID" $args
done
