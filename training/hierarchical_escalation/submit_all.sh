#!/bin/bash
# Submit the v6 hierarchical-escalation sweeps.
#   ./submit_all.sh          -> both arms
#   ./submit_all.sh hier     -> recurrent two-level model only
#   ./submit_all.sh esc      -> escalation-features + shallow classifier only
cd "$(dirname "$0")" || exit 1
mkdir -p training_outputs
case "${1:-all}" in
  hier) JOBS="hier.sub" ;;
  esc)  JOBS="esc.sub" ;;
  all)  JOBS="esc.sub hier.sub" ;;   # esc first: minutes, not GPU-days
  *)    echo "usage: $0 [hier|esc|all]"; exit 1 ;;
esac
# --requeue matches v4/v5 and is also set inside the .sub files.
for j in $JOBS; do sbatch --requeue "$j"; done
