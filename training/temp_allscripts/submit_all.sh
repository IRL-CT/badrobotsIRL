#!/bin/bash
# Submit the sweeps in this folder.
#   ./submit_all.sh          -> everything (11 jobs)
#   ./submit_all.sh linear   -> knn mlp rf sgd svm
#   ./submit_all.sh deep     -> gru/lstm/transformer, binary + multiclass
cd "$(dirname "$0")" || exit 1
mkdir -p training_outputs

LINEAR="knn.sub mlp.sub rf.sub sgd.sub svm.sub"
DEEP="gru_binary.sub gru_multiclass.sub lstm_binary.sub lstm_multiclass.sub transformer_binary.sub transformer_multiclass.sub"

case "${1:-all}" in
  linear) JOBS="$LINEAR" ;;
  deep)   JOBS="$DEEP" ;;
  all)    JOBS="$LINEAR $DEEP" ;;
  *)      echo "usage: $0 [linear|deep|all]"; exit 1 ;;
esac

for j in $JOBS; do
  sbatch --requeue "$j"
done
