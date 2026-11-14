#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/run_all.sh /path/to/RealWaste /path/to/dataset-resized" >&2
  exit 2
fi
REALWASTE="$1"
TRASHNET="$2"
for dataset in realwaste trashnet; do
  if [ "$dataset" = realwaste ]; then ROOT="$REALWASTE"; else ROOT="$TRASHNET"; fi
  mkdir -p "results/cache/$dataset"
  for model in clip siglip openclip; do
    python scripts/run_zero_shot.py --data "$ROOT" --dataset "$dataset" --backbone "$model" \
      --reference-split "reference/splits/$dataset.csv" --cache-dir "results/cache/$dataset" --out "results/zero_shot/$dataset/$model"
  done
  for protocol in experiment8 linguistic; do
    if [ "$protocol" = experiment8 ]; then BOOT=2000; else BOOT=5000; fi
    python scripts/run_controls.py --data "$ROOT" --dataset "$dataset" --protocol "$protocol" \
      --reference-split "reference/splits/$dataset.csv" --cache-dir "results/cache/$dataset" --out "results/controls/$dataset/$protocol" --bootstrap "$BOOT"
  done
  python scripts/run_few_shot.py --data "$ROOT" --dataset "$dataset" \
    --reference-split "reference/splits/$dataset.csv" --cache-dir "results/cache/$dataset" --out "results/few_shot/$dataset" --repeats 30
 done
for dataset in realwaste trashnet; do
  python scripts/sensitivity_analysis.py --predictions "results/controls/$dataset/linguistic/predictions.csv" \
    --dataset "$dataset" --out "paper_outputs/sensitivity_$dataset.csv"
done
python scripts/make_paper_tables.py --results results --out paper_outputs
python scripts/verify_reference.py --results results --reference reference
