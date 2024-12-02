#!/bin/bash

# TODO: Add different seed to the BO campaign
bayes_types=('GP') # 'Benchmark')
seeds=(13 8012 5479 9949 2550 4823 9174 2036 7648 5912 0 1234 5678 1357 2468)

# Loop through the combinations of sizes and targets
for type in "${bayes_types[@]}"; do
  for seed in "${seeds[@]}"; do
    sbatch interface/slurm_launcher.slrm benchmarks.py \
      --output-dir "results/Benchmark_${type}" \
      --model-name "Benchmark" \
      --data-path 'datasets/rdkit_folder' \
      --model-type "${type}" \
      --epochs 100 \
      --iterations 1000 \
      --seed "${seed}" \
      --initial-sample 10 \
      --exclusion 10
  done
done
