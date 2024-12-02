#!/bin/bash

# TODO: Add different seed to the BO campaign
sizes=(50000) # 10000) # 1000 500)
models=('equiformer_v2') # 'MPNN_Benchmark')
targets=('Multiple') # 'Single')
bayes_types=('Laplace') # 'GP')
seeds=(0) # 1234 5678 1357 2468) # 13 8012 5479 9949 2550) # 4823 9174 2036 7648 5912) #


# Loop through the combinations of sizes and targets
for size in "${sizes[@]}"; do
  for model in "${models[@]}"; do
    for target in "${targets[@]}"; do
      for type in "${bayes_types[@]}"; do
        for seed in "${seeds[@]}"; do
          sbatch interface/slurm_launcher.slrm main.py \
            --output-dir "results/laplace_${model}_${size// 1000/}_{${target}" \
            --model-name "${model}" \
            --data-path 'datasets/qm9' \
            --model-type "${type}" \
            --epochs 100 \
            --iterations 1000 \
            --seed "${seed}" \
            --initial-sample 10 \
            --dataset-size "${size}" \
            --encoder-target "${target}" \
            --exclusion 10
        done
      done
    done
  done
done