#!/bin/bash

# TODO: Careful with the output directory, the targat is controlled from the parameter file
sizes=(500)              # 1000) # 50000) # 10000)
models=('equiformer_v2') # 'MPNN_Benchmark')

# Loop through the combinations of sizes and targets
for size in "${sizes[@]}"; do
  for model in "${models[@]}"; do
    sbatch Interface/slurm_launcher.slrm pretrain.py \
      --output-dir "results/models/laplace_${model}_${size//1000/}_Single" \
      --model-name "${model}" \
      --data-path 'datasets/qm7b' \
      --epochs 1000 \
      --dataset-size "${size}"
    # --resume-training 1
  done
done
