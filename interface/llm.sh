seeds=(7648 5912) # 0 1234 5678 1357 2468) # 13 8012 5479 9949 2550) # 4823 9174 2036) # 

for seed in "${seeds[@]}"; do
  sbatch interface/slurm_launcher.slrm llm_main.py \
    --data-path "my_datasets/moelcule_net" \
    --seed "${seed}"
done
