#!/bin/bash

#SBATCH --job-name=bench-h100
#SBATCH --partition=seas_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=0-01:00:00
#SBATCH --mem=64gb
#SBATCH --array=0-52
#SBATCH --output=/n/holylabs/LABS/acc_lab/Users/golden/CS2881r/mem-run-estimator-fork/job_logs/bench_%x_%j.out
#SBATCH --error=/n/holylabs/LABS/acc_lab/Users/golden/CS2881r/mem-run-estimator-fork/job_logs/bench_%x_%j.err
#SBATCH --open-mode=append
#SBATCH --chdir=/n/holylabs/LABS/acc_lab/Users/golden/CS2881r/mem-run-estimator-fork/

OUT_DIR=/n/holylabs/LABS/acc_lab/Users/golden/CS2881r/mem-run-estimator-fork/outputs

# gemma_2b 7
# hf_T5 7
# timm_convnext_v2 8
# llama_v3_1b 9
# hf_clip 7
# timm_vit 7
# hf_GPT2 8

# Define model config counts
declare -A model_configs
model_configs[gemma_2b]=7
model_configs[hf_T5]=7
model_configs[timm_convnext_v2]=8
model_configs[llama_v3_1b]=9
model_configs[hf_clip]=7
model_configs[timm_vit]=7
model_configs[hf_GPT2]=8

# Calculate total configs and model offsets
total_configs=0
declare -A model_offsets

for model in "${!model_configs[@]}"; do
  model_offsets[$model]=$total_configs
  (( total_configs += ${model_configs[$model]} ))
done

# Determine model and config index
for model in "${!model_configs[@]}"; do
  if (( $SLURM_ARRAY_TASK_ID >= ${model_offsets[$model]} && $SLURM_ARRAY_TASK_ID < ${model_offsets[$model]} + ${model_configs[$model]} )); then
    MODEL_NAME=$model
    CONFIG_IDX=$(( $SLURM_ARRAY_TASK_ID - ${model_offsets[$model]} ))
    break
  fi
done

# #for single model run
# MODEL_NAME=hf_GPT2
# CONFIG_IDX=$SLURM_ARRAY_TASK_ID

# srun python driver.py \
#     --real_execution \
#     --model_name $MODEL_NAME \
#     --preset_config \
#     --config_idx $CONFIG_IDX

srun python driver.py \
    --runtime_estimation \
    --runtime_estimation_mode operator-level-benchmark \
    --model_name $MODEL_NAME \
    --preset_config \
    --config_idx $CONFIG_IDX
