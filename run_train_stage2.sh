#!/bin/bash
#SBATCH --job-name=phase2_id_on_ds
#SBATCH --output=log/phase2_id_debug.log
#SBATCH --error=log/phase2_id_err.log
#SBATCH --gres=gpu:1


export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_JAX=1
export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

python -u train.py \
  --mode id_on_ds \
  --t5_path (doesn't matter)  \
  --data_path (for id training) \
  --phase1_data_path (for ds training) \
  --out_dir \
  --init_from (for id training, fill the path for ds model)  \
  --pool mean \
  --max_length 512 \
  --hidden_dim 256 \
  --ds_width 256 \
  --dropout 0.2 \
  --id_model_name_or_path t5-large \
  --id_max_source_len 512 \
  --id_max_target_len 16 \
  --num_train_epochs 15 \
  --per_device_train_batch_size 8\
  --per_device_eval_batch_size 8 \
  --learning_rate_id 3e-5 \
  --weight_decay_id 0.01 \
  --select_metric micro_f1 \
  --no_f1_drop_tol 0.05 \
  --min_precision_no -1 \
  --min_recall_no -1 \
  --val_ratio 0.1 \
  --el2n_key el2n_z \
  --stat_key stat_vector

