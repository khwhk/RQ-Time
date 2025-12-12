#!/bin/bash

# Usage: bash ./scripts/pretrain.sh <codebook_size> <dim_model>
# IMPORTANT: Make sure to change the data_root to your own path.
# For debugging, add the flag "--dev" to the end of the command.
# export CUDA_VISIBLE_DEVICES=2,3
# echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"
PROJECT_ROOT=/data/wanghaokai/VQ_NEW
# 将项目根目录加入PYTHONPATH（优先级最高）
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}
# 强制切换到项目根目录
cd ${PROJECT_ROOT}
# =====================
codebook_size=$1
dim_model=$2
data_root=/data/wanghaokai/RQ-VAE-Project

python ./vqshape/pretrain.py \
    --data_root $data_root \
    --dim_embedding $dim_model \
    --normalize_length 96 \
    --patch_size 16 \
    --num_patch 6 \
    --num_token 12 \
    --num_transformer_enc_heads 8 \
    --num_transformer_enc_layers 8 \
    --num_tokenizer_heads 8 \
    --num_tokenizer_layers 4 \
    --num_transformer_dec_heads 8 \
    --num_transformer_dec_layers 2 \
    --num_code $codebook_size \
    --dim_code 16 \
    --codebook_type standard \
    --len_s 24 \
    --s_smooth_factor 1 \
    --lambda_x 1 \
    --lambda_z 1 \
    --lambda_s 1 \
    --lambda_dist 0.8 \
    --lambda_vq_commit 0.25 \
    --lambda_vq_entropy 0.1 \
    --entropy_gamma 1 \
    --lr 1e-4 \
    --batch_size 2048 \
    --accumulate_grad_batches 4 \
    --gradient_clip 1 \
    --weight_decay 0.01 \
    --mask_ratio 0.25 \
    --warmup_step 1000 \
    --train_epoch 100 \
    --val_frequency 0.2 \
    --name forecasting_dim"$dim_model"_codebook"$codebook_size" \
    --num_nodes 1 \
    --num_devices 1 \
    --strategy "auto" \
    --precision "bf16-mixed" \
    --num_workers 8 \
    --balance_datasets
    