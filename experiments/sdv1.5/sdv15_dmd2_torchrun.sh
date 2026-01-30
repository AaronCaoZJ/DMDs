#!/bin/bash

# Change to project directory
# cd /home/zhijun/Code/DMD2

# Set environment variables
export CHECKPOINT_PATH=/data/sdv15
export WANDB_ENTITY=aaroncaozj_team
export WANDB_PROJECT=sdv15_dmd2_test
export PYTHONPATH=/home/zhijun/Code/DMD2:$PYTHONPATH

# HuggingFace model cache location
export HF_HOME=/home/zhijun/Code/DMD2/ckpt
export HF_HUB_CACHE=/home/zhijun/Code/DMD2/ckpt/hub
export TRANSFORMERS_CACHE=/home/zhijun/Code/DMD2/ckpt/transformers

# Distributed settings for 1 node 2 GPUs
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"

# 使用 torchrun 启动
# --nproc_per_node=2 指定双卡
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    main/train_sd.py \
    --generator_lr 1e-5  \
    --guidance_lr 1e-5 \
    --train_iters 100000000 \
    --output_path  $CHECKPOINT_PATH/sdv15_2gpu_torchrun_test \
    --cache_dir $CHECKPOINT_PATH/cache \
    --log_path $CHECKPOINT_PATH/logs \
    --batch_size 4 \
    --grid_size 2 \
    --initialie_generator --log_iters 1000 \
    --resolution 512 \
    --latent_resolution 64 \
    --seed 10 \
    --real_guidance_scale 1.75 \
    --fake_guidance_scale 1.0 \
    --max_grad_norm 10.0 \
    --model_id "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --train_prompt_path $CHECKPOINT_PATH/captions_laion_score6.25.pkl \
    --real_image_path $CHECKPOINT_PATH/sd_vae_latents_laion_500k_lmdb \
    --wandb_iters 50 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_name "sdv15_torchrun_bf16"  \
    --wandb_project $WANDB_PROJECT \
    --use_fp16 \
    --log_loss \
    --dfake_gen_update_ratio 10 \
    --fsdp 


