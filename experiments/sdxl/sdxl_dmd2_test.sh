#!/bin/bash

# Change to project directory
cd /home/zhijun/Code/DMD2

# Set environment variables
export CHECKPOINT_PATH=/data/sdxl
export WANDB_ENTITY=aaroncaozj_team
export WANDB_PROJECT=sdxl_dmd2_test
export PYTHONPATH=/home/zhijun/Code/DMD2:$PYTHONPATH

# HuggingFace model cache location
export HF_HOME=/home/zhijun/Code/DMD2/ckpt
export HF_HUB_CACHE=/home/zhijun/Code/DMD2/ckpt/hub
export TRANSFORMERS_CACHE=/home/zhijun/Code/DMD2/ckpt/transformers

# 2-GPU configuration (5090 + 5090D)
accelerate launch --config_file fsdp_configs/fsdp_1node_2x5090.yaml main/train_sd.py  \
    --generator_lr 5e-7  \
    --guidance_lr 5e-7 \
    --train_iters 100000000 \
    --output_path  $CHECKPOINT_PATH/sdxl_2gpu_test_lr5e-7_denoising4step_gan5e-3_guidance8  \
    --cache_dir $CHECKPOINT_PATH/cache \
    --log_path $CHECKPOINT_PATH/logs \
    --batch_size 1 \
    --grid_size 2 \
    --initialie_generator --log_iters 1000 \
    --resolution 1024 \
    --latent_resolution 128 \
    --seed 10 \
    --real_guidance_scale 8 \
    --fake_guidance_scale 1.0 \
    --max_grad_norm 10.0 \
    --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
    --wandb_iters 100 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --wandb_name "sdxl_2gpu_test_lr5e-7_denoising4step_gan5e-3_guidance8"  \
    --log_loss \
    --dfake_gen_update_ratio 5 \
    --fsdp \
    --sdxl \
    --use_fp16 \
    --max_step_percent 0.98 \
    --cls_on_clean_image \
    --gen_cls_loss \
    --gen_cls_loss_weight 5e-3 \
    --guidance_cls_loss_weight 1e-2 \
    --diffusion_gan \
    --diffusion_gan_max_timestep 1000 \
    --denoising \
    --num_denoising_step 4 \
    --denoising_timestep 1000 \
    --backward_simulation \
    --train_prompt_path $CHECKPOINT_PATH/captions_laion_score6.25.pkl \
    --real_image_path $CHECKPOINT_PATH/sdxl_vae_latents_laion_500k_lmdb/ 