#!/bin/bash

# 简单推理脚本 - 使用训练好的SD1.5模型进行4步推理

cd /home/zhijun/Code/DMD2

export PYTHONPATH=/home/zhijun/Code/DMD2:$PYTHONPATH
export HF_HOME=/home/zhijun/Code/DMD2/ckpt
export HF_HUB_CACHE=/home/zhijun/Code/DMD2/ckpt/hub
export TRANSFORMERS_CACHE=/home/zhijun/Code/DMD2/ckpt/transformers

# 设置checkpoint路径
CHECKPOINT_DIR="/data/sdv15/cache/time_1769923833_seed10/checkpoint_model_000500"

# 找到最新的checkpoint
LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/pytorch_model.bin 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Using checkpoint: $LATEST_CHECKPOINT"

# 运行推理 - 使用4步backward simulation
python main/simple_inference.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --output_dir "/data/sdv15/test_output/0201_5-6gid_1-5gen_1.75realgid/decoupled_dmd_500" \
    --seed 42 \
    --device cuda \
    --num_denoising_step 4 \
    --denoising_timestep 1000 \
    --prompts \
        "a beautiful landscape with mountains and a lake at sunset" \
        "a cute orange cat sitting on a windowsill" \
        "a futuristic cyberpunk city with neon lights" \
        "a portrait of a woman with flowers in her hair" \
        "a steaming cup of coffee on a wooden table" \
        "A crowd waits by along the St. Patrick's Day Parade route on 5th Avenue in 1951 New York." \
        "Ubein Bridge in Burma" \
        "The Most Romantic Hot Air Balloon Rides In The World Blog Masai Mara Nature Reserve, Kenya."

