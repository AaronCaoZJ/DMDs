#!/bin/bash

# 使用原始SD1.5 teacher模型进行标准采样推理

cd /home/zhijun/Code/DMD2

export PYTHONPATH=/home/zhijun/Code/DMD2:$PYTHONPATH
export HF_HOME=/home/zhijun/Code/DMD2/ckpt
export HF_HUB_CACHE=/home/zhijun/Code/DMD2/ckpt/hub
export TRANSFORMERS_CACHE=/home/zhijun/Code/DMD2/ckpt/transformers

# 运行teacher模型推理
python main/simple_inference_teacher.py \
    --output_dir "/data/sdv15/test_output/teacher_newprompt/cfg7.5" \
    --seed 42 \
    --device cuda \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --scheduler ddim \
    --prompts \
        "A blue apple sitting next to a red banana on a wooden table" \
        "a beautiful landscape with mountains and a lake at sunset" \
        "a cute orange cat sitting on a windowsill" \
        "A teddy bear made of shiny chrome metal, reflecting a neon city" \
        "a portrait of a woman with flowers in her hair" \
        "a steaming cup of coffee on a wooden table" \
        "A crowd waits by along the St. Patrick's Day Parade route on 5th Avenue in 1951 New York" \
        "Ubein Bridge in Burma" \
        "The Most Romantic Hot Air Balloon Rides In The World Blog Masai Mara Nature Reserve, Kenya" \
