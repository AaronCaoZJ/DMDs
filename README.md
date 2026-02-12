## 📄 Paper Info
### [Improved Distribution Matching Distillation for Fast Image Synthesis](https://tianweiy.github.io/dmd2/)          

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman        

*NeurIPS 2024 ([arXiv 2405.14867](https://arxiv.org/abs/2405.14867))*  

[[Huggingface Repo](https://huggingface.co/tianweiy/DMD2)][[ComfyUI](https://gist.github.com/comfyanonymous/fcce4ced378f74f4c46026b134faf27a)][[Colab](https://colab.research.google.com/drive/1iGk7IW2WosophOVYpdW_KZGIfYpATOm7?usp=sharing)]

## 🔥 Environment Setup
```bash
conda create -n dmd2 python=3.10 -y
conda activate dmd2

pip install --upgrade anyio
pip install -r requirements.txt
python setup.py  develop
# you may need to update some of the package if necessary
```

## 🚀 Inference Example
Details in [README-og.md/Inference Example](README-og.md)

## 🛁 Distillation Training and Evaluation

### ImageNet-64x64 

Please refer to [ImageNet-64x64](experiments/imagenet/README.md) for details.

### SDXL

Please refer to [SDXL](experiments/sdxl/README.md) for details.

### SDv1.5 

#### Training

First, it is necessary to prepare the dataset and related pre-trained weights, appropriately modify the target path in the script, and then run it:

```bash
bash scripts/download_sdv15.sh
```

Next, it is necessary to create an FSDP config suitable for the number of server nodes and computing cards of one's own:
```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'yes'
fsdp_config:
  # changed from SIZE_BASED_WRAP to TRANSFORMER_BASED_WRAP
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: false
  # changed here
  fsdp_transformer_layer_cls_to_wrap: BasicTransformerBlock,ResnetBlock2D
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

In the implementation of this project, in addition to the original DMD2, there are [Decoupled DMD](https://arxiv.org/pdf/2511.22677) and [f-divergence (One-step Diffusion Models with f-Divergence Distribution Matching)](https://arxiv.org/pdf).

For specific implementation details, please refer to [decoupled_guidance.py](main/decoupled_guidance.py) and [sd_guidance.py](main/sd_guidance.py)

To specify a particular training algorithm, it is necessary to control the calculation method of DMD Loss (whether to decouple or not, using the Reverse-KL or the general f-divergence), whether GAN Loss is involved, and more parameter details through a .sh file:
```bash
#!/bin/bash

# Change to project directory
cd /home/zhijun/Code/DMDs

# Set environment variables
export CHECKPOINT_PATH=/data/sdv15
export WANDB_ENTITY=aaroncaozj_team
export WANDB_PROJECT=sdv15_decoupled_dmd
export PYTHONPATH=/home/zhijun/Code/DMDs:$PYTHONPATH

# HuggingFace model cache location
export HF_HOME=/home/zhijun/Code/DMDs/ckpt
export HF_HUB_CACHE=/home/zhijun/Code/DMDs/ckpt/hub
export TRANSFORMERS_CACHE=/home/zhijun/Code/DMDs/ckpt/transformers

accelerate launch --config_file fsdp_configs/fsdp_1node_2x5090_mix.yaml main/train_sd.py \
    --generator_lr 1e-5  \
    --guidance_lr 1e-5 \
    --train_iters 100000000 \
    --output_path  $CHECKPOINT_PATH/sdv15_decoupled_dmd \
    --cache_dir $CHECKPOINT_PATH/cache \
    --log_path $CHECKPOINT_PATH/logs \
    --batch_size 4 \
    --grid_size 2 \
    --initialize_generator --log_iters 500 \
    --resolution 512 \
    --latent_resolution 64 \
    --seed 10 \
    --real_guidance_scale 7.5 \
    --fake_guidance_scale 1.0 \
    --max_grad_norm 10.0 \
    --model_id "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --train_prompt_path $CHECKPOINT_PATH/captions_laion_score6.25.pkl \
    --real_image_path $CHECKPOINT_PATH/sd_vae_latents_laion_500k_lmdb \
    --wandb_iters 50 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_name "sdv15_ddmd_4step_lr1-5_cfg7.5_new"  \
    --wandb_project $WANDB_PROJECT \
    --use_fp16 \
    --log_loss \
    --dfake_gen_update_ratio 5 \
    --fsdp \
    --denoising \
    --num_denoising_step 4 \
    --denoising_timestep 1000 \
    --backward_simulation \
    --use_decoupled_dmd \
    --min_step_percent 0.0 \
    --max_step_percent 1.0 \
```

#### Evaluation

Two test scripts are written here, which are used to implement [50-step inference for the Teacher model](scripts/teacher_eval.sh) and [Few-step inference for the student model](scripts/simple_eval.sh) respectively. Using the same prompt and seed, you need to modify the weight path and output path in the scripts, or modify the prompt yourself to observe the effect of specific prompts:
```bash
#!/bin/bash

cd /home/zhijun/Code/DMDs

export PYTHONPATH=/home/zhijun/Code/DMDs:$PYTHONPATH
export HF_HOME=/home/zhijun/Code/DMDs/ckpt
export HF_HUB_CACHE=/home/zhijun/Code/DMDs/ckpt/hub
export TRANSFORMERS_CACHE=/home/zhijun/Code/DMDs/ckpt/transformers

# 设置checkpoint路径
CHECKPOINT_DIR="/data/sdv15/cache/time_1770521735_seed10/checkpoint_model_002000"

# 找到最新的checkpoint
LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/pytorch_model.bin 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Using checkpoint: $LATEST_CHECKPOINT"

python main/simple_inference_lcm.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --output_dir "/data/sdv15/test_output/lr1-5_ratio5_cfg3.0/dmd2_gan/dmd2_2000_newprompt" \
    --seed 42 \
    --device cuda \
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
    --compute_clip_score \
    # --clip_model ViT-G/14 \
    # --compute_image_reward
```

Some of the test results are listed in this document for reference. If you find any problems, please contact me, because I found that it seems difficult to widen the gap between DMD2/Decoupled DMD on SDv1.5, which is not very consistent with the description in the paper：
https://vgfw41e20bo.sg.larksuite.com/wiki/R9QpweimTiHKfwktAwPlcv0eg4b?from=from_copylink

More details provided by the author, please refer to [SDv1.5](experiments/sdv1.5/README.md) for details.
