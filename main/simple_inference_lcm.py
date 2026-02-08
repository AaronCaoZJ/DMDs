"""
简单的推理脚本，用于测试训练好的 SDv1.5 模型
支持CLIP score和ImageReward评估
使用LCMScheduler进行高效的few-step推理
"""
from diffusers import UNet2DConditionModel, AutoencoderKL, LCMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import torch
import argparse
import os
import numpy as np
from main.coco_eval.coco_evaluator import compute_clip_score, compute_image_reward


def load_generator(checkpoint_path):
    """加载生成器模型"""
    print(f"Loading generator from {checkpoint_path}")
    
    # 加载基础模型结构
    generator = UNet2DConditionModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        subfolder="unet"
    ).float()
    
    # 加载训练好的权重
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    generator.load_state_dict(state_dict, strict=True)
    generator.requires_grad_(False)
    
    return generator


@torch.no_grad()
def generate_images(generator, vae, text_encoder, tokenizer, prompts, args, noise_scheduler):
    generator.eval()
    all_images = []
    clip_scores = []
    
    # 固定timesteps
    target_timesteps = [999, 749, 499, 249]
    print(f"使用LCMScheduler，固定时间步: {target_timesteps}")
    
    for idx, prompt in enumerate(prompts):
        print(f"Generating image {idx+1}/{len(prompts)}: {prompt}")
        
        torch.manual_seed(args.seed + idx)
        torch.cuda.manual_seed(args.seed + idx)
        
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(args.device)
        text_embedding = text_encoder(text_input_ids)[0]
        
        # 生成初始随机噪声
        latents = torch.randn(
            1, 4, args.latent_resolution, args.latent_resolution,
            dtype=generator.dtype,  # 跟随模型精度
            device=args.device
        )
        
        # ⚠️ 关键：为每张图重新设置scheduler状态
        # LCMScheduler有内部状态，必须在每次生成前重置
        noise_scheduler.set_timesteps(num_inference_steps=4)
        noise_scheduler.timesteps = torch.tensor(target_timesteps, device=args.device, dtype=torch.long)
        
        # 4步去噪过程
        for t in noise_scheduler.timesteps:
            # 扩展 latents 输入 (LCM 不需要 scale_model_input，但标准流程通常会加)
            # latents_input = noise_scheduler.scale_model_input(latents, t) 
            # 对于纯 LCM scheduler 通常 latents_input = latents，这行可以省略
            
            noise_pred = generator(
                latents, t, text_embedding
            ).sample
            
            latents = noise_scheduler.step(
                noise_pred, 
                t, 
                latents,
                return_dict=False
            )[0]
        
        # 解码
        latents = latents / vae.config.scaling_factor
        images = vae.decode(latents).sample
        
        images = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        
        pil_image = Image.fromarray(images[0])
        all_images.append(pil_image)
        
        # 如果启用了CLIP评分，立即计算这张图的分数
        if args.compute_clip_score:
            try:
                image_np = images[0]
                clip_score = compute_clip_score(
                    images=[image_np],
                    captions=[prompt],
                    clip_model=args.clip_model,
                    device=args.device,
                    how_many=1  # 只计算1张图
                )
                clip_scores.append(float(clip_score))
                
                print(f"  → CLIP Score: {clip_score:.4f}")
            except Exception as e:
                import traceback
                print(f"  → CLIP Score: Failed ({str(e)})")
                traceback.print_exc()
                clip_scores.append(None)
    
    # 返回图像和分数
    if clip_scores:
        return all_images, clip_scores
    else:
        return all_images, None


def save_images(images, prompts, output_dir, metrics=None):
    """保存生成的图像和评估指标"""
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (image, prompt) in enumerate(zip(images, prompts)):
        # 创建安全的文件名
        safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)
        safe_prompt = safe_prompt[:50]
        
        filename = f"{idx:04d}_{safe_prompt}.png"
        filepath = os.path.join(output_dir, filename)
        
        image.save(filepath)
        print(f"Saved: {filepath}")
    
    # 保存评估指标
    if metrics:
        import json
        metrics_file = os.path.join(output_dir, "metrics.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_file}")
    
    print(f"\nAll images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Simple inference for trained SDv1.5 model with LCMScheduler")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to checkpoint (pytorch_model.bin)")
    parser.add_argument("--output_dir", type=str, default="./inference_outputs",
                        help="Output directory")
    parser.add_argument("--prompts", type=str, nargs="+",
                        default=[
                            "a beautiful landscape with mountains",
                            "a cute cat sitting on a table",
                            "a futuristic city at night",
                            "a portrait of a woman",
                            "a cup of coffee on a wooden table"
                        ],
                        help="Prompts for generation")
    parser.add_argument("--latent_resolution", type=int, default=64,
                        help="Latent resolution (64 for 512x512 images)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--print_model", action="store_true",
                        help="Whether to print model structure and parameter counts")
    
    # 评估指标参数
    parser.add_argument("--compute_clip_score", action="store_true",
                        help="Compute CLIP score using ViT-B/32")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-G/14"],
                        help="CLIP model to use for scoring")
    
    args = parser.parse_args()
    
    # 设置全局随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 80)
    print("Simple Inference for SDv1.5 - LCMScheduler (4-Step)")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Number of images: {len(args.prompts)}")
    print(f"Random seed: {args.seed}")
    print(f"Using LCMScheduler with fixed timesteps: [999, 749, 499, 249]")
    print("=" * 80)
    
    # 加载模型组件
    print("\nLoading models...")
    generator = load_generator(args.checkpoint).to(args.device)
    # 打印模型结构
    if args.print_model:
        print("\n" + "="*80)
        print("MODEL STRUCTURE")
        print("="*80)
        print(generator)
        
        print("\n" + "="*80)
        print("MODEL PARAMETER COUNT")
        print("="*80)
        total_params = sum(p.numel() for p in generator.parameters())
        trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"Frozen parameters: {total_params - trainable_params:,} ({(total_params - trainable_params)/1e6:.2f}M)")
        
        print("\n" + "="*80)
        print("PARAMETER BREAKDOWN BY MODULE")
        print("="*80)
        for name, module in generator.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name:30s}: {module_params:12,} ({module_params/1e6:8.2f}M) | Trainable: {module_trainable:12,} ({module_trainable/1e6:8.2f}M)")
        print("="*80 + "\n")
    
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae"
    ).to(args.device).float()
    vae.eval()
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder"
    ).to(args.device).float()
    text_encoder.eval()
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id, subfolder="tokenizer"
    )

    # 加载 LCM scheduler (官方推荐)
    noise_scheduler = LCMScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )

    # 加载CLIP模型用于实时评分（如果需要）
    if args.compute_clip_score:
        print("\nLoading CLIP model for real-time scoring...")
        # coco_evaluator中的compute_clip_score会自动加载CLIP模型
        pass
    
    # 生成图像（并在生成时计算CLIP score）
    print("\nGenerating images...")
    images, clip_scores = generate_images(
        generator, vae, text_encoder, tokenizer, args.prompts, args, noise_scheduler
    )
    
    # 保存图像
    print("\nSaving images...")
    
    # 汇总评估指标
    metrics = {}
    
    if clip_scores is not None:
        valid_scores = [s for s in clip_scores if s is not None]
        if valid_scores:
            metrics['clip_score_mean'] = float(np.mean(valid_scores))
            metrics['clip_score_std'] = float(np.std(valid_scores))
            metrics['clip_score_per_image'] = clip_scores
            
            print("\n" + "=" * 80)
            print("CLIP Score Summary:")
            print(f"  Mean: {metrics['clip_score_mean']:.4f}")
            print(f"  Std:  {metrics['clip_score_std']:.4f}")
            print("=" * 80)
    
    save_images(images, args.prompts, args.output_dir, metrics=metrics if metrics else None)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
