"""
使用原始SD1.5模型进行标准多步采样推理
用于与蒸馏模型的输出进行对比
支持多种scheduler，包括LCM用于few-step采样对比
"""
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler, LCMScheduler
from PIL import Image
import torch
import argparse
import os


def save_images(images, prompts, output_dir):
    """保存生成的图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (image, prompt) in enumerate(zip(images, prompts)):
        # 创建安全的文件名
        safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)
        safe_prompt = safe_prompt[:50]
        
        filename = f"{idx:04d}_{safe_prompt}.png"
        filepath = os.path.join(output_dir, filename)
        
        image.save(filepath)
        print(f"Saved: {filepath}")
    
    print(f"\nAll images saved to {output_dir}")


@torch.no_grad()
def generate_images_teacher(pipeline, prompts, args):
    """使用teacher模型生成图像"""
    all_images = []
    
    print(f"使用 {args.num_inference_steps} 步采样")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Scheduler: {args.scheduler}")
    
    # 如果使用LCM且设置了固定timesteps，使用之
    if args.scheduler == "lcm" and args.lcm_timesteps:
        print(f"LCM固定timesteps: {args.lcm_timesteps}")
    
    for idx, prompt in enumerate(prompts):
        print(f"Generating image {idx+1}/{len(prompts)}: {prompt}")
        
        # 为每张图片设置固定种子（与蒸馏模型保持一致）
        generator = torch.Generator(device=args.device).manual_seed(args.seed + idx)
        
        # 使用标准pipeline生成图像
        if args.scheduler == "lcm" and args.lcm_timesteps:
            # LCM with fixed timesteps (like DMD2)
            image = pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                height=args.image_resolution,
                width=args.image_resolution,
                timesteps=args.lcm_timesteps,
            ).images[0]
        else:
            # Standard sampling
            image = pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                height=args.image_resolution,
                width=args.image_resolution,
            ).images[0]
        
        all_images.append(image)
    
    return all_images


def main():
    parser = argparse.ArgumentParser(description="Teacher model inference for SD1.5 with standard sampling")
    parser.add_argument("--output_dir", type=str, default="./teacher_outputs",
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
    parser.add_argument("--image_resolution", type=int, default=512,
                        help="Image resolution")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                        help="HuggingFace model ID")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps for teacher model")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for classifier-free guidance")
    parser.add_argument("--scheduler", type=str, default="ddim",
                        choices=["ddim", "ddpm", "euler", "lcm"],
                        help="Scheduler type (use 'lcm' only for few-step sampling like 4-8 steps)")
    parser.add_argument("--lcm_timesteps", type=int, nargs="+", default=None,
                        help="Fixed timesteps for LCM (e.g., 999 749 499 249). Only used with --scheduler lcm")
    
    args = parser.parse_args()
    
    # 设置全局随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 80)
    print("Teacher Model Inference for SD1.5 - Standard Multi-Step Sampling")
    print("=" * 80)
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Number of images: {len(args.prompts)}")
    print(f"Random seed: {args.seed}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Scheduler: {args.scheduler}")
    if args.scheduler == "lcm" and args.lcm_timesteps:
        print(f"LCM timesteps: {args.lcm_timesteps}")
    print(f"Image resolution: {args.image_resolution}x{args.image_resolution}")
    print("=" * 80)
    
    # 给出配置建议
    print("\n💡 Configuration Suggestions:")
    print("  Standard quality (50 steps):   --scheduler ddim --num_inference_steps 50 --guidance_scale 7.5")
    print("  Fast quality (25 steps):       --scheduler euler --num_inference_steps 25 --guidance_scale 7.5")
    print("  Few-step comparison (4 steps): --scheduler lcm --num_inference_steps 4 --lcm_timesteps 999 749 499 249 --guidance_scale 0")
    print("=" * 80)
    
    # 加载模型和pipeline
    print("\nLoading teacher model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # 验证scheduler和步数的合理性
    if args.scheduler == "lcm" and args.num_inference_steps > 10:
        print("\n" + "="*80)
        print("⚠️  WARNING: LCM scheduler is designed for few-step sampling (4-8 steps)")
        print(f"   You are using {args.num_inference_steps} steps, which may produce suboptimal results.")
        print("   Recommendation: Use --scheduler ddim or --scheduler euler for >10 steps")
        print("="*80 + "\n")
    
    if args.scheduler != "lcm" and args.lcm_timesteps:
        print("\n" + "="*80)
        print("⚠️  WARNING: --lcm_timesteps is only used with --scheduler lcm")
        print("   Your timesteps setting will be ignored.")
        print("="*80 + "\n")
    
    # 设置scheduler
    if args.scheduler == "ddim":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        print("Using DDIM scheduler (good for 20-50 steps)")
    elif args.scheduler == "ddpm":
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        print("Using DDPM scheduler (good for 50-1000 steps)")
    elif args.scheduler == "euler":
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        print("Using Euler scheduler (good for 20-50 steps)")
    elif args.scheduler == "lcm":
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        print("Using LCM scheduler (optimized for 4-8 steps)")
        if args.lcm_timesteps:
            print(f"  → Fixed timesteps: {args.lcm_timesteps}")
        else:
            print(f"  → Auto timesteps for {args.num_inference_steps} steps")
    
    pipeline = pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=False)
    
    # 生成图像
    print("\nGenerating images with teacher model...")
    images = generate_images_teacher(pipeline, args.prompts, args)
    
    # 保存图像
    print("\nSaving images...")
    save_images(images, args.prompts, args.output_dir)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
