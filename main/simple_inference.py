"""
简单的推理脚本，用于测试训练好的 SDv1.5 模型
"""
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import torch
import argparse
import os


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


def get_x0_from_noise(x_t, noise, alphas_cumprod, t):
    """从噪声预测恢复x0"""
    alpha_t = alphas_cumprod[t].reshape(-1, 1, 1, 1)
    x0 = (x_t - (1 - alpha_t).sqrt() * noise) / alpha_t.sqrt()
    return x0


@torch.no_grad()
def generate_images(generator, vae, text_encoder, tokenizer, prompts, args, noise_scheduler):
    """生成图像 - 使用4步backward simulation
    
    这个实现与训练代码中的sample_backward方法保持完全一致：
    1. 从纯噪声开始
    2. 迭代4步，每步预测噪声并恢复x0
    3. 在每步之间添加噪声到下一个时间步
    """
    generator.eval()
    all_images = []
    
    # 设置4步去噪的时间步（与训练配置完全一致）
    # denoising_timestep = 1000, num_denoising_step = 4
    # denoising_step_list = range(999, 0, -250) = [999, 749, 499, 249]
    num_denoising_step = args.num_denoising_step  # 4
    denoising_timestep = args.denoising_timestep  # 1000
    timestep_interval = denoising_timestep // num_denoising_step  # 250
    
    denoising_step_list = list(range(denoising_timestep-1, 0, -timestep_interval))
    print(f"使用4步推理，时间步列表: {denoising_step_list}")
    print(f"时间步间隔: {timestep_interval}")
    
    # 获取alphas_cumprod用于正确的去噪计算
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(args.device)
    
    for idx, prompt in enumerate(prompts):
        print(f"Generating image {idx+1}/{len(prompts)}: {prompt}")
        
        # 为每张图片设置固定种子（基于全局种子 + 图片索引）
        torch.manual_seed(args.seed + idx)
        torch.cuda.manual_seed(args.seed + idx)
        
        # 编码文本
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(args.device)
        text_embedding = text_encoder(text_input_ids)[0]
        
        # 生成初始随机噪声（从t=999开始）
        noisy_image = torch.randn(
            1, 4, args.latent_resolution, args.latent_resolution,
            dtype=torch.float32,
            device=args.device
        )
        
        # 4步迭代去噪过程（与训练时的sample_backward完全一致）
        # 遍历所有4个时间步: [999, 749, 499, 249]
        for step_idx, timestep_value in enumerate(denoising_step_list):
            current_timesteps = torch.ones(1, device=args.device, dtype=torch.long) * timestep_value
            
            # 使用生成器预测噪声（epsilon prediction）
            predicted_noise = generator(
                noisy_image, current_timesteps, text_embedding
            ).sample
            
            # 从噪声预测中恢复x0（干净图像）
            # 使用与训练代码相同的get_x0_from_noise函数
            generated_image = get_x0_from_noise(
                noisy_image, predicted_noise.double(), alphas_cumprod.double(), current_timesteps
            ).float()
            
            # 如果不是最后一步，添加噪声到下一个时间步
            if step_idx < len(denoising_step_list) - 1:
                # 计算下一个时间步：当前时间步 - 间隔
                # 999 -> 749, 749 -> 499, 499 -> 249
                next_timestep = current_timesteps - timestep_interval
                
                # 使用noise_scheduler添加噪声（与训练代码一致）
                # 这会根据noise schedule添加适当水平的噪声
                noisy_image = noise_scheduler.add_noise(
                    generated_image, 
                    torch.randn_like(generated_image), 
                    next_timestep
                ).to(noisy_image.dtype)
            else:
                # 最后一步（t=249），直接使用生成的干净图像
                noisy_image = generated_image
        
        # 最终的noisy_image就是生成的干净latent
        # 解码latents到图像空间
        generated_latents = noisy_image / vae.config.scaling_factor  # 使用VAE的scaling_factor
        images = vae.decode(generated_latents).sample
        
        # 转换到[0, 255]
        images = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        
        # 转换为PIL图像
        pil_image = Image.fromarray(images[0])
        all_images.append(pil_image)
    
    return all_images


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


def main():
    parser = argparse.ArgumentParser(description="Simple inference for trained SDv1.5 model with 4-step backward simulation")
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
    # 4步推理的关键参数（与训练配置一致）
    parser.add_argument("--num_denoising_step", type=int, default=4,
                        help="Number of denoising steps (default: 4)")
    parser.add_argument("--denoising_timestep", type=int, default=1000,
                        help="Total timesteps for denoising (default: 1000)")
    
    args = parser.parse_args()
    
    # 设置全局随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 80)
    print("Simple Inference for SDv1.5 - 4-Step Backward Simulation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Number of images: {len(args.prompts)}")
    print(f"Random seed: {args.seed}")
    print(f"Denoising steps: {args.num_denoising_step}")
    print(f"Denoising timestep: {args.denoising_timestep}")
    print(f"Timestep interval: {args.denoising_timestep // args.num_denoising_step}")
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

    # 加载 noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )
    
    # 生成图像
    print("\nGenerating images...")
    images = generate_images(generator, vae, text_encoder, tokenizer, args.prompts, args, noise_scheduler)
    
    # 保存图像
    print("\nSaving images...")
    save_images(images, args.prompts, args.output_dir)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
