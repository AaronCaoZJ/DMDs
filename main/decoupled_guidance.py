from main.sd_guidance import SDGuidance, predict_noise
from main.utils import get_x0_from_noise
import torch
import torch.nn.functional as F


class SDDecoupledGuidance(SDGuidance):
    def __init__(self, args, accelerator):
        # 调用父类的初始化方法，继承所有的属性（如 self.real_unet, self.fake_unet 等）
        super().__init__(args, accelerator)

        # new attributes for Decoupled DMD
        # self.cfg_weight = getattr(args, 'cfg_weight', 1.0)
        # self.dm_weight = getattr(args, 'dm_weight', 0.5)

        # self.new_component = ...

    def compute_decoupled_loss(
        self, 
        latents,  # 这是 generator 的输出 (generated_image)
        text_embedding, 
        uncond_embedding, 
        unet_added_conditions, 
        uncond_unet_added_conditions,
        timesteps = None  # 新增参数：generate data 构造的 timesteps
    ):
        # latents 是 generator 产生的输出，对应伪代码中的 pred_generator_image
        pred_generator_image = latents
        batch_size = latents.shape[0]
        
        with torch.no_grad():
            if timesteps is None:
                raise ValueError("timesteps must be provided for Decoupled DMD loss computation")
            
            # 随机采样 CA 和 DM 的时间步
            # 对于CA：每个样本使用其自己的timestep作为上界（CA在更早/更干净的timestep上进行）
            # 对于batch中的每个样本，采样 [min_step, timesteps[i]) 范围内的时间步
            # 注意：较小的timestep = 更干净的图像，CA应该在比generator更干净的阶段进行CFG对齐
            ca_timesteps = torch.zeros(batch_size, device=latents.device, dtype=torch.long)
            # max_timestep = min(self.max_step+1, self.num_train_timesteps)
            
            for i in range(batch_size):
                # 每个样本根据自己的 timestep 采样 CA timestep
                # CA的timestep < generator的timestep（更早的去噪阶段，更干净的图像）
                t = int(timesteps[i].item())
                if t > self.min_step:
                    ca_timesteps[i] = torch.randint(self.min_step, t, (1,), device=latents.device, dtype=torch.long)
                else:
                    ca_timesteps[i] = self.min_step

            # DM 的时间步：统一从 [min_step, max_step+1) 采样
            dm_timesteps = torch.randint(
                self.min_step,
                min(self.max_step+1, self.num_train_timesteps),
                [batch_size],
                device=latents.device,
                dtype=torch.long
            )

            # CA 和 DM 使用独立的噪声
            ca_noisy_latents = self.scheduler.add_noise(latents, torch.randn_like(latents), ca_timesteps)
            dm_noisy_latents = self.scheduler.add_noise(latents, torch.randn_like(latents), dm_timesteps)

            # DM fake cond
            pred_dm_fake_cond_noise = predict_noise(
                self.fake_unet, dm_noisy_latents, text_embedding, uncond_embedding, 
                dm_timesteps, guidance_scale=self.fake_guidance_scale,
                unet_added_conditions=unet_added_conditions,
                uncond_unet_added_conditions=uncond_unet_added_conditions,
                decoupled=False
            )
            pred_dm_fake_cond_image = get_x0_from_noise(
                dm_noisy_latents.double(), pred_dm_fake_cond_noise.double(), self.alphas_cumprod.double(), dm_timesteps
            )

            if self.use_fp16:
                if self.sdxl:
                    bf16_unet_added_conditions = {} 
                    bf16_uncond_unet_added_conditions = {} 

                    for k,v in unet_added_conditions.items():
                        bf16_unet_added_conditions[k] = v.to(torch.bfloat16)
                    for k,v in uncond_unet_added_conditions.items():
                        bf16_uncond_unet_added_conditions[k] = v.to(torch.bfloat16)
                else:
                    bf16_unet_added_conditions = unet_added_conditions 
                    bf16_uncond_unet_added_conditions = uncond_unet_added_conditions

                pred_ca_real_cond_noise, pred_ca_real_uncond_noise = predict_noise(
                    self.real_unet, ca_noisy_latents.to(torch.bfloat16), text_embedding.to(torch.bfloat16), 
                    uncond_embedding.to(torch.bfloat16), 
                    ca_timesteps, guidance_scale=self.real_guidance_scale,
                    unet_added_conditions=bf16_unet_added_conditions,
                    uncond_unet_added_conditions=bf16_uncond_unet_added_conditions,
                    decoupled=True
                )
                pred_dm_real_cond_noise, _ = predict_noise(
                    self.real_unet, dm_noisy_latents.to(torch.bfloat16), text_embedding.to(torch.bfloat16), 
                    uncond_embedding.to(torch.bfloat16), 
                    dm_timesteps, guidance_scale=self.real_guidance_scale,
                    unet_added_conditions=bf16_unet_added_conditions,
                    uncond_unet_added_conditions=bf16_uncond_unet_added_conditions,
                    decoupled=True
                )
            else:
                pred_ca_real_cond_noise, pred_ca_real_uncond_noise = predict_noise(
                    self.real_unet, ca_noisy_latents, text_embedding, uncond_embedding, 
                    ca_timesteps, guidance_scale=self.real_guidance_scale,
                    unet_added_conditions=unet_added_conditions,
                    uncond_unet_added_conditions=uncond_unet_added_conditions,
                    decoupled=True
                )
                pred_dm_real_cond_noise, _ = predict_noise(
                    self.real_unet, dm_noisy_latents, text_embedding, uncond_embedding, 
                    dm_timesteps, guidance_scale=self.real_guidance_scale,
                    unet_added_conditions=unet_added_conditions,
                    uncond_unet_added_conditions=uncond_unet_added_conditions,
                    decoupled=True
                )

            # DM real cond
            pred_dm_real_cond_image = get_x0_from_noise(
                dm_noisy_latents.double(), pred_dm_real_cond_noise.double(), self.alphas_cumprod.double(), dm_timesteps
            )
            # CA real cond and uncond
            pred_ca_real_cond_image = get_x0_from_noise(
                ca_noisy_latents.double(), pred_ca_real_cond_noise.double(), self.alphas_cumprod.double(), ca_timesteps
            )
            pred_ca_real_uncond_image = get_x0_from_noise(
                ca_noisy_latents.double(), pred_ca_real_uncond_noise.double(), self.alphas_cumprod.double(), ca_timesteps
            )

            # CA update vector
            ca_update_vector = (self.real_guidance_scale - 1) * (pred_ca_real_cond_image - pred_ca_real_uncond_image)
            ca_norm_factor = (pred_ca_real_cond_image - pred_generator_image).abs().mean(dim=[1, 2, 3], keepdim=True)  # [B, 1, 1, 1]
            ca_update_vector = ca_update_vector / (ca_norm_factor + 1e-8)
            
            # DM update vector
            dm_update_vector = (pred_dm_real_cond_image - pred_dm_fake_cond_image)
            dm_norm_factor = (pred_dm_real_cond_image - pred_generator_image).abs().mean(dim=[1, 2, 3], keepdim=True)  # [B, 1, 1, 1]
            dm_update_vector = dm_update_vector / (dm_norm_factor + 1e-8)

        # 对 pred_generator_image (即 latents) 计算损失，梯度会传递回 generator
        loss_ca = F.mse_loss(pred_generator_image.float(), (pred_generator_image + ca_update_vector).detach().float(), reduction="mean")
        loss_dm = F.mse_loss(pred_generator_image.float(), (pred_generator_image + dm_update_vector).detach().float(), reduction="mean")

        loss = loss_dm + loss_ca

        loss_dict = {
            "loss_decoupled": loss,
            "loss_ca": loss_ca,
            "loss_dm": loss_dm
        }

        decoupled_log_dict = {
            "ca_noisy_latents": ca_noisy_latents.detach().float(),
            "dm_noisy_latents": dm_noisy_latents.detach().float(),
            "pred_ca_real_cond_image": pred_ca_real_cond_image.detach().float(),
            "pred_ca_real_uncond_image": pred_ca_real_uncond_image.detach().float(),
            "pred_dm_real_cond_image": pred_dm_real_cond_image.detach().float(),
            "pred_dm_fake_cond_image": pred_dm_fake_cond_image.detach().float(),
            "ca_update_vector": ca_update_vector.detach().float(),
            "dm_update_vector": dm_update_vector.detach().float(),
            "ca_norm_factor": ca_norm_factor.detach().float().mean(),  # 记录平均值
            "dm_norm_factor": dm_norm_factor.detach().float().mean(),  # 记录平均值
            "ca_update_norm": torch.norm(ca_update_vector).item(),  # 添加更新向量的范数
            "dm_update_norm": torch.norm(dm_update_vector).item(),  # 添加更新向量的范数
            "ca_timesteps_mean": ca_timesteps.float().mean().item(),  # CA 时间步平均值
            "dm_timesteps_mean": dm_timesteps.float().mean().item(),  # DM 时间步平均值
        }

        return loss_dict, decoupled_log_dict

    def generator_forward(
        self,
        image,
        text_embedding,
        uncond_embedding,
        unet_added_conditions=None,
        uncond_unet_added_conditions=None,
        timesteps = None  # 新增参数：从 backward simulation 传入的 timesteps
    ):
        loss_dict = {}
        log_dict = {}

        # 使用 Decoupled DMD 计算 Distribution Matching Loss
        if not self.gan_alone:
            dm_dict, dm_log_dict = self.compute_decoupled_loss(
                image, text_embedding, uncond_embedding, 
                unet_added_conditions, uncond_unet_added_conditions,
                timesteps=timesteps  # 传递 timesteps
            )

            loss_dict.update(dm_dict)
            log_dict.update(dm_log_dict)

        # 如果需要，也可以添加 clean image classification loss
        if self.cls_on_clean_image:
            clean_cls_loss_dict = self.compute_generator_clean_cls_loss(
                image, text_embedding, unet_added_conditions
            )
            loss_dict.update(clean_cls_loss_dict)

        return loss_dict, log_dict

    def forward(
        self,
        generator_turn=False,
        guidance_turn=False,
        generator_data_dict=None, 
        guidance_data_dict=None
    ):    
        if generator_turn:
            # 从 generator_data_dict 中提取 timesteps
            timesteps = generator_data_dict.get("timesteps", None)
            
            loss_dict, log_dict = self.generator_forward(
                image=generator_data_dict["image"],
                text_embedding=generator_data_dict["text_embedding"],
                uncond_embedding=generator_data_dict["uncond_embedding"],
                unet_added_conditions=generator_data_dict["unet_added_conditions"],
                uncond_unet_added_conditions=generator_data_dict["uncond_unet_added_conditions"],
                timesteps=timesteps  # 传递 timesteps
            )   
        elif guidance_turn:
            # Guidance turn 使用父类的逻辑
            loss_dict, log_dict = self.guidance_forward(
                image=guidance_data_dict["image"],
                text_embedding=guidance_data_dict["text_embedding"],
                uncond_embedding=guidance_data_dict["uncond_embedding"],
                real_train_dict=guidance_data_dict["real_train_dict"],
                unet_added_conditions=guidance_data_dict["unet_added_conditions"],
                uncond_unet_added_conditions=guidance_data_dict["uncond_unet_added_conditions"]
            ) 
        else:
            raise NotImplementedError

        return loss_dict, log_dict