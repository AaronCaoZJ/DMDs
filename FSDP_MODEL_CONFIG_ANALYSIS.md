# DMD2 项目 FSDP 与模型配置详细分析

## 一、FSDP 核心配置

### 1.1 FSDP 配置文件
**路径**: `fsdp_configs/fsdp_1node_2x5090_mix.yaml`

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16                    # 混合精度训练：BFloat16
downcast_bf16: 'yes'                    # 自动向下转换为bf16

fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP          # 基于Transformer层的包装策略
  fsdp_transformer_layer_cls_to_wrap: BasicTransformerBlock,ResnetBlock2D  # 包装的层类型
  fsdp_backward_prefetch_policy: BACKWARD_PRE            # 反向传播预取策略
  fsdp_forward_prefetch: false                           # 前向传播不预取
  fsdp_offload_params: false                             # 不卸载参数到CPU
  fsdp_sharding_strategy: 1                              # HYBRID_SHARD (混合分片)
  fsdp_state_dict_type: SHARDED_STATE_DICT              # 分片状态字典
  fsdp_sync_module_states: true                          # 同步模块状态
  fsdp_use_orig_params: true                             # 使用原始参数
```

### 1.2 训练脚本中的 FSDP 设置
**路径**: `main/train_sd.py`

```python
# 第33行：Accelerator 初始化
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision="bf16" if args.use_fp16 else "no",  # 混合精度
    log_with="wandb",
    project_config=accelerator_project_config,
    kwargs_handlers=None
)
```

**FSDP 模型包装逻辑 (第184-230行)**:
```python
if self.fsdp:
    # 仅包装 feedforward_model 和 guidance_model，self.model 本身不包装
    self.model.feedforward_model, self.model.guidance_model = accelerator.prepare(
        self.model.feedforward_model, self.model.guidance_model
    )
```

---

## 二、三个核心模型详细分析

### 2.1 Generator Model (feedforward_model / UNet)

#### 2.1.1 模型加载路径
**文件**: `main/sd_unified_model.py` (第44-83行)

```python
if args.initialize_generator:
    self.feedforward_model = UNet2DConditionModel.from_pretrained(
        args.model_id,  # "stable-diffusion-v1-5/stable-diffusion-v1-5"
        subfolder="unet"
    ).float()
```

**完整路径**: 
- HuggingFace Hub: `stable-diffusion-v1-5/stable-diffusion-v1-5/unet`
- 本地缓存: `/home/zhijun/Code/DMD2/ckpt/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/.../unet/`

#### 2.1.2 冻结状态
```python
# 如果使用 LoRA (args.generator_lora=True):
self.feedforward_model.requires_grad_(False)  # 冻结基础模型
# 然后通过 add_adapter 添加可训练的 LoRA 层

# 如果不使用 LoRA:
self.feedforward_model.requires_grad_(True)   # 全参数训练
```

**当前配置**: 全参数训练（未使用 LoRA）

#### 2.1.3 精度变换链路

| 阶段 | 精度 | 代码位置 | 说明 |
|------|------|----------|------|
| **初始加载** | FP32 | `.from_pretrained().float()` | 从 HuggingFace 加载后转为 FP32 |
| **FSDP前转换** | BF16 | `sd_unified_model.py:83` | `if args.use_fp16 and args.fsdp: self.feedforward_model.to(torch.bfloat16)` |
| **FSDP包装** | BF16 | `train_sd.py:186` | FSDP 包装时保持 BF16 |
| **前向计算** | BF16 | `sd_unified_model.py:121` | `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` |
| **损失计算** | FP32 | 多处 `.float()` | 损失计算前转回 FP32 以提高精度 |

**关键代码**:
```python
# sd_unified_model.py:82-84
if args.use_fp16 and getattr(args, 'fsdp', False):
    self.feedforward_model.to(torch.bfloat16)
    self.guidance_model.to(torch.bfloat16)

# sd_unified_model.py:121
self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_fp16 else NoOpContext()
```

#### 2.1.4 梯度检查点
```python
if self.gradient_checkpointing:
    self.feedforward_model.enable_gradient_checkpointing()
```

---

### 2.2 Guidance Model (SDDecoupledGuidance)

#### 2.2.1 包含的子模型

**文件**: `main/sd_guidance.py` + `main/decoupled_guidance.py`

**2.2.1.1 Real UNet (Teacher Model)**
```python
# sd_guidance.py:58-63
self.real_unet = UNet2DConditionModel.from_pretrained(
    args.model_id,  # "stable-diffusion-v1-5/stable-diffusion-v1-5"
    subfolder="unet"
).float()

self.real_unet.requires_grad_(False)  # ✅ 冻结，仅用于推理
```

**路径**: 同 Generator，从相同预训练模型加载

**2.2.1.2 Fake UNet (Student Model)**
```python
# sd_guidance.py:66-71
self.fake_unet = UNet2DConditionModel.from_pretrained(
    args.model_id,  # 同上
    subfolder="unet"
).float()

self.fake_unet.requires_grad_(True)  # ✅ 可训练
```

**2.2.1.3 Dummy Network**
```python
# sd_guidance.py:74-75
self.dummy_network = DummyNetwork()  # 单层 Linear(10, 10)
self.dummy_network.requires_grad_(False)
```
**作用**: FSDP 要求至少一个网络有 dense parameters，此网络仅用于满足 FSDP 需求

#### 2.2.2 冻结状态总结
| 子模型 | 是否冻结 | 作用 |
|--------|----------|------|
| **real_unet** | ✅ 冻结 | Teacher model，提供真实分布的梯度信号 |
| **fake_unet** | ❌ 可训练 | Student model，学习匹配真实分布 |
| **dummy_network** | ✅ 冻结 | FSDP 技术需求 |
| **cls_pred_branch** (如启用) | ❌ 可训练 | 分类头 |

#### 2.2.3 精度变换链路

| 子模型 | 初始精度 | FSDP前转换 | 推理时精度 | 代码位置 |
|--------|----------|------------|------------|----------|
| **real_unet** | FP32 | → BF16 | BF16 | `sd_guidance.py:79-80` |
| **fake_unet** | FP32 | → BF16 | BF16 | `sd_unified_model.py:84` |
| **dummy_network** | FP32 | 保持 FP32 | FP32 | 无转换 |

**关键精度转换代码**:
```python
# sd_guidance.py:79-80
if args.use_fp16:
    self.real_unet = self.real_unet.to(torch.bfloat16)

# decoupled_guidance.py:87-96 (推理时临时转换)
if self.use_fp16:
    # 将输入和条件转为 BF16
    pred_ca_real_cond_noise, pred_ca_real_uncond_noise = predict_noise(
        self.real_unet, 
        ca_noisy_latents.to(torch.bfloat16), 
        text_embedding.to(torch.bfloat16), 
        uncond_embedding.to(torch.bfloat16),
        ca_timesteps, 
        ...
    )
```

**decoupled_guidance.py 中的损失计算**:
```python
# 第148-149行: 最终损失计算转为 FP32
loss_ca = F.mse_loss(pred_generator_image.float(), ...)
loss_dm = F.mse_loss(pred_generator_image.float(), ...)
```

---

### 2.3 辅助模型

#### 2.3.1 Text Encoder

**文件**: `main/sd_unified_model.py` (第91-96行)

```python
if self.sdxl:
    self.text_encoder = SDXLTextEncoder(args, accelerator).to(accelerator.device)
else:
    self.text_encoder = CLIPTextModel.from_pretrained(
        args.model_id,  # "stable-diffusion-v1-5/stable-diffusion-v1-5"
        subfolder="text_encoder"
    ).to(accelerator.device)

self.text_encoder.requires_grad_(False)  # ✅ 冻结
```

**路径**: 
- HuggingFace Hub: `stable-diffusion-v1-5/stable-diffusion-v1-5/text_encoder`
- 本地缓存: `/home/zhijun/Code/DMD2/ckpt/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/.../text_encoder/`

**精度**: 保持原始精度（FP32），未转换为 BF16

#### 2.3.2 VAE (AutoencoderKL)

**文件**: `main/sd_unified_model.py` (第103-119行)

```python
if args.tiny_vae:
    # SDXL 使用 tiny VAE
    self.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesdxl", torch_dtype=torch.float32
    ).float().to(accelerator.device)
else:
    self.vae = AutoencoderKL.from_pretrained(
        args.model_id,  # "stable-diffusion-v1-5/stable-diffusion-v1-5"
        subfolder="vae"
    ).float().to(accelerator.device)

self.vae.requires_grad_(False)  # ✅ 冻结

# 精度转换 (仅对非 SDXL 的原始 VAE)
if self.use_fp16 and self.not_sdxl_vae:
    self.vae.to(torch.float16)  # SDv1.5 VAE 可用 FP16
```

**路径**: 
- HuggingFace Hub: `stable-diffusion-v1-5/stable-diffusion-v1-5/vae`
- 本地缓存: `/home/zhijun/Code/DMD2/ckpt/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/.../vae/`

**精度**:
- SDv1.5: FP16 (因为 `use_fp16=True` 且 `not_sdxl_vae=True`)
- SDXL: FP32 (SDXL VAE 不支持半精度)

---

## 三、FSDP 工作流程

### 3.1 模型初始化与同步

**问题**: FSDP HYBRID_SHARD 模式下，不同节点的随机初始化参数可能不一致

**解决方案** (train_sd.py:163-180):
```python
if self.fsdp and (args.ckpt_only_path is None):
    # 1. 主进程保存当前参数
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_path, f"checkpoint_model_{self.step:06d}"), exist_ok=True)
        torch.save(self.model.feedforward_model.state_dict(), generator_path)
        torch.save(self.model.guidance_model.state_dict(), guidance_path)
    
    # 2. 所有进程等待
    accelerator.wait_for_everyone()
    
    # 3. 所有进程从磁盘加载相同参数
    self.model.feedforward_model.load_state_dict(torch.load(generator_path, map_location="cpu"), strict=True)
    self.model.guidance_model.load_state_dict(torch.load(guidance_path, map_location="cpu"), strict=True)
```

### 3.2 FSDP 包装

```python
if self.fsdp:
    # 只包装两个子网络，而不包装 self.model
    self.model.feedforward_model, self.model.guidance_model = accelerator.prepare(
        self.model.feedforward_model, self.model.guidance_model
    )
```

**包装策略**:
- `TRANSFORMER_BASED_WRAP`: 按 `BasicTransformerBlock` 和 `ResnetBlock2D` 为单位进行 sharding
- `HYBRID_SHARD (strategy=1)`: 第一个维度用 FSDP，其余维度用 DDP

### 3.3 优化器准备

```python
if self.fsdp:
    # FSDP 模式：仅准备优化器和调度器
    (self.optimizer_generator, self.optimizer_guidance, 
     self.scheduler_generator, self.scheduler_guidance) = accelerator.prepare(...)
else:
    # DDP 模式：同时准备模型、优化器和调度器
    (self.model.feedforward_model, self.model.guidance_model, 
     self.optimizer_generator, self.optimizer_guidance, 
     self.scheduler_generator, self.scheduler_guidance) = accelerator.prepare(...)
```

### 3.4 检查点保存

**FSDP 模式** (train_sd.py:254-291):
```python
def fsdp_state_dict(self, model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy):
        state_dict = model.state_dict()
    return state_dict

# 保存时
if self.fsdp:
    feedforward_state_dict = self.fsdp_state_dict(self.model.feedforward_model)
    guidance_model_state_dict = self.fsdp_state_dict(self.model.guidance_model)
    
    if accelerator.is_main_process:
        torch.save(feedforward_state_dict, generator_path)
        torch.save(guidance_model_state_dict, guidance_path)
```

**Accelerator在这里是分布式训练的"中央协调者"和"抽象层"。它承担三大核心职责：**

1. 分布式策略管理 - 通过FSDP配置文件自动选择ZeRO-3分片策略，将UNet和Guidance两个大模型的参数、梯度、优化器状态分片到8张GPU上，使单卡无法训练的模型成为可能。

2. 训练流程自动化 - 用prepare()包装模型、优化器、数据加载器后，自动处理分布式采样、梯度同步、参数收集/释放等底层细节；用backward()替代原生反向传播，自动管理梯度累积和混合精度；用gather()收集多卡指标用于日志记录。

3. 进程协调控制 - 通过is_main_process确保文件IO、WandB日志、checkpoint保存只执行一次；通过wait_for_everyone()在关键节点（如初始化同步、保存检查点）实现进程同步，避免竞态条件。

### 3.5 FSDP 参数选择

#### Sharding Strategy（分片策略）
这是最核心的配置，决定参数、梯度、优化器状态如何分片
```yaml
fsdp_sharding_strategy: 0 - 4
```
1. `0 = NO_SHARD`，不分片，等同于DDP

2. `1 = FULL_SHARD`，完全分片，参数、梯度、优化器状态都分片，最省显存，ZeRO-3等价

3. `2 = SHARD_GRAD_OP`，只分片梯度和优化器状态，参数完整保留在每个GPU，ZeRO-2等价

4. `3 = HYBRID_SHARD`，混合分片（多节点训练），节点内FULL_SHARD，节点间复制，适合跨节点带宽较低的场景

5. `4 = HYBRID_SHARD_ZERO2`，混合分片的ZeRO-2版本

**ZeRO**
Zero Redundancy Optimizer，是微软DeepSpeed提出的分布式训练内存优化技术
* ZeRO-1 仅对优化器分片
* ZeRO-2 在 ZeRO-1 的基础上，梯度也分片
* ZeRO-3 完全分片

#### Auto Wrap Policy（自动包裹策略）
```yaml
fsdp_auto_wrap_policy: SIZE_BASED_WRAP / TRANSFORMER_BASED_WRAP / NO_WRAP (手动)
fsdp_min_num_params: 50000000           # 50M参数阈值
```

#### Backward Prefetch Policy（反向预取）
可以在反向传播过程中，后一层计算时或计算后释放前，提前预取上一层的数据，弥补通信延迟
```yaml
fsdp_backward_prefetch_policy: BACKWARD_PRE / BACKWARD_POST / NO_PREFETCH
```

#### State Dict Type（状态字典保存方式）
```yaml
fsdp_state_dict_type: SHARDED_STATE_DICT / FULL_STATE_DICT / LOCAL_STATE_DICT
```

---

## 四、警告信息分析与优化建议

### 4.1 当前警告

#### 警告 1: FSDP Upcast Low Precision Parameters
```
UserWarning: Upcasted low precision parameters in UNet2DConditionModel because 
mixed precision turned on in FSDP.
```

**原因**: 
- FSDP mixed_precision=bf16 会自动将某些层（如 norm, conv）的 bf16 参数上转为 fp32
- 这是 FSDP 的保护机制，防止数值不稳定

**影响**:
1. ✅ **内存增加**: 上转的参数占用更多显存
2. ⚠️ **检查点精度不一致**: checkpoint 可能包含 fp32 参数
3. ⚠️ **性能轻微下降**: 类型转换开销

**当前受影响的层**:
- UNet: `conv_in`, `conv_out`, `conv_norm_out`, `time_embedding`, 所有 attention 的 `norm/proj_in/proj_out`
- ResNet: `norm1`, `norm2`, `conv1`, `conv2`, `conv_shortcut`

#### 警告 2: Deprecated API
```python
# scheduler.num_train_timesteps (直接访问)
deprecate("direct config name access", "1.0.0", ...)
```

**建议修改**:
```python
# 将所有 scheduler.num_train_timesteps 改为:
scheduler.config.num_train_timesteps
```

#### 警告 3: Deprecated FSDP API
```python
with FSDP.state_dict_type(...):  # 已过时
```

**建议修改**:
```python
from torch.distributed.checkpoint import state_dict

# 新 API
state_dict.get_state_dict(
    model, 
    options=StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )
)
```

### 4.2 优化建议

#### 优化 1: 显式控制混合精度策略

**在 sd_unified_model.py 添加**:
```python
from torch.distributed.fsdp import MixedPrecision

# 第82行附近添加
if args.use_fp16 and getattr(args, 'fsdp', False):
    # 自定义混合精度策略
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        # 可选：指定哪些模块保持 fp32
        # cast_forward_inputs=True
    )
    
    self.feedforward_model.to(torch.bfloat16)
    self.guidance_model.to(torch.bfloat16)
```

**然后在 train_sd.py 的 accelerator.prepare() 时传入策略**

#### 优化 2: 统一精度管理

**创建精度管理器** (`main/precision_manager.py`):
```python
class PrecisionManager:
    def __init__(self, use_fp16, use_fsdp):
        self.compute_dtype = torch.bfloat16 if use_fp16 else torch.float32
        self.model_dtype = torch.bfloat16 if (use_fp16 and use_fsdp) else torch.float32
        self.loss_dtype = torch.float32  # 损失计算始终用 FP32
    
    def cast_model(self, model):
        return model.to(self.model_dtype)
    
    def cast_for_compute(self, tensor):
        return tensor.to(self.compute_dtype)
    
    def cast_for_loss(self, tensor):
        return tensor.to(self.loss_dtype)
```

#### 优化 3: 检查点精度验证

**在保存检查点后添加验证**:
```python
def validate_checkpoint_dtype(checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    dtype_counts = {}
    for k, v in state_dict.items():
        dtype = str(v.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    
    print(f"Checkpoint dtype distribution: {dtype_counts}")
    # 预期: {'torch.bfloat16': xxx, 'torch.float32': yyy}
```

---

## 五、配置总览表

### 5.1 模型总览

| 模型 | 加载路径 | 冻结 | 初始精度 | FSDP前精度 | FSDP包装 | 备注 |
|------|----------|------|----------|------------|----------|------|
| **Generator (feedforward_model)** | HF: sdv1.5/unet | ❌ | FP32 | BF16 | ✅ | 全参数训练 |
| **Real UNet** | HF: sdv1.5/unet | ✅ | FP32 | BF16 | ✅ (作为guidance_model一部分) | Teacher |
| **Fake UNet** | HF: sdv1.5/unet | ❌ | FP32 | BF16 | ✅ (作为guidance_model一部分) | Student |
| **Dummy Network** | 随机初始化 | ✅ | FP32 | FP32 | ✅ (作为guidance_model一部分) | FSDP技术需求 |
| **Text Encoder** | HF: sdv1.5/text_encoder | ✅ | FP32 | FP32 | ❌ | 推理only |
| **VAE** | HF: sdv1.5/vae | ✅ | FP32 | FP16 | ❌ | SDv1.5用FP16 |

### 5.2 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **mixed_precision** | bf16 | FSDP混合精度 |
| **fsdp_sharding_strategy** | 1 (HYBRID_SHARD) | 混合分片 |
| **batch_size** | 4 | 每GPU批次大小 |
| **generator_lr** | 1e-5 | Generator学习率 |
| **guidance_lr** | 5e-6 | Guidance学习率 (Fake UNet) |
| **real_guidance_scale** | 1.75 | CFG scale (Real UNet) |
| **fake_guidance_scale** | 1.0 | CFG scale (Fake UNet) |
| **gradient_checkpointing** | 根据args | 梯度检查点 |
| **num_processes** | 2 | 2x RTX 5090 |

---

## 六、总结与行动项

### 6.1 当前配置特点
✅ **优点**:
1. 使用 BF16 混合精度，训练速度快
2. FSDP HYBRID_SHARD 策略在多GPU间平衡通信和计算
3. 损失计算用 FP32，数值稳定性好
4. Real UNet 冻结，减少计算量

⚠️ **需要注意**:
1. FSDP 自动 upcast 导致部分参数为 FP32，影响显存
2. 三个 UNet 模型（Generator、Real、Fake）同时存在，显存占用大
3. Decoupled DMD 需要多次前向传播（CA、DM分支），计算开销高

### 6.2 优化优先级

**高优先级**:
1. ✅ 添加显式的 MixedPrecision 策略，减少不必要的 upcast
2. ✅ 更新 deprecated API (scheduler.config.*)
3. ✅ 验证检查点精度一致性

**中优先级**:
4. 考虑使用 gradient checkpointing 减少显存
5. 监控不同精度转换点的数值稳定性

**低优先级**:
6. 探索 LoRA 训练以减少显存
7. 使用更激进的 FSDP offload 策略（如 offload_params=true）

---

## 七、代码修改建议（可选）

如需应用上述优化，请告知，我可以帮你修改代码。重点修改：
1. 统一精度管理
2. 更新 deprecated API
3. 添加检查点验证
4. 优化 MixedPrecision 配置
