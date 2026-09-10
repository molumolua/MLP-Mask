# Two-mask MLP-channel GRPO with cross-route KL

独立 recipe，不在运行时导入其它 `recipe/`。默认对象为 dense Qwen3-4B；训练数据、
六个验证集和 offline W&B 设置沿用现有实验。资源估算见 [BUDGET.md](BUDGET.md)。

## 训练目标

每个 rollout/update step 为每层 SwiGLU 独立抽取两份 mask A/B。每份 mask 在每层
屏蔽恰好 `round(mask_ratio * intermediate_size)` 个 channels，默认 10%。
在 `down_proj` 前将对应中间激活乘零，不改变权重值，不使用 `1/(1-p)` 缩放。
两份 mask 允许重叠，也不会强制互补；同一步的所有 prompts 共享这对 mask。

每个 prompt 默认生成 16 条回答：A 路 8 条、B 路 8 条。所有 16 条保留同一个
原始 `uid`，GRPO 的 reward 均值、标准差在完整 16 条上计算。主损失使用仓库现有
vanilla clipped PPO policy loss，分别在各 route 的有效 response tokens 上求均值，
再按 1/2 等权相加。这里的 advantage estimator 是 GRPO，不引入 critic。

对全部 16 条回答，在相同 prompt、已生成 response 和 causal positions 上交叉对齐：
A 的 8 条由 A 提供 detached teacher 分布，让 B 学习；B 的 8 条由 B 提供 detached
teacher 分布，让 A 学习。每条回答独立对齐，不需要将 A 的回答与 B 的回答逐位置配对。

```text
P_A,t = policy(theta, mask_A)(next token | prompt, response[:t])
P_B,t = policy(theta, mask_B)(next token | prompt, response[:t])

L_main = 0.5 * mean_tokens_A(PPO_A) + 0.5 * mean_tokens_B(PPO_B)
KL_A_to_B,t = KL(stop_gradient(P_A,t) || P_B,t)  # 仅在 A 生成的轨迹上
KL_B_to_A,t = KL(stop_gradient(P_B,t) || P_A,t)  # 仅在 B 生成的轨迹上
L_aux  = (sum_A_tokens(KL_A_to_B) + sum_B_tokens(KL_B_to_A)) / (N_A_tokens + N_B_tokens)
L      = L_main + kl_coef * L_aux
```

每条轨迹只产生一个 teacher→student forward KL，辅助梯度只经过对侧分支。
A/B 共享模型参数；detach 阻断本次 teacher 计算路径，不冻结某一组永久参数。
主损失仍分别通过各自的生成分支更新。该目标是交叉蒸馏，不等价于旧版逐轨迹
双侧求导的对称 KL。`kl_coef` 的数值尺度应重新观察，不能假定梯度强度与旧版相同。

默认 `kl_top_k=64`：每个位置从**生成该轨迹的 teacher 路**选 top-64 token，另将
所有其余 token 的概率精确聚合成一个 tail。student 使用相同的 65 类划分计算 KL。
teacher 的概率和 token IDs 均不求导。这个近似不约束 tail 内部的概率重新分配，
是相应完整词表 forward KL 的下界。A 轨迹用 A 的 top-k，B 轨迹用 B 的 top-k。

`kl_top_k=0` 使用完整词表。`kl_token_chunk_size=128` 只分块处理词表计算的有效
response token 行，不截断 response，也不抽样对齐位置。尾部使用 logsumexp，
避免 top-k 概率接近 1 时相减导致精度损失。第一阶梯度用解析公式按块计算，
避免保留全部词表上的 KL 中间计算图；不支持对这个辅助算子求二阶导数。

## 一致性与 backward 顺序

1. 同步 actor 权重到 vLLM 一次，A/B 分别 generation；切换前后清空 prefix cache。
2. 回填每条样本的 `route_id`、`mask_version`，恢复原始行顺序。
3. old log-prob 重算及 PPO 更新使用该样本生成时的路由与同一版 mask。
4. 每个主损失 micro-batch 执行生成路由的 PPO forward，顺便保存 detached teacher
   分布与逐行有效 token 数，再执行 PPO backward。
5. 将缓存按轨迹切成辅助子批次，仅让对侧路由 forward，计算 teacher→student KL
   并 backward。A 轨迹走 B，B 轨迹走 A，每条轨迹只额外一次 forward/backward。
6. 清理 teacher 缓存，恢复生成路由，继续下一个主损失 micro-batch。
7. 累积所有主、辅助梯度后，统一 gradient clipping 和一次 optimizer step。

teacher 直接复用当前 PPO 前向的完整概率分布或 top-k＋tail，不单独 replay teacher，
也不把 vLLM 返回的单个采样 token log-prob 当成 teacher 的完整分布。
每次 backward 完成前都保持路由不变，适配 gradient checkpointing 的 forward 重算。

动态 batching 下，每个 actor rank 保持相等 A/B 样本配额；辅助子批次数通过 collective
对齐，缺少子批次的 rank 执行零权重重放，避免 FSDP collective 次序不一致。
主损失与辅助损失按全局有效 token 数修正 DP 平均，因此不会随 rank 长度分布或
micro-batch 切分改变梯度权重。dummy 重放不计入回答数、token 数和 loss。

validation 使用完整 clean 模型。下一个 rollout step 才更换 mask。

## 启动

在仓库根目录、已配置好模型与数据的 Linux CUDA 主机上运行：

```bash
bash recipe/mlp_channel_rdrop/grpo_mlp_channel_rdrop_qwen3-4b_offline.sh
```

默认 4 GPUs，FSDP2 全分片，vLLM TP=1、DP=1、PP=1；`n_rollouts=16`。
默认完整权重训练，gradient checkpointing 开启，SP=1，关闭 fused logits kernel，
关闭其它原生 dropout、entropy bonus 和 reference-policy KL，以便诊断主/辅助目标。
训练 sampling 使用 temperature>0、top-p=1、top-k=-1，与 actor 概率一致。
不支持多轮、MoE、LoRA 或 async rollout。该版本验证过 CPU 数学/路由/分布式逻辑；
真实 FSDP2+vLLM CUDA 训练仍需 GPU 主机验证。

主要参数：

```bash
mask_ratio=0.10 kl_coef=0.01 kl_top_k=64 random_seed=42 \
kl_micro_batch_size_per_gpu=1 kl_token_chunk_size=128 \
bash recipe/mlp_channel_rdrop/grpo_mlp_channel_rdrop_qwen3-4b_offline.sh
```

保持双 mask rollout，关闭辅助损失：

```bash
bash recipe/mlp_channel_rdrop/baseline_no_aux_qwen3-4b_offline.sh
```

关闭 channel mask 和辅助损失的 matched control（仍执行两次 generation、同样 16 条预算）：

```bash
bash recipe/mlp_channel_rdrop/baseline_clean_qwen3-4b_offline.sh
```

三个入口的默认实验名和 checkpoint 目录不同，并包含 `rdrop-cross` 以区分旧对称 KL
实验，避免默认自动续跑旧目录。可以追加 Hydra 参数，例如
`trainer.total_training_steps=2`；`MODEL_PATH`、`TRAIN_FILE`、`TEST_FILE`、`num_gpus`、
`n_rollouts`、`max_response_length`、`gradient_sample_size_per_rank` 等也可通过环境变量设置。
完整词表模式用 `kl_top_k=0`。默认辅助子批次为 1，若显存不足可先降低
`actor_ppo_max_token_len`，或设置 `use_dynamic_bsz=False` 让主损失也逐条处理。
启动脚本在 macOS 使用 `molu` 环境；Linux 默认 `python`，可用 `python_bin` 覆盖。

## 指标

以下 `mlp_rdrop/` 指标按完整 optimizer step 汇总：

| 指标 | 含义 |
| --- | --- |
| `main_pg_loss_step` | 两路等权的完整 PPO policy loss |
| `kl` | 对全部有效 response tokens 求均值的未加权交叉 forward KL |
| `a_to_b_kl` / `b_to_a_kl` | 对应来源轨迹上 teacher→student KL 的全局 token 均值 |
| `weighted_kl_step` | `kl_coef * kl`，实际加入目标的辅助项 |
| `total_loss_step` | 完整主损失加加权辅助项 |
| `aux_to_main_loss_abs_ratio` | `abs(weighted_kl_step) / max(abs(main_pg_loss_step), 1e-12)` |
| `loss_ratio_denominator_near_zero` | 主损失接近零时为 1；此时 loss 比值不适合判断优化强弱 |
| `main_grad_rms_sampled` / `aux_grad_rms_sampled` | 完整 step 的主/加权辅助梯度 RMS |
| `main_grad_l2_sampled` / `aux_grad_l2_sampled` | 固定坐标样本上的梯度 L2，不是全参数梯度范数 |
| `aux_to_main_grad_ratio_sampled` | 加权辅助梯度 / 主梯度的 RMS 比例 |
| `main_aux_grad_cosine_sampled` | 两个累积梯度的余弦，相同方向为 1，正交为 0，反向为 -1 |
| `main_aux_grad_angle_degrees_sampled` | 对应角度，单位度 |
| `grad_angle_defined` / `grad_ratio_defined` | 零范数导致比值/角度无定义时为 0，相应数值占位为 0 |
| `gradient_sample_fraction` / `gradient_sample_count` | 梯度诊断覆盖率与坐标数，默认每 rank 262144 个 |
| `aligned_response_rows` / `response_tokens` | 全局实际进入 KL 的回答数与有效 token 数 |
| `a_to_b_aligned_rows` / `b_to_a_aligned_rows` | 两个方向实际对齐的回答数，默认每 prompt 各 8 条 |
| `a_to_b_response_tokens` / `b_to_a_response_tokens` | 两个方向分别对齐的有效 token 数 |
| `auxiliary_padding_slots` | 为对齐 collective 次数而增加的零权重子批次 |
| `auxiliary_forward_calls` / `auxiliary_backward_calls` | 所有 actor rank 的额外模型调用数，含零权重子批次 |
| `teacher_cache_peak_tokens` | 单个主损失 micro-batch 缓存的有效 teacher token 数峰值，各 rank 取最大值 |
| `cross_route_forward_kl` | 当前使用交叉 forward KL 目标，固定为 1；是否启用仍看 `auxiliary_enabled` |
| `mask_a_fraction` / `mask_b_fraction` | 实际每路屏蔽比例 |
| `both_masked_fraction` / `either_masked_fraction` | 两路 mask 交集和并集占全部 channels 的比例 |
| `mask_version` / `kl_top_k` | 当前 mask 版本与 KL 配置 |
| `mask_a_reward_mean` / `mask_b_reward_mean` | 每路回答的平均原始 reward |

梯度统计发生在 clipping/optimizer 前，且在两路和全部 micro-batches 上先累积、再计算
比例与角度，包含 `kl_coef`。这些是 loss 梯度的关系，**不是 AdamW 两份参数更新的分解**。
梯度来自固定分层坐标抽样，是估计值；开启参数/梯度 offload 时仍可能产生额外搬运。
`timing_s/mlp_rdrop_auxiliary_step` 是各 rank 累积 student 前向/反向等辅助阶段耗时的
最大值。teacher 缓存现在发生在 PPO forward 内，另记为
`timing_s/mlp_rdrop_teacher_capture_step`，不计入前者，也不包含 PPO 模型 forward。
两者是主机 wall-clock 计时，CUDA 异步执行会影响归属，不是精确 kernel profile；
比较整体开销应结合 `timing_s/update_actor`、`timing_s/step` 和 generation 耗时。

每次 validation 同时记录原 consistency 的参数更新口径：

| 指标（`val-aux/parameter_update/`） | 含义 |
| --- | --- |
| `updated_fraction_atol_1e-5` | 相对 pre-RL BF16 参数，变化绝对值大于 `1e-5` 的坐标比例 |
| `sparsity_atol_1e-5` | 未变化比例，等于 1 减上述指标 |
| `updated_parameter_count` | 上述变化坐标的数量 |
| `mean_abs_delta_bfloat16` / `rms_delta_bfloat16` | BF16 差值的平均绝对值、RMS |
| `parameter_count_billions` / `enabled` | 全模型参数量与是否启用诊断 |

这是全模型参数相对 RL 开始前的**累计变化**，不是这一步非零梯度比例，也不是
channel 屏蔽比例。BF16 转换和减法与原 consistency 一致，阈值固定为 `1e-5`。
默认每个 actor rank 保存一份 pre-RL BF16 CPU shard。
可用 `parameter_update_diagnostics_enabled=False` 关闭这部分 CPU 内存与 validation 扫描。

## Checkpoint 与本地验证

checkpoint 额外保存 `mlp_channel_rdrop.pt`，包含两份 mask、版本、seed、比例与原始
`model.path`。resume 时必须使用相同的 pre-RL `model.path`，先由该模型建立 CPU
reference，再加载 RL 权重，因此累计参数更新指标不会在 resume 后重新归零。
不接受缺少该文件的训练 checkpoint；普通初始模型通过 `model.path` 指定。

```bash
./scripts/check-local-env
./scripts/test-local recipe/mlp_channel_rdrop -q
RUN_RDROP_DISTRIBUTED_TESTS=1 ./scripts/test-local recipe/mlp_channel_rdrop/test_distributed.py -q
```

测试覆盖：两份独立 mask 与恢复、HF/fused-vLLM 数学映射和 TP slice、checkpoint 重算、
完整/压缩词表 forward KL 的 student 梯度与有限差分、teacher stop-gradient、
16 条回答的交叉梯度 oracle、PPO 缓存复用与对侧调用次数、micro-batch 切分不变性、
零权重重放、异常清理、梯度与参数指标，以及真实两 rank Gloo/DDP 与
DTensor 分片统计。CPU 测试使用小模型，不会下载模型或启动真实 vLLM。
