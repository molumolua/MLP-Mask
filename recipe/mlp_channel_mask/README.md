# GRPO MLP-channel structured intervention

这是一个独立 recipe。它只依赖 verl 核心组件，不导入 `recipe/` 下的任何其他实验。
默认实验对象是 dense Qwen3-4B，训练数据和六个验证集与 DenoiseRL 的对应脚本一致。

## 实验目标

在同一个 policy step 中同时优化两条路径：

- `clean`：所有 MLP channels 正常可用；
- `masked`：每一个 Transformer block 内，通过 soft-top 屏蔽 10% MLP channels。

最终部署和验证仍使用完整的 clean 模型。masked 路径是训练期 intervention，不是剪枝，也不改变参数量。

Qwen/Llama 风格的 gated MLP 为

\[
z_l=\operatorname{SiLU}(W_{g,l}h_l)\odot W_{u,l}h_l,
\qquad
\operatorname{MLP}_l(h_l)=W_{d,l}z_l.
\]

masked 路径只在 `down_proj` 之前插入逐 channel mask：

\[
\widetilde z_l=m_l\odot z_l,
\qquad m_{l,j}\in\{0,1\}.
\]

因此一个 channel 对应 `gate_proj`、`up_proj` 的一个中间维输出和 `down_proj` 的一个输入维；这正是 dense Transformer MLP neuron 的自然结构化单位。

## 一个训练 step

1. 将每个 prompt 扩成 8 个 clean 和 8 个 masked 样本，总预算仍是 16。
2. actor 权重只同步到 vLLM 一次。
3. vLLM 在 clean route 下调用一次 `generate`。
4. 清除 prefix cache，切换固定地址的 mask buffer，再在 masked route 下调用一次 `generate`。
5. 将两次输出恢复到原始交错顺序，分别保留 route、mask version 和 vLLM behavior log-prob。
6. clean 与 masked 使用不同的 GRPO `uid`，各自在本 route 的 8 个样本内计算均值和标准差。
7. actor update 先按 route 拆成同质 micro-batches，再在同一个 optimizer step 中累计两条路径的梯度。
8. clean backward 收集所选评分需要的统计；因果版本则使用真实的两路 reward gap。
9. step 结束后按新评分准备下一版 soft-top mask。

默认在第一个 rollout 前先生成一版均匀随机 mask。评分可用后，后续版本改用
soft-top 权重；同一个 step 的 rollout、old-logprob 和 actor update 始终共享同一版本。

## route 条件下的 GRPO

若同一个 prompt 的原始 id 为 `u`，recipe 将它改成：

\[
u_{clean}=u\mathbin{::}\text{clean},
\qquad
u_{masked}=u\mathbin{::}\text{masked}.
\]

由此两条 route 的 advantage 分别为

\[
A_i^{(r)}=
\frac{R_i^{(r)}-\mu_r}{\sigma_r+\epsilon},
\qquad r\in\{clean,masked\}.
\]

最终 actor objective 等权平均，而不是把 16 个异质 rollout 放进同一个 GRPO 组：

\[
L=\frac12L_{clean}+\frac12L_{masked}.
\]

`loss_group_id` 和 `loss_group_normalizer=2` 保证 dynamic batching 怎样切 micro-batch 都不会改变这两个 route 的相对权重。两条路径在一次 `backward/optimizer.step` 序列中累积，因此更新后的仍是同一个 policy。

跨 actor DP rank 的 token balancing 也按 route 分别求解：每个 rank 获得完全相同数量的 clean 样本和完全相同数量的 masked 样本，再在各自配额内平衡 token workload。这样不会因普通 length balancing 恰好把某个 rank 分成单一路径而改变目标权重。

与 DenoiseRL 一致，`bypass_old_logprob_for_rollout=false`，因此 actor 会重新计算
`old_log_probs`。重算前按 `route_id` 拆成 clean/masked 同质 micro-batch，masked
样本不会走 clean actor route，反之亦然。vLLM 产生的 `rollout_log_probs` 仅用于
rollout/actor 差异诊断，不直接充当 `old_log_probs`。

## 四种评分与统一 soft-top

`score_method` 与选择方式彼此独立。保留的原评分是
`relative_activation`：先按样本计算有效 response causal positions 上的 channel
RMS，再与此前的正常激活 EMA 比较：

\[
s_{l,j}=\frac{\operatorname{RMS}(z_{l,j})-\operatorname{EMA}_{l,j}}
{\max(\operatorname{EMA}_{l,j},\epsilon)}.
\]

新增三种评分：

- `output_contribution`：
  \(s_{l,j}=\operatorname{RMS}(z_{l,j})\lVert W_{d,l}[:,j]\rVert_2\)。它把
  channel 激活与下投影列范数结合，对 `z_j` 和对应权重列的互逆缩放不敏感。
- `gradient_activation`：
  \(s_{l,j}=\operatorname{mean}|z_{l,t,j}\,\partial J/\partial z_{l,t,j}|\)。它使用
  clean policy objective 的真实 backward gradient，因而是当前任务条件下的一阶评分。
- `causal_ablation`：从实际 clean–masked reward gap 和每步变化的随机分组 mask
  在线估计 channel mask 概率对分组干预效果的影响。实现使用 soft-top 无放回采样
  的精确 log-prob score-function contrast，因此在权重自适应变为非均匀后仍与实际
  assignment policy 一致。一次 rollout 只提供整个 mask 的标量结果，所以这是随机
  分组估计，不是昂贵的逐 channel 精确消融；指标名称明确使用 `group_*`。

前三种只统计产生 response token log-prob 的 causal position，并排除 padding 和
无效 response token。多 actor rank 会先 all-reduce，再更新评分。`score_ema_beta`
控制最终选择分数的平滑；原评分另用 `activation_ema_beta` 维护正常激活基线。

四种评分统一使用 `selection_strategy=soft_top`。每层先把评分变成升序百分位
`r in [0,1]`，再按

```text
weight = 1 + (weighted_max_ratio - 1) * r ** weighted_rank_power
```

无放回采样恰好 `round(mask_ratio * d_ff)` 个 channels。默认最高权重是最低权重的
4 倍、`weighted_rank_power=2`。这不是 hard top-k：高分 channel 更常被选中，但
不会被永久锁定；每层配额严格相同。评分尚未建立时使用均匀权重。

历史的 `weighted_random` 仍作为原 relative-activation 实验的兼容别名保留；新实验
统一写 `soft_top`。`selection_strategy=random` 的逐层/全局均匀随机对照也继续可用。

## 为什么 validation 使用 clean route

训练/测试 forward 不完全一致是有意设计的：优化的是 clean 与 intervention 的混合目标，部署目标仍是完整模型。clean 样本稳定原模型能力；masked 样本迫使参数在主路径暂时不可用时找到替代计算。

验证顺序固定为：

1. 强制 route=`clean`，不使用任何 mask；
2. 完成 validation；
3. 根据本 step 的评分统计刷新 mask；
4. 新 mask 从下一个 train step 生效。

这样验证不会受到旧 mask 或刚刷新 mask 的影响。

## 性能设计

- 两个 vLLM `generate` 调用位于同一个 `rollout_mode()` 内，所以每 step 只有一次 actor→vLLM 权重同步。
- 默认是 8+8，而不是 16+16；总生成条数与基线 n=16 相同。因此生成 token FLOPs 接近原预算，而不是天然翻倍，但会增加一次调用调度和一次 cache reset。
- clean 与 masked 的 KV cache 不能共享，因为 mask 改变了每层 hidden state。prefix caching 保持开启，以便同一路径内的重复 prompt 可以复用 prefill；但在 clean 前、clean→masked 边界和 masked 结束后都显式 reset，因此 cache 永远不会跨 route 使用。
- vLLM 类在 engine 构建和 CUDA graph capture 之前完成 patch。每层 mask tensor 的地址固定，route 切换只做 in-place copy，不会每 token 在线重建 hook 或 mask。
- mask 乘法复杂度为 `O(L*T*d_ff)`，相对 MLP 矩阵乘法通常很小，但它不会减少 GEMM FLOPs，所以这不是推理加速方案。
- actor 仍处理总共 16 条 trajectory。route 分开 packing 会增加少量调度开销，但总 token 数不变。
- `relative_activation` 和 `output_contribution` 增加一次 channel reduction；
  `gradient_activation` 还计算 `abs(z*grad)`。every-step 模式下这些工作每步发生，
  因而提供独立 timing。`causal_ablation` 不安装 activation hook。

如果把配置改成 16 clean + 16 masked，总 rollout 和 actor token 预算才会接近基线的两倍。

## W&B metrics

route 指标：

- `clean_actor/reward_mean`, `masked_actor/reward_mean`
- `clean_actor/entropy`, `masked_actor/entropy`
- `route/entropy_gap_masked_minus_clean`
- `clean_actor/advantage_mean`, `masked_actor/advantage_mean`
- `clean_actor/pg_loss`, `masked_actor/pg_loss`
- `clean_actor/pg_clipfrac`, `masked_actor/pg_clipfrac`
- `clean_actor/ppo_kl`, `masked_actor/ppo_kl`
- 两条 route 各自的 sample 数、response length、rollout/old log-prob 以及 reward extra-info
- `route/reward_gap_clean_minus_masked`

mask 指标：

- `mlp_mask/version`：刷新后的当前版本
- `mlp_mask/rollout_version_used`：本 step rollout 实际使用的版本
- `mlp_mask/random_resample_every_step`：是否在每个 step 的 rollout 前随机重采样
- `mlp_mask/random_scope_is_global`：1 表示全局随机配额，0 表示逐层配额
- `mlp_mask/selection_is_soft_top`
- `mlp_mask/weighted_max_ratio` / `weighted_rank_power`
- `mlp_mask/weighted_selected_rank_mean`
- `mlp_mask/weighted_selected_top_1pct_fraction`
- `mlp_mask/soft_top_used_score`
- `mlp_mask/current_channels` / `current_fraction`
- `mlp_mask/masked_per_layer_min` / `max`：用于确认 block 内配额严格平衡
- `mlp_mask/ever_unique_channels` / `ever_unique_fraction`
- `mlp_mask/ever_unique_per_layer_mean|min|max`
- `mlp_mask/new_unique_channels`
- `mlp_mask/overlap_with_previous` / `turnover_fraction`
- `mlp_mask/cumulative_assignments`
- `mlp_score/is_relative_activation|is_output_contribution|is_gradient_activation|is_causal_ablation`
- `mlp_score/current_mean|max|min` 与 `mlp_score/ema_mean|max|min`
- `mlp_score/collection_enabled`, `mlp_score/update_every_step`, `mlp_score/ema_beta`
- `mlp_causal/group_reward_gap|group_reward_gap_ema|group_reward_gap_residual`
- `mlp_causal/observations|score_updated`

所有新增阶段都有秒级 timing，包括：

- `timing_s/dual_batch_build`
- `timing_s/balance_dual_route`
- `timing_s/dual_rollout_weight_sync`
- `timing_s/gen_clean`, `timing_s/gen_masked`
- `timing_s/mlp_prefix_cache_reset_before_clean|between_routes|after_masked`
- rollout/actor route switch
- clean/masked actor forward-backward
- `timing_s/mlp_activation_accumulate_cpu`（hook 的 CPU dispatch，不含异步 GPU 完成时间）
- `timing_s/mlp_activation_enabled_actor_update`（开启评分统计的完整 actor-update wall time）
- score all-reduce、mask selection、总 refresh 和 actor→rollout mask sync
- `timing_s/testing_clean`

validation 会在每个 `data_source` 内按规范化后的完整题面合并重复行，而不是按可能
不同的行号或 `uid` 区分。若同一道题有 16 条生成结果，会额外记录标准无偏估计的
`pass@1`、`pass@2`、`pass@4`、`pass@8` 和 `pass@16`；同时记录 unique prompt 数及
每题采样数的 min/max，便于确认 AIME 的 30 题 × 16 次是否被正确恢复。

GRPO 不使用 critic，因此不存在可正确解释的 `clean_critic/masked_critic`。本 recipe 明确拒绝启用 critic，而不是记录两个虚假的 critic 指标。

## Checkpoint

每个 actor checkpoint 额外保存 `mlp_channel_mask.pt`，内容包括：

- 当前 keep mask；
- 历史 ever-masked bitmap；
- 激活基线、选择分数及其 EMA 状态；
- 分组因果估计的 reward-gap 基线与观测数；
- mask version；
- cumulative assignments。

恢复时同时加载 actor controller 和 vLLM controller，下一次 rollout 会检查 batch version 与 worker version 是否一致，不一致则直接报错。

## 启动

```bash
cd /Users/molu/verl-mlp_channels_mask
bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_offline.sh
```

上面的默认脚本保留原 `relative_activation` 评分，并改为 soft-top 选择。三种新增评分
各有独立脚本：

```bash
bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_output_contribution_soft_top_offline.sh
bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_gradient_activation_soft_top_offline.sh
bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_causal_ablation_soft_top_offline.sh
```

按原 refresh 周期随机屏蔽每层 1% channels 的对照实验（脚本名为历史保留）：

```bash
bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_random10_offline.sh
```

每个训练 step 都重新随机屏蔽每层 10% channels：

```bash
bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_random10_every_step_offline.sh
```

每个训练 step 在全模型范围重新随机屏蔽精确 10% channels：

```bash
bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_global_random10_every_step_offline.sh
```

每个训练 step 逐层 weighted-random 屏蔽 1%，并使用更强的
`weight(r) = 1 + 10 r^2`：

```bash
bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_weighted_random1_strong_every_step_offline.sh
```

每个训练 step 在每层按固定 saliency 权重随机屏蔽 10% channels：

```bash
bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_weighted_random10_every_step_offline.sh
```

若希望全局随机但只按原 refresh 周期换 mask，可运行：

```bash
random_scope=global bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_random10_offline.sh
```

脚本默认 `WANDB_MODE=offline`，本地记录写入 `./wandb_offline`。之后可手动执行 `wandb sync <offline-run-dir>`。

数据或模型不在默认相对路径时可覆盖：

```bash
MODEL_PATH=/path/to/Qwen3-4B-Base \
TRAIN_FILE=/path/to/MATH7500-train.parquet \
TEST_FILE='["/path/to/aime25_test.parquet", "/path/to/MATH500-test.parquet"]' \
bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_offline.sh
```

## 测试与限制

CPU 单元测试：

```bash
python3 -m unittest -v recipe.mlp_channel_mask.test_intervention
```

当前刻意限制为：synchronous vLLM、FSDP/FSDP2 actor、dense Qwen2/Qwen3 SwiGLU、GRPO、无 critic、无 reference KL、vLLM PP=1 且 vLLM data-parallel-size=1。未知布局和 MoE 会在初始化阶段显式失败，避免训练过程中静默 mask 错位置。
