# Online MLP-channel rarity weighting

这个 recipe 在标准 GRPO 的 actor old-log-prob 前向中读取 prompt token 的
post-gate SwiGLU 激活，不增加第二次模型前向，也不修改 rollout policy。

## 每一步的定义

对问题 `q`、层 `l`、channel `j`，首先对 prompt token 做 RMS：

```text
a[q,l,j] = sqrt(mean_t h[q,t,l,j]^2)
```

正常激活水平使用前序 optimizer step 的 EMA：

```text
deviation[q,l,j] = (a[q,l,j] - ema[l,j]) / max(ema[l,j], eps)
```

每层选 deviation 最大的 top-k channel。设 `C[l,j]` 是累计曝光题数、`N` 是
累计题数。默认关闭频率先验，直接使用累计经验频率：

```text
empirical_frequency[l,j] = C[l,j] / N
effective_frequency[l,j] = empirical_frequency[l,j]
```

如果显式设置 `use_frequency_prior=true`，则加入以 `p0 = k / d_ff` 为中心、
强度为 `tau` 的对称先验：

```text
p0 = k / d_ff
effective_frequency[l,j] = (C[l,j] + tau*p0) / (N + tau)
```

之后两种模式共用：

```text
rarity[l,j] = min(-log(max(effective_frequency[l,j], eps)), rarity_max)
raw_s_q = mean(rarity of selected channels)
```

先验默认关闭；开启时默认 `tau=64` 道伪题。无论是否开启先验，均使用
`rarity_max=8` 限制单个 channel 的 self-information。GRPO 的多个 response
在计数时除以固定 group size，因此 `C` 和 `N` 都以问题为单位。

设当前全局 actor batch 中共有 `B` 个样本，先令

```text
v_q = raw_s_q / mean_q(raw_s_q)
s_q = clip(v_q - lambda, 0.2, 5.0)
```

其中标量 `lambda` 通过二分求解，使 `sum_q(s_q) / B = 1`。这不是简单的“先
归一化再裁剪”：后者会在触碰上下界后破坏均值。当前做法是投影到
`[0.2, 5.0]` 与“全局均值为 1”两个约束的交集，因此同时硬性满足范围和均值，
并保持题目间的权重次序。所得 `s_q` 直接作为该问题所有 response 的 actor loss
multiplier。第一步尚无历史 EMA，因此所有问题的 `s_q=1`；该步只初始化正常
激活水平，不累计曝光。

EMA 在计算本 step deviation 之后更新，避免当前问题进入自己的基线。累计曝光、
EMA 和 step 计数保存在 actor checkpoint 的 `mlp_channel_rarity.pt` 中。

## 每步问题级 JSONL

启动脚本默认设置 `trainer.rollout_data_dir` 为
`${CKPTS_DIR}/rollout_data`。verl 原有的 response 级明细仍写到 `<step>.jsonl`；
本 recipe 另外把同一问题的全部 GRPO response 聚合为一行，写到：

```text
${CKPTS_DIR}/rollout_data/question_rarity/<step>.jsonl
```

每行包含问题 `prompt`、`average_accuracy`、`raw_s_q`、实际 actor loss 权重
`s_q`、各 rollout 的 accuracy/reward/rarity、数据集元信息、本实验 rarity 配置，
以及该步完整的 `mlp_rarity/*` 统计。`average_accuracy` 使用 verifier 返回的
`acc`；如果自定义 reward 没有返回 `acc`，该字段为 `null`，不会把可能带 shaping
的总 reward 误当成正确率。可通过环境变量 `rollout_data_dir=/path/to/dir`
修改输出位置。

## 启动

```bash
bash recipe/mlp_channel_rarity/grpo_mlp_channel_rarity_qwen3-4b_offline.sh
```

常用覆盖参数：

```bash
activation_ema_beta=0.99 \
topk_ratio=0.005 \
use_frequency_prior=True \
frequency_prior_strength=64 \
max_channel_rarity=8 \
min_loss_weight=0.2 \
max_loss_weight=5.0 \
bash recipe/mlp_channel_rarity/grpo_mlp_channel_rarity_qwen3-4b_offline.sh
```

`actor_rollout_ref.mlp_channel_rarity.layers=null` 表示观察全部 block；也可以在
Hydra 参数中传入固定层列表。当前实现要求 outcome GRPO、同步 rollout、FSDP/FSDP2、
Ulysses SP=1，并关闭 old-log-prob 的 dynamic batching，保证激活权重和 DataProto
行顺序严格一致。静态 old-log-prob micro batch 默认每 GPU 1 条；显存充足时可用
`log_prob_micro_batch_size_per_gpu=2`（或更大值）覆盖。

主要监控项：

- `mlp_rarity/normal_activation_mean`
- `mlp_rarity/selected_deviation_mean`
- `mlp_rarity/exposed_channel_fraction`
- `mlp_rarity/effective_frequency_min/max`
- `mlp_rarity/empirical_frequency_min_positive/max`
- `mlp_rarity/channel_rarity_max_observed`
- `mlp_rarity/raw_score_mean`
- `mlp_rarity/loss_weight_min/mean/max`
