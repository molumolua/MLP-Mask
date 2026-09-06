# Hard MLP-channel consistency GRPO

这个独立 recipe 保持 rollout、old-log-prob、GRPO actor loss 和 validation 全部为
clean policy。每个 optimizer step 只在 actor 内增加一次 teacher-forced auxiliary
pass：每个 Transformer block 随机 hard-mask 精确 10% SwiGLU channels，并最小化
clean 输出分布到 masked 输出分布的 KL。

它不导入任何其他 `recipe/`。实现只复用 verl 核心 worker/trainer，并在核心 actor 中
使用一个通用的 response-logits callback 和 auxiliary-backward hook。

## 与已有工作的关系

- R-Drop 在同一个样本上执行两个 dropout 子模型 forward，并最小化双向 KL。
- LR-Drop 把一致性进一步扩展到 Transformer 的层级表示。
- Consistent Dropout for Policy Gradient 指出：policy-gradient rollout 和 update 若使用
  不同 dropout mask，会破坏 policy-gradient/importance-ratio 的一致性。

本 recipe 不是上述方法的原样复现。它采用不对称的 clean-teacher → masked-student
KL，并且 mask 只用于 auxiliary loss。用于 PPO ratio 的 rollout 和 update 都是 clean，
所以随机 mask 不会进入 GRPO importance ratio。

## 训练目标

首先用完整模型产生全部 rollout，并计算标准 GRPO：

```text
L = L_grpo(clean) + kl_coef * KL(stopgrad(p_clean) || p_masked)
```

两次 actor forward 使用完全相同的 `input_ids`、prompt、completion 和 causal token
positions。masked pass 是 teacher forcing，不会重新采样 token，也没有独立 reward、
uid 或 advantage。

一个 actor micro-batch 的顺序固定为：

1. clean forward，保存 detached teacher distribution；
2. clean GRPO backward；
3. 切换本 step 的 hard mask；
4. 相同序列 masked forward，计算 KL 并立即 backward；
5. 恢复 clean route；
6. 所有 micro-batch 累积完成后，统一 gradient clipping 和 optimizer step。

clean 和 masked 分别立即 backward 是刻意设计：gradient checkpointing 在 backward
中会重算 forward，分阶段执行可确保重算时 controller 仍处于正确 route。

## Hard channel mask

Qwen/Llama gated MLP 为：

```text
z = SiLU(gate_proj(h)) * up_proj(h)
MLP(h) = down_proj(z)
```

masked auxiliary pass 使用：

```text
z_masked = keep_mask * z
keep_mask in {0, 1}
```

每层始终选择恰好 `round(mask_ratio * d_ff)` 个 channel。默认的
`selection_strategy=random, score_method=none` 保留原来的均匀无放回抽样；也可以用
上一 optimizer step 的 clean GRPO 信号选择下一步的 mask。每个 actor data-parallel
rank 对统计量做 all-reduce，并由相同的 `random_seed + mask_version` 确定性地产生相同
mask。没有 `1 / (1-p)` inverted-dropout scaling。

mask 不安装到 vLLM，因而 generation 和 validation 永远不会意外进入 masked route。

## Channel score 与选择规则

从 `mlp_channel_mask` 提取了三种不增加模型 pass 的 response-conditioned score：

- `relative_activation`：先计算每个样本 response token 上的 `RMS(z_j)`，再和历史
  clean-policy EMA 比较 `(current - ema) / max(ema, eps)`。它偏向最近变得异常活跃、
  可能正在发生功能漂移的 channel；第一次 update 只建立 baseline。
- `output_contribution`：`RMS(z_j) * ||W_down[:, j]||_2`。它同时考虑 activation 大小
  和 channel 写回 residual stream 的尺度，最贴近“移除后 clean 分布会明显改变”的
  consistency 初衷，建议作为第一选择。
- `gradient_activation`：clean GRPO backward 上的
  `mean_response |z_j * dL_grpo/dz_j|`。它是当前 RL objective 的一阶 task-conditioned
  importance；比前两者多一些逐 token FP32 hook/reduction 工作，但仍没有额外 forward、
  backward、rollout 或模型切换。

另外支持一个直接的 parameter-update score：

- `updated_fraction`：第 `l` 层第 `c` 个 channel 定义为
  `concat(gate_proj[c,:], up_proj[c,:], down_proj[:,c])`，统计其中相对 pre-RL BF16
  reference 满足 `|delta| > atol` 的坐标比例。比例越高，越容易被 consistency mask。
  它在 optimizer step 后扫描当前 MLP shard，要求 FSDP2 和
  `parameter_update_diagnostics_enabled=true`。

选择规则与 score 独立：

- `soft_top`（推荐）：把每层 score percentile rank 映射到
  `1 + (weighted_max_ratio - 1) * rank ** weighted_rank_power`，再按权重精确无放回抽样。
  默认权重范围 `[1, 4]`，既集中到重要 channel，也持续覆盖其它 channel。
- `hard_top`：每层确定性 mask score 最大的 channel，适合作为强干预 ablation；它容易
  长期重复打击少量 channel，不建议作为唯一主实验。
- `random`：原始均匀 baseline，必须配 `score_method=none`。

activation score 在 step `t` 的 clean GRPO backward 中顺手收集；`updated_fraction`
在 step `t` 的 optimizer step 后计算。两者都到 step `t+1` 才用于选 mask，因此一个
optimizer step 内的所有 auxiliary micro-batch 始终共享同一个固定 mask，不会产生
“一边收集一边改 mask”的目标漂移。score state 与当前 mask 一起进入 checkpoint。

## KL 实现

`kl_top_k=64` 是默认的低显存模式。对每个有效 response token：

1. clean teacher 取 top-64 token；
2. 保留每个 top token 的概率；
3. 把其余词表概率精确聚合成一个 tail bucket；
4. masked student 在相同 65 个类别上计算 forward KL。

所以它是 coarsened vocabulary 上严格非负的 categorical KL，而不是仅比较 sampled
token log-prob。它忽略 tail 内部 token 之间的重新分配，因此是 full-vocabulary KL 的
data-processing lower bound。

设置 `kl_top_k=0` 可计算完整 categorical KL，但 clean teacher 的
`response_tokens × vocab_size` log-prob 必须保留到 masked pass，长序列 4B 训练很容易
增加数 GB 显存；默认不建议。

## 启动

```bash
cd /Users/molu/verl-mlp_channels_mask
bash recipe/mlp_channel_consistency/grpo_mlp_channel_consistency_qwen3-4b_offline.sh
```

完全不执行 masked forward/backward 和 auxiliary loss、但保留相同诊断指标的 matched
clean-GRPO baseline：

```bash
cd /Users/molu/verl-mlp_channels_mask
bash recipe/mlp_channel_consistency/baseline_grpo_qwen3-4b_offline.sh
```

baseline 强制 `auxiliary_enabled=False` 和 `kl_coef=0`。它仍记录主 loss、主梯度 RMS
以及每次 validation 的参数更新稀疏度；所有 auxiliary loss/gradient 和 mask 指标保留
相同 key，但值为零，因而实验组和 baseline 可以直接使用同一套 dashboard。

常用覆盖：

```bash
mask_ratio=0.10 \
kl_coef=0.01 \
kl_top_k=64 \
kl_micro_batch_size_per_gpu=1 \
bash recipe/mlp_channel_consistency/grpo_mlp_channel_consistency_qwen3-4b_offline.sh
```

四个 score-aware 入口：

```bash
bash recipe/mlp_channel_consistency/grpo_mlp_channel_consistency_qwen3-4b_output_contribution_soft_top_offline.sh
bash recipe/mlp_channel_consistency/grpo_mlp_channel_consistency_qwen3-4b_gradient_activation_soft_top_offline.sh
bash recipe/mlp_channel_consistency/grpo_mlp_channel_consistency_qwen3-4b_relative_activation_soft_top_offline.sh
bash recipe/mlp_channel_consistency/grpo_mlp_channel_consistency_qwen3-4b_updated_fraction_soft_top_offline.sh
```

也可以直接覆盖；例如把 soft sampling 改为 deterministic top-k：

```bash
selection_strategy=hard_top score_method=output_contribution \
bash recipe/mlp_channel_consistency/grpo_mlp_channel_consistency_qwen3-4b_offline.sh
```

模型和数据路径也可覆盖：

```bash
MODEL_PATH=/path/to/Qwen3-4B-Base \
TRAIN_FILE=/path/to/train.parquet \
TEST_FILE='["/path/to/test.parquet"]' \
bash recipe/mlp_channel_consistency/grpo_mlp_channel_consistency_qwen3-4b_offline.sh
```

## 主要指标

- `mlp_consistency/kl`：未乘系数的 response-token mean KL；
- `mlp_consistency/weighted_kl`：实际加入梯度的 `kl_coef * KL`；
- `mlp_consistency/main_pg_loss_step` 与 `weighted_kl_step`：把 contribution-scaled
  micro-batch 值求和后，本 optimizer step 的两个完整目标值；
- `mlp_consistency/aux_to_main_loss_abs_ratio`：两个完整目标绝对值之比；loss
  接近零时该比值会不稳定，因此调参时应以梯度比为主；
- `mlp_consistency/main_grad_rms_sampled`：本次 optimizer step 的 clean GRPO
  梯度在固定坐标样本上的 RMS；
- `mlp_consistency/aux_grad_rms_sampled`：已经乘过 `kl_coef` 的一致性 KL
  梯度 RMS；
- `mlp_consistency/aux_to_main_grad_ratio_sampled`：上述两者之比。这是调
  `kl_coef` 最直接的量；由于先跨全部 micro-batch 累积梯度再求比值，改变
  `kl_micro_batch_size_per_gpu` 不会改变定义；
- `mlp_consistency/main_aux_grad_cosine_sampled`：两个分支梯度方向的 cosine；
- `mlp_consistency/gradient_sample_fraction|count`：估计所用的固定坐标样本；
- `mlp_consistency/response_tokens`；
- `mlp_consistency/micro_batches`：每个 clean micro-batch 对应的 masked KL 子批次数；
- `mlp_consistency/mask_version`；
- `mlp_consistency/realized_mask_fraction`；
- `mlp_consistency/masked_per_layer_min|max`；
- `mlp_consistency/selection_is_random|soft_top|hard_top`；
- `mlp_consistency/selection_used_score`：warmup/fallback random 时为 0；
- `mlp_consistency/selected_rank_mean`：soft-top 实际抽中 channel 的平均 percentile rank；
- `mlp_consistency/score_current_mean|min|max` 与 `score_initialized|updated`；
- `mlp_consistency/score_is_relative_activation|output_contribution|gradient_activation|updated_fraction`；
- `mlp_consistency/updated_fraction_atol`；
- `timing_s/mlp_consistency_updated_fraction`：逐 step MLP 参数搬运、比较与汇总耗时；
- `mlp_consistency/hard_mask=1`；
- `mlp_consistency/inverted_dropout_scaling=0`。

默认每卡固定分层抽样 262,144 个 trainable parameter 坐标，两个 FP32 梯度向量约
占 2 MiB。跨卡汇总后得到整个 sharded actor 的 RMS 比值；这是无偏近似诊断，避免
为精确拆分两个 loss 的全量梯度而再保存一份 4B 参数梯度。

每次 validation 还会记录：

- `val-aux/parameter_update/sparsity_atol_1e-5`：与 pre-RL 初始化相比仍未变化的
  BF16 参数比例；
- `val-aux/parameter_update/updated_fraction_atol_1e-5`：发生变化的比例，即
  `1 - sparsity`；
- `val-aux/parameter_update/mean_abs_delta_bfloat16` 与
  `rms_delta_bfloat16`：BF16 参数位移的平均绝对值和 RMS；
- `val-aux/parameter_update/parameter_count_billions` 与 `atol`：统计规模及阈值。

这里复现 Mukherjee et al., *Reinforcement Learning Finetunes Small Subnetworks in
Large Language Models* 的口径：令 pre-RL 参数为 `theta_0`、当前参数为 `theta_t`，
`sparsity = 1 - ||theta_t - theta_0||_0 / n`；实现上先转 BF16，再用
`torch.isclose(delta, 0, atol=1e-5)` 判为未更新。论文报告的是训练完成的 checkpoint，
因此训练早期或中途 validation 的数值不应被要求立刻落入论文最终结果区间。

- 论文：https://arxiv.org/abs/2505.11711
- 官方实现：https://github.com/SagnikMukherjee/sparsity_in_rl/blob/main/src/check_sparsity.py

pre-RL BF16 reference 在 worker 初始化时保存为每卡的 CPU FSDP shard，不保留额外
GPU 全量模型。以 4B 模型、4-way full shard 为例，每个 actor process 约增加 2 GB
CPU 内存（整机合计约 8 GB）。resume 时 reference 仍来自 `model.path`，所以
`model.path` 必须保持为原始 pre-RL 模型，而不能改成已训练 checkpoint。
如果只需要新的 online channel score、不需要复现该全局诊断，可以显式关闭这份
reference 及 validation 扫描：

```bash
parameter_update_diagnostics_enabled=False \
bash recipe/mlp_channel_consistency/grpo_mlp_channel_consistency_qwen3-4b_output_contribution_soft_top_offline.sh
```

### updated_fraction 选权的实现与代价

`score_method=updated_fraction` 已把原来的全局 validation 指标拆成逐 layer/channel
统计。它复用同一份 pre-RL BF16 CPU reference；每个 optimizer step 后只遍历三组 MLP
projection，把当前 local shard 搬到 CPU，按 channel 累加 changed/total counts，再跨
FSDP rank all-reduce `[2, num_layers, d_ff]` 统计量。它不载入第二个模型，不切换
actor/vLLM route，也不增加 rollout、forward 或 backward。

主要代价是：常驻 CPU reference 仍接近每 rank 一份 BF16 FSDP shard；此外每 step
都要传输并比较 MLP 参数，而不再只是 validation 时低频扫描。因此这个入口强制 FSDP2
和 `parameter_update_diagnostics_enabled=true`。建议先测量
`timing_s/update_actor` 与 step wall time，再决定是否降低使用频率；当前实现为了语义直接，
每 step 更新一次。

还要注意它是相对 pre-RL 的 cumulative binary coverage：训练后期可能逐渐接近 1、降低
channel 间区分度。`gradient_activation` 更便宜且反映当前 objective；
`updated_fraction` 更直接反映“RL 到目前为止主要改动了哪些 channel”，两者回答的问题
不同。

仓库现有 `recipe/mlp_channel_relative_update` 还提供了一个更直接的候选：它在自定义
FSDP2 AdamW 内按 `gate_proj` row、`up_proj` row、`down_proj` column 汇总每个 channel
的 AdamW-preconditioned relative update energy。复用这套统计不需要 CPU pre-RL 模型，
也不需要额外 forward/backward；代价是必须接管 optimizer step、遍历 MLP update 并
all-reduce 一个 `[num_layers, d_ff]` 张量。若做下一阶段，建议新增一个“只记录、不改变
update”的 optimizer 模式，再把前一步 relative-update EMA 喂给 consistency，而不是
把会随训练逐渐饱和的二值 cumulative `updated_fraction` 当作唯一权重。

## 实验解释与风险

R-Drop/LR-Drop 为 dropout consistency 提供了直接先例，structured dropout 也有独立
正则化研究；但尚没有证据证明“Qwen3-4B outcome-GRPO、所有 MLP 层同时 hard-mask
10%、clean→masked KL”一定有效。这仍是一个新的组合实验。

10% hard mask 比普通小扰动强，而且同时作用于所有 block。优先观察：

- KL 是否很快爆炸或长期不下降；
- actor grad norm、PPO KL 和 clip fraction 是否异常；
- clean validation 是否低于完全相同预算的标准 GRPO；
- `kl_coef=0` matched control 是否与标准 GRPO 一致。

建议第一轮至少比较 `kl_coef=0.001/0.01/0.05`，不要只跑一个系数。mask 本身不进入
rollout，但 auxiliary pass 会增加一次 actor forward/backward；端到端开销取决于
generation 与 actor update 的占比。

`kl_micro_batch_size_per_gpu` 只控制 masked teacher-forced KL 分支，不改变 clean GRPO
的动态 micro-batch，也不改变 loss 的 response-token mean 归一化。Qwen3-4B 的大词表
会让 KL 分支的 response logits 占用较多显存，默认值 `1` 是面向 80GB GPU 的保守
设置；确认显存余量后可以调大来换取吞吐。
