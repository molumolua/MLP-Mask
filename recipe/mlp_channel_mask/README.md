# GRPO MLP-channel structured intervention

这是一个独立 recipe。它只依赖 verl 核心组件，不导入 `recipe/` 下的任何其他实验。
默认实验对象是 dense Qwen3-4B，训练数据和六个验证集与 DenoiseRL 的对应脚本一致。

## 实验目标

在同一个 policy step 中同时优化两条路径：

- `clean`：所有 MLP channels 正常可用；
- `masked`：每一个 Transformer block 内，屏蔽贡献度最高的 10% MLP channels。

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
8. 如果当前 step 是 mask-refresh step，clean backward 同时收集一次贡献度；其余 step 不注册 saliency hook。
9. 到达 `test_freq` 时，先做无 mask 的 clean validation，再选择下一版 mask。

初始 `mask_version=0` 时 mask 全为 1，因此第一个 train step 的两个 route 在函数上相同，只是独立采样。这个 step 用来建立第一版贡献度；step 结束后生成 `mask_version=1`。

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

## 贡献度与每层 top 10%

只在 clean policy-loss backward 上统计：

\[
c_{l,j}=\frac{1}{N}
\sum_{t\in\text{有效 response loss positions}}
\left|z_{l,t,j}\frac{\partial L_{clean}}{\partial z_{l,t,j}}\right|.
\]

这里使用的是产生 response token log-prob 的 causal position，即 response token 前一个位置，而不是简单取输入序列最后 `response_length` 个激活。padding 和无效 response token 均被排除。

多 GPU 上先对分子和 token 数做 all-reduce，再形成平均值。随后使用 EMA：

\[
\bar c_l^{(v)}=
\beta\bar c_l^{(v-1)}+(1-\beta)c_l^{(v)},
\]

默认 `beta=0.95`；第一次刷新直接使用当前统计。对每一层独立计算

\[
k=\max\left(1,\operatorname{round}(0.1d_{ff})\right),
\qquad
m_{l,j}=0\ \text{if}\ j\in\operatorname{TopK}(\bar c_l,k).
\]

所以不会出现浅层或深层垄断全局 mask 配额的情况。每层始终屏蔽相同数量的 channels。

作为对照实验，也支持 `selection_strategy=random`。random 模式不计算
gradient × activation，而是在每次 refresh 时对每一层独立、均匀地采样恰好
`round(mask_ratio * d_ff)` 个 channels。采样使用 `random_seed + mask_version`，
因此不同 actor worker 会得到相同 mask，同一实验可以复现，而新版本会重新采样。
它仍保留 version 0 的全 1 warmup：第一个 train step 结束后才产生第一版随机 mask。

random 采样范围由 `random_scope` 控制：

- `per_layer`：每层精确采样 `round(mask_ratio * d_ff)` 个 channels；
- `global`：在全部 `(layer, channel)` 中精确采样
  `round(mask_ratio * num_layers * d_ff)` 个位置，每层实际数量允许波动。

两者的全模型期望比例相同；模型每层宽度一致时，`per_layer` 的总量也与 global
基本相同，但它消除了层间采样方差。

设置 `random_resample_every_step=true` 后使用另一种时序：每个 train step 的
rollout 之前先生成一版新随机 mask，所以 step 1 就使用 `mask_version=1`，没有
全 1 warmup。同一个 step 内 mask 保持不变，只在下一个 step 开始前再次采样。
这个频率独立于 validation/checkpoint 的 `test_freq`。

top-saliency 模式只在 `warmup_steps=1` 和之后每个 `test_freq` 对应的单个 train step 上收集贡献度。random 模式完全不挂 activation-gradient hook，也不执行 saliency all-reduce。

## 为什么 validation 使用 clean route

训练/测试 forward 不完全一致是有意设计的：优化的是 clean 与 intervention 的混合目标，部署目标仍是完整模型。clean 样本稳定原模型能力；masked 样本迫使参数在主路径暂时不可用时找到替代计算。

验证顺序固定为：

1. 强制 route=`clean`，不使用任何 mask；
2. 完成 validation；
3. 根据本 step 的 clean backward 统计刷新 mask；
4. 新 mask 从下一个 train step 生效。

这样验证不会受到旧 mask 或刚刷新 mask 的影响。

## 性能设计

- 两个 vLLM `generate` 调用位于同一个 `rollout_mode()` 内，所以每 step 只有一次 actor→vLLM 权重同步。
- 默认是 8+8，而不是 16+16；总生成条数与基线 n=16 相同。因此生成 token FLOPs 接近原预算，而不是天然翻倍，但会增加一次调用调度和一次 cache reset。
- clean 与 masked 的 KV cache 不能共享，因为 mask 改变了每层 hidden state。prefix caching 保持开启，以便同一路径内的重复 prompt 可以复用 prefill；但在 clean 前、clean→masked 边界和 masked 结束后都显式 reset，因此 cache 永远不会跨 route 使用。
- vLLM 类在 engine 构建和 CUDA graph capture 之前完成 patch。每层 mask tensor 的地址固定，route 切换只做 in-place copy，不会每 token 在线重建 hook 或 mask。
- mask 乘法复杂度为 `O(L*T*d_ff)`，相对 MLP 矩阵乘法通常很小，但它不会减少 GEMM FLOPs，所以这不是推理加速方案。
- actor 仍处理总共 16 条 trajectory。route 分开 packing 会增加少量调度开销，但总 token 数不变。
- saliency 的额外 `abs(z*grad)` 和 channel reduction 只发生在刷新 step。它会增加该 step 的显存带宽和临时张量压力，因此提供了独立 timing。

如果把配置改成 16 clean + 16 masked，总 rollout 和 actor token 预算才会接近基线的两倍。

## W&B metrics

route 指标：

- `clean_actor/reward_mean`, `masked_actor/reward_mean`
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
- `mlp_mask/current_channels` / `current_fraction`
- `mlp_mask/masked_per_layer_min` / `max`：用于确认 block 内配额严格平衡
- `mlp_mask/ever_unique_channels` / `ever_unique_fraction`
- `mlp_mask/ever_unique_per_layer_mean|min|max`
- `mlp_mask/new_unique_channels`
- `mlp_mask/overlap_with_previous` / `turnover_fraction`
- `mlp_mask/cumulative_assignments`
- `mlp_saliency/mean|max|min|response_tokens|layers_observed`
- `mlp_saliency/collection_enabled`

所有新增阶段都有秒级 timing，包括：

- `timing_s/dual_batch_build`
- `timing_s/balance_dual_route`
- `timing_s/dual_rollout_weight_sync`
- `timing_s/gen_clean`, `timing_s/gen_masked`
- `timing_s/mlp_prefix_cache_reset_before_clean|between_routes|after_masked`
- rollout/actor route switch
- clean/masked actor forward-backward
- saliency accumulate、all-reduce、top-k selection、总 refresh 和 actor→rollout mask sync
- `timing_s/testing_clean`

GRPO 不使用 critic，因此不存在可正确解释的 `clean_critic/masked_critic`。本 recipe 明确拒绝启用 critic，而不是记录两个虚假的 critic 指标。

## Checkpoint

每个 actor checkpoint 额外保存 `mlp_channel_mask.pt`，内容包括：

- 当前 keep mask；
- 历史 ever-masked bitmap；
- EMA saliency；
- mask version；
- cumulative assignments。

恢复时同时加载 actor controller 和 vLLM controller，下一次 rollout 会检查 batch version 与 worker version 是否一致，不一致则直接报错。

## 启动

```bash
cd /Users/molu/verl-mlp_channels_mask
bash recipe/mlp_channel_mask/grpo_mlp_channel_mask_qwen3-4b_offline.sh
```

随机屏蔽每层 10% channels 的对照实验：

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
