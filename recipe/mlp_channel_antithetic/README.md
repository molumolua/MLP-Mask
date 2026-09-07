# Antithetic MLP-channel GRPO

这个 recipe 在每个训练 step 为所有 dense MLP 层采样一条固定方向
`epsilon[l, c] in {-1, +1}`，然后把同一 prompt 的 rollout 平分到两个模型内部路由：

```text
positive: h[l, c] <- (1 + sigma * epsilon[l, c]) * h[l, c]
negative: h[l, c] <- (1 - sigma * epsilon[l, c]) * h[l, c]
```

`0 < sigma < 1`，所以没有通道被屏蔽。两个 gain 逐通道的平均值严格为 1。
在局部展开下，正负两侧的一阶扰动项互相抵消；训练目标因此是 clean policy
附近的对称平滑目标，而不是把另一些神经元错误地当作不存在。它并不声称得到完全无偏的
clean-policy 梯度：二阶平滑项以及 PPO/GRPO 本身的 surrogate 仍然存在。

运行时会先把 `sigma` 量化为模型 dtype 中可在 1 两侧对称表示的 `delta`，再使用
`1 +/- delta`。默认配置 `sigma=0.10` 的名义 gain 范围是 `0.90` 到 `1.10`；
bfloat16 的实际 `delta` 是 `0.1015625`，实际 gain 是 `0.8984375` 和 `1.1015625`。
这样避免分别 cast `0.90/1.10` 导致二者平均值偏离 1。该值会记录在
`mlp_antithetic/effective_strength_bfloat16`。

## 训练数据流

默认总预算仍为每个 prompt 16 条：8 条 positive，8 条 negative。实现使用两次顺序
vLLM generation，但只做一次 actor-to-rollout 权重同步。两次 generation 之间必须清空
prefix cache，因为 cache key 不包含 MLP 路由。

生成结果恢复到原来的行顺序，且所有 16 条轨迹保留相同的原始 `uid`。因此 GRPO advantage
在完整的 16 条轨迹上统一计算；`route_id` 只用于让 generation、old-logprob 和 actor
forward/backward 选择完全相同的 gain。actor DP balance 也会保证每个 rank 收到相等数量
的 positive/negative 样本。

验证始终使用 neutral 路由（所有 gain 都是 1），所以 pass@k 衡量的是未扰动模型。

## 运行

```bash
bash recipe/mlp_channel_antithetic/grpo_mlp_channel_antithetic_qwen3-4b_offline.sh
```

常用覆盖：

```bash
perturbation_strength=0.02 random_seed=7 n_total=16 \
  bash recipe/mlp_channel_antithetic/grpo_mlp_channel_antithetic_qwen3-4b_offline.sh
```

第一版有意限制为 synchronous vLLM、rollout DP=1、PP=1、偶数 `rollout.n`、无 critic、
无 reference KL、单 PPO epoch。完整 CUDA 训练需要 Linux GPU 环境；macOS 本地仅验证
控制器、路由分配和 source/config contract。
