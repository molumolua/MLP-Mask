# MLP-channel historical relative-update GRPO

这个 recipe 保持标准 GRPO 的题目权重和 actor loss 不变，只重新分配 dense
SwiGLU MLP channel 之间的 AdamW update budget。历史上实际相对更新较少的
channel 获得更大的 update multiplier，历史更新较多的 channel 获得更小的
multiplier。

## 参数分组

第 `l` 层第 `c` 个完整 channel 定义为：

```text
Theta[l,c] = concat(
    gate_proj.weight[c, :],
    up_proj.weight[c, :],
    down_proj.weight[:, c],
)
```

三部分使用同一个 multiplier。recipe 要求 FSDP2，因为 FSDP1 默认把这些矩阵
展平成内部 `FlatParameter`，无法通过稳定的公共接口保留 channel 行/列边界。
当前版本还要求 Ulysses SP=1、关闭 FSDP2 offload，并且不支持 LoRA。

## 历史相对更新

普通 GRPO backward 和全局 gradient clipping 完成后，AdamW 首先更新一阶、二阶
矩估计，并产生尚未应用的基础 update：

```text
u[t] = m_hat[t] / (sqrt(v_hat[t]) + adam_eps)
delta_base[t] = -lr[t] * u[t]
```

对每个 channel 计算：

```text
base_relative_sq[l,c,t]
    = sum(delta_base[l,c,t]^2)
      / (sum(Theta[l,c,t]^2) + parameter_count * parameter_rms_epsilon^2)
```

controller 保存实际应用后的相对 update energy EMA：

```text
H[l,c,t] = beta * H[l,c,t-1]
           + (1-beta) * actual_relative_sq[l,c,t]

history_rms[l,c,t] = sqrt(H[l,c,t] / (1-beta^t))
```

本步 multiplier 只读取 `t-1` 时刻的历史，避免当前 update 提前进入自己的基线。
每层使用 channel 历史中位数作为参考：

```text
floor[l] = max(history_floor_ratio * median_c(history_rms[l,c]), eps)

raw_multiplier[l,c]
    = ((median_c(history_rms[l,c]) + floor[l])
       / (history_rms[l,c] + floor[l])) ^ history_power
```

前 `warmup_steps` 步只收集普通 AdamW 更新历史，multiplier 恒为 1。

## 十倍比例和固定 update budget

`multiplier_ratio_cap=10` 对应对数对称边界：

```text
min_multiplier = 1 / sqrt(10)
max_multiplier = sqrt(10)
```

所以任意两个 channel 的 multiplier 比值最多为 10。十倍是上限，不是每一步强制
达到的目标；历史分化不足或者固定能量约束生效时，实际比值会更小。

对 raw multiplier 乘一个公共标量并裁剪，通过二分求解使：

```text
sum_c(multiplier[c]^2 * sum(delta_base[c]^2))
    = sum_c(sum(delta_base[c]^2))
```

因此 MLP 总 squared update norm 与普通 AdamW 相同。非 MLP 参数保持普通 AdamW
更新，decoupled weight decay 也不参与 multiplier 或历史统计。当前 launcher 设置
`weight_decay=0`。

## 启动

默认十倍上限实验：

```bash
bash recipe/mlp_channel_relative_update/grpo_mlp_channel_relative_update_qwen3-4b_offline.sh
```

完全关闭组件并使用原生 `torch.optim.AdamW` 的标准 FSDP2 baseline：

```bash
bash recipe/mlp_channel_relative_update/grpo_mlp_channel_relative_update_qwen3-4b_standard_adamw_offline.sh
```

使用相同自定义优化器但强制 multiplier=1 的 matched control：

```bash
bash recipe/mlp_channel_relative_update/grpo_mlp_channel_relative_update_qwen3-4b_control_offline.sh
```

常用覆盖：

```bash
history_ema_beta=0.99 \
history_power=0.5 \
history_floor_ratio=0.1 \
multiplier_ratio_cap=10 \
warmup_steps=16 \
bash recipe/mlp_channel_relative_update/grpo_mlp_channel_relative_update_qwen3-4b_offline.sh
```

controller 的历史随 checkpoint 保存为 `mlp_channel_relative_update.pt`。恢复训练时
配置必须与 checkpoint 中的统计配置完全一致。

## 实验设计

第一阶段至少运行以下三组，并保持模型、数据顺序、rollout seed、温度、batch、
learning rate、gradient clipping、验证频率和 token budget 一致：

1. 标准 PyTorch AdamW GRPO baseline。
2. `multiplier_ratio_cap=1` matched control，用于排除自定义 AdamW 实现和统计开销。
3. `multiplier_ratio_cap=10` relative-update treatment。

如果第 1、2 组不一致，应先解决数值或系统差异，不能把第 3 组结果归因于 channel
allocation。每组最好至少使用 3 个 seed；主要横轴使用已消费的 unique prompts 或
rollout tokens，而不只是 optimizer step 和 wall time。

第二阶段再比较：

- `ratio_cap`: 2、4、10；
- `history_power`: 0.25、0.5、1.0；
- `history_ema_beta`: 0.95、0.99、0.999；
- MLP-only relative update 与现有 question-rarity weighting。

第一阶段不要同时打开 rarity loss weighting，否则无法区分“改变目标”和“改变优化
路径”的贡献。

## 重点 metrics

算法是否按设计工作：

- `mlp_relative_update/update_energy_ratio`：应接近 1；
- `mlp_relative_update/update_energy_error`：应接近 0；
- `mlp_relative_update/multiplier_min/max/max_to_min`；
- `mlp_relative_update/low_history_to_high_history_multiplier_ratio`；
- `mlp_relative_update/boosted_fraction`、`damped_fraction`；
- `mlp_relative_update/min_saturation_fraction`、`max_saturation_fraction`；
- `mlp_relative_update/energy_share_shift_to_low_history`：应为正，表示预算确实转移；
- `mlp_relative_update/base_effective_channel_fraction` 与
  `actual_effective_channel_fraction`：观察 update energy 是否覆盖更多 channel；
- `mlp_relative_update/log_history_multiplier_correlation`：warmup 后应为负。

优化稳定性：

- `actor/grad_norm`；
- actor learning rate；
- policy KL / old-policy log-prob difference；
- PPO clip fraction；
- entropy；
- NaN、skipped update、OOM 和每 step wall time。

训练是否真的更有效：

- train reward/accuracy 相对于 unique prompts 或 rollout tokens 的曲线；
- 各验证集 accuracy，而不只看训练集；
- 达到固定验证阈值所需的 prompts/tokens；
- 最佳验证值以及后期是否回落；
- 不同 seed 的均值、方差和失败率。

## 逻辑边界

这个指标衡量“优化器实际移动过多少”，不是“channel 离最优还有多远”。预期收益
来自把固定更新预算从反复移动的 channel 转移给历史移动较少但当前仍有梯度的
channel。它有可能提高有限数据下的覆盖速度，但也可能放大罕见、噪声较大的 RL
梯度。若 treatment 出现更高 KL、clip fraction、seed 方差或后期性能回落，应优先
降低 `ratio_cap`/`history_power`，或者在下一版加入 gradient agreement/SNR 门控，
而不是继续放大比例。
