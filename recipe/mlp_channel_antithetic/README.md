# Antithetic MLP-channel GRPO

完整的研究动机、目标函数、工程不变量、风险和实验验收方案见
[`DESIGN.md`](./DESIGN.md)。本文只保留实现概览与运行方式。

这个 recipe 在每个训练 step 为所有 dense MLP 层采样一条固定方向
`epsilon[l, c] in {-1, +1}`，然后把同一 prompt 的 rollout 平分到两个模型内部路由：

```text
positive: z[l, c] <- (1 + sigma * epsilon[l, c]) * z[l, c]
negative: z[l, c] <- (1 - sigma * epsilon[l, c]) * z[l, c]
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

## 可选：reward 差分驱动的辅助 channel 更新

默认关闭，继续运行原有 antithetic GRPO。关闭时不计算 reward 差分、不安装 optimizer
hook、不复制旧权重，也不执行辅助范数通信或参数写回。`max_update_ratio=0` 或
`learning_rate=0` 同样会禁用这些工作；零 reward 差的 batch 则跳过权重快照和范数通信。

专用开启脚本，默认 `reward_update_ratio=0.05`、`reward_update_lr=1e-3`：

```bash
bash recipe/mlp_channel_antithetic/grpo_mlp_channel_antithetic_qwen3-4b_reward_update_offline.sh
```

也可通过原始启动脚本开启（辅助更新最多为每层 `down_proj` 的 GRPO 实际更新范数的 5%）：

```bash
reward_update_enabled=True reward_update_ratio=0.05 reward_update_lr=1e-3 \
  bash recipe/mlp_channel_antithetic/grpo_mlp_channel_antithetic_qwen3-4b_offline.sh
```

关闭：

```bash
reward_update_enabled=False \
  bash recipe/mlp_channel_antithetic/grpo_mlp_channel_antithetic_qwen3-4b_offline.sh
```

对应 Hydra 配置：

```yaml
actor_rollout_ref:
  mlp_channel_antithetic:
    reward_difference_update:
      enabled: false
      learning_rate: 1.0e-3
      max_update_ratio: 0.05
```

每步仍使用原来的 8+8 条 rollout，不额外生成轨迹。先在每道题内分别平均正负 **原始
reward**，再在完整 prompt batch 上等权平均，得到 `R_plus` 和 `R_minus`；不使用
GRPO advantage，也不按 token 数或 actor rank 分组计算。driver 将同一个 reward 差和
方向版本传给全部 actor rank。

对于宽度为偶数 `m` 的一层，当前每步一个方向（K=1）的计算为：

```text
s = (R_plus - R_minus) / (2 * delta)
d = ((m - 1) / m) * s * epsilon
V = learning_rate * W_before_Adam * d[None, :]
PG = W_after_Adam - W_before_Adam
alpha = min(1, max_update_ratio * ||PG||_F / ||V||_F)
W_next = W_after_Adam + alpha * V
```

`delta` 是生成/训练 dtype 中实际可表示的对称扰动，例如 BF16、名义强度 0.1 时为
0.1015625。`(m-1)/m` 修正层内平衡 Rademacher 方向的协方差，因此小扰动下估计的是
去掉每层统一缩放方向后的投影梯度，仍有有限差分偏差和 reward 采样噪声。

比例是**每层真实参数更新的范数上限**，不是 loss 系数，也不是强制达到的固定比例。
候选更新较小时保留其幅度；该层主更新为零时辅助更新也为零。辅助项在 Adam 之后写入，
不进入 Adam 动量；`gate_proj`、`up_proj` 和其他权重只接受原有 GRPO 更新。

`reward_update_lr=1e-3` 是未经实验调优的启发式默认值，没有特殊理论要求。它把 reward
差分转换成 channel gain 的变化，而主学习率 `lr` 缩放的是 Adam 预处理后的梯度，二者
数值不能直接比较。令 `D=W_before_Adam*diag(d)`，忽略写回舍入时，辅助范数为
`min(reward_update_lr*||D||_F, reward_update_ratio*||PG||_F)`。因此候选步长触及上限后，
继续增大辅助学习率不会再增大该层更新；未触及上限时，它才控制实际幅度。5% 是上限，
并非每一步都必须达到。专用开启脚本中的上述默认值仍可通过环境变量覆盖。

FSDP1/FSDP2 使用本地权重分片，不 all-gather 完整模型；开启时额外保存本 rank 的
`down_proj` 旧权重，并通信逐层标量统计。范数汇总修正复制分片的重复计数。实际写回前
还检查浮点舍入后的更新范数，超限时缩小候选步长，必要时跳过该层，确保实际辅助项也不
超过上限。发生非有限梯度、主 optimizer 跳过更新时，辅助项同样不会执行。

约束：完整权重训练、每个 rollout batch 只有一次 optimizer step、偶数 intermediate
width、FP32/FP64 optimizer master weights，以及相同的 actor/rollout 计算 dtype。
不支持 LoRA 或含多个 Shard placement 的 DTensor 布局；不满足条件时明确报错。

所有诊断位于 `mlp_antithetic/reward_update/`。建议按下表建立 W&B 面板；表中省略这个公共前缀。

| 要判断的问题 | 主要指标 | 怎样解读 |
| --- | --- | --- |
| 两侧 reward 是否有区别？ | `reward_gap_abs`、`prompt_gap_abs_mean` | 前者是 batch 平均 reward 差的绝对值，后者是逐题差的绝对值再平均；两者都小，说明这一批可用的差分较弱。 |
| 各题偏好的方向是否互相抵消？ | `prompt_gap_cancellation`、`prompt_gap_nonzero_fraction` | cancellation 为 `1-abs(mean(gap))/mean(abs(gap))`；接近 1 表示抵消明显。所有逐题差为零时定义为 0，需结合 nonzero_fraction 区分没有差异和抵消。 |
| 胜出方向是否容易被采样改变？ | `split_half_gap_first`、`split_half_gap_second`、`split_half_same_sign` | 在相同 prompt 上将每侧已有 rollout 按原始生成顺序交替分成两份，分别估计 batch reward 差。长期频繁反号，说明方向信号不稳定；不增加 rollout。 |
| 逐题差的均值有多不稳定？ | `reward_gap_standard_error` | `std(prompt_gaps, ddof=1)/sqrt(prompt_count)`；同时包含题目差异和采样噪声，不是单纯的 rollout 噪声估计，也不是因果效应显著性检验。 |
| 辅助学习率是否已经被比例上限限制？ | `raw_ratio`、`clipped_layer_fraction` | raw_ratio 是候选辅助范数/主更新范数；多数层触发截断时，再增大辅助 lr 的作用有限。它的数值允许超过 0.05。 |
| 辅助更新真正用了多少预算？ | `actual_ratio`、`max_layer_ratio`、`budget_utilization` | 前两项都应不超过设定的 ratio；utilization=`actual_ratio/max_update_ratio`，例如 0.6 表示实际使用了 5% 上限中的 60%，即约 3%。 |
| 辅助项在加强主方向，还是改变主方向？ | `aux_main_cosine`、`aux_parallel_ratio`、`aux_orthogonal_ratio` | cosine 正值表示同向，负值表示相反，接近 0 表示正交；parallel_ratio=`actual_ratio*cosine`，orthogonal_ratio=`actual_ratio*sqrt(1-cosine^2)`。正交分量衡量超出主更新方向的改变量。 |
| 有多少层生效或与主更新冲突？ | `active_layer_fraction`、`opposing_layer_fraction` | active_fraction 以全部层为分母；opposing_fraction 以主更新和辅助更新均非零的层为分母，统计内积为负的比例。 |
| 比例偏低是否由精度或跳步造成？ | `rounding_backtrack_fraction`、`skipped_zero_gap`、`optimizer_step_executed`、`applied` | 分别对应浮点舍入触发缩步、reward 差为零、主 optimizer 是否执行、辅助项是否实际写入。 |

**有效性标记必须一起看：** `update_metrics_available=0` 表示跳过了权重快照/范数测量，
此时更新范数、比例和夹角等字段写 0 只是占位，不能解释成测得主更新为零。
`alignment_available=1` 时夹角才有定义。`split_half_available=1` 要求每道题每侧至少
有两条 rollout，并保留原始生成顺序；`split_half_both_nonzero` 区分反号和零信号，
两份均为零不会记作方向一致。standard error 仅在 `reward_gap_standard_error_available=1`
（至少两道题）时有定义。主更新范数为零时，raw_ratio/actual_ratio 也使用 0 占位。

所有更新范数与夹角均在 **down_proj 参数子空间** 内计算；夹角使用实际可表示、经过
舍入和缩步后的辅助更新，而非原始梯度。跨 rank 汇总内积和平方范数，并修正复制分片，
不平均各 rank 的局部 cosine。保留原有一份旧权重分片快照，按需重算单个分片的候选项，
因此不额外保存整套主更新张量；开启时增加少量张量运算与标量统计通信，关闭时这些指标
和运算一并跳过。`time_s` 记录 optimizer hook 的主机墙钟时间，不包含主 Adam 步、生成
或主 backward；它不是精确的 CUDA kernel profiler。

此外仍记录 `reward_positive`、`reward_negative`、`reward_gap`、`prompt_count`、
`effective_strength`、`directional_derivative`、`max_update_ratio`、`main_down_proj_norm`、
`raw_aux_down_proj_norm` 和 `aux_down_proj_norm`。每步方向会刷新，所以 signed reward_gap
的长期均值接近零并不自动表示无信号；优先结合绝对差和 split-half 结果判断。

这些是机制诊断：split-half 同号不能证明方向可泛化，负 cosine 也不自动表示有害。
最终收益仍需要与关闭辅助更新的匹配实验比较 neutral validation 的 reward/pass@k。

开启时默认实验名增加 learning rate 和 ratio 后缀，避免自动恢复到关闭辅助更新的实验。
显式设置 `experiment_name` 会覆盖默认命名。辅助更新没有额外跨步动量或状态，结果直接
进入 actor 权重 checkpoint；恢复时仍须使用所需的开关和超参数。

本地验证：

```bash
./scripts/test-local recipe/mlp_channel_antithetic -q
RUN_REWARD_UPDATE_DISTRIBUTED_TESTS=1 ./scripts/test-local \
  recipe/mlp_channel_antithetic/test_reward_update_distributed.py -q
```

第二条是两进程 CPU Gloo 测试，需要允许本机通信端口；验证真实 FSDP1 flat/original
参数和 DTensor 行分片、列分片、复制布局与未分片实现一致。

第一版有意限制为 synchronous vLLM、rollout DP=1、PP=1、偶数 `rollout.n`、无 critic、
无 reference KL、单 PPO epoch。完整 CUDA 训练需要 Linux GPU 环境；macOS 本地仅验证
控制器、路由分配和 source/config contract。
