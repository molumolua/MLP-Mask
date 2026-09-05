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

每层独立、无放回采样恰好 `round(mask_ratio * d_ff)` 个 channel。默认每个 optimizer
step 重采样，每个 actor data-parallel rank 由相同的 `random_seed + mask_version`
确定性地产生相同 mask。没有 `1 / (1-p)` inverted-dropout scaling。

mask 不安装到 vLLM，因而 generation 和 validation 永远不会意外进入 masked route。

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

常用覆盖：

```bash
mask_ratio=0.10 \
kl_coef=0.01 \
kl_top_k=64 \
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
- `mlp_consistency/response_tokens`；
- `mlp_consistency/mask_version`；
- `mlp_consistency/realized_mask_fraction`；
- `mlp_consistency/masked_per_layer_min|max`；
- `mlp_consistency/hard_mask=1`；
- `mlp_consistency/inverted_dropout_scaling=0`。

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
