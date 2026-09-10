# Qwen3-4B resource estimate

以下为容量规划，不是本机 CUDA 实测。假设约 4.0B 参数、词表 151936、每个 prompt
16 条回答、最长 prompt 8192 token、最长 response 4096 token；辅助子批次每 GPU 为 1。
实际开销还取决于有效序列长度、生成吞吐、软件版本、checkpoint 和 offload 设置。

## KL 分布缓存

| 内容 | 完整词表 | top-64 + tail（默认） |
| --- | ---: | ---: |
| 一个 4096-token response 的 teacher FP32 分布 | 2.32 GiB | 1.02 MiB |
| teacher 分布及其 token IDs | 2.32 GiB | 3.02 MiB |

完整词表单份计算式为 `4096 * 151936 * 4 bytes`。压缩版本单份概率为
`4096 * 65 * 4 bytes`，teacher 选择的 int64 token IDs 为 `4096 * 64 * 8 bytes`。
当前版本只缓存生成路由的 detached teacher 分布，student 分布按 token chunk 临时计算。

**缓存大小由主损失 micro-batch 决定，不能只看辅助 micro-batch=1。** teacher 复用 PPO
前向，因此保存该主损失 micro-batch 的全部有效 response tokens。设总数为 `T`，
完整词表缓存约 `T * 151936 * 4 bytes`，top-64 缓存（含 IDs）约 `T * 772 bytes`。
例如主损失一次处理两条 4096-token 回答，分别约 4.64 GiB 或 6.03 MiB。
辅助子批次只切片复用这些缓存，不复制整个分布；下一个主损失 micro-batch 前释放。
可用 `mlp_rdrop/teacher_cache_peak_tokens` 观察实际 `T` 的峰值。

top-k 降低跨 forward 保存的分布内存，但仍需正常模型 forward 和词表归一化。
例如 12288-token 的单条序列，未经 response 切片的 BF16 logits 本身约 **3.48 GiB**；
4096 response token 对应的一份 BF16 logits/梯度约 **1.16 GiB**。
这些分配以及模型激活不会因 top-k 变成 65 类而全部消失。KL token chunk 默认 128，
每块 `128 * 151936` 个 FP32 元素约 74 MiB，临时使用数份这样的块。

## GPU 与 CPU

按 FSDP2、4 GPUs、完整权重 AdamW 训练估计：若 FP32 原始参数/梯度各 4 bytes，
Adam 一二阶状态共 8 bytes，持久训练状态总计约 `16 * 4e9 bytes`，均分后约
**14.9 GiB/GPU**。BF16 compute 权重、层级 all-gather buffer、activations、logits、
梯度临时张量及 CUDA/NCCL workspace 需要额外空间；该数字不能当作峰值显存。

默认 rollout TP=1 时，每个 vLLM 副本的 BF16 模型约 **7.45 GiB**，另外还有 KV cache。
`gpu_memory_utilization=0.7` 是 vLLM 阶段的内存预算设置。现有 worker 在 rollout/actor
间切换并管理 vLLM sleep/wake；不要将两个阶段的所有峰值直接相加，也不要假设
所有分配都能被释放到零。

建议沿用 **4 × 80 GB GPU** 作为第一轮配置。4 × 48 GB 可以从更短序列、主/辅助
micro-batch=1 开始试，但本文不保证默认最长序列能放下。24 GB 卡不作为当前默认
长序列全参数实验的容量承诺。最终用一次 warmup 和若干真实训练 step 的峰值显存确认。

参数更新诊断额外占用总计约 **7.45 GiB CPU RAM**，4 rank 时每 rank 约 **1.86 GiB**。
这份 reference 与 KL 分布缓存无关。每次 validation 转换当前 shard、计算差值还需要
临时 CPU 张量，按参数 tensor 逐个处理；不会常驻另一个完整 FP32 模型副本。
Ray、tokenizer、数据缓存和初始化峰值另外计入。4 GPU 主机可按 **64 GiB RAM 起步、
128 GiB 更宽裕、16 个 CPU cores 左右**做初步规划；开启参数/optimizer offload 后
CPU RAM 和带宽需求会明显增加。上述 CPU 配置是工程建议，不是测得的最低要求。

## 时间

令 `F` 是对同样 token 数的一次 actor forward，近似一次 backward 为 `2F`：

| 更新方式 | 每批对应的 actor 计算量粗估 |
| --- | ---: |
| 普通 GRPO | `F + 2F = 3F` |
| 原 clean→masked consistency | 主损失 `3F` + masked `3F` = `6F` |
| 旧版逐轨迹双侧对称 KL | 主损失 `3F` + no-grad A `F` + B `3F` + A replay `3F` = `10F` |
| 当前交叉 KL | 主损失 `3F`（复用 teacher 分布）+ 对侧 student `3F` = `6F` |

当前版本每条轨迹仅额外执行一次对侧 forward 和一次 backward：A 的 8 条让 B 学，
B 的 8 条让 A 学。actor 更新计算量约为普通 GRPO 的 **2 倍**，与原单向 consistency
大致同阶；相对旧版 `10F`，约减少 **40%**。单看辅助模型计算，则由 `7F` 降为 `3F`，
约减少 **57%**。teacher top-k 缓存仍有额外开销，只是无需额外模型前向。
这是 FLOP 级近似：checkpoint 重算、小 batch 利用率、词表 KL、FSDP 通信、动态
批次的零权重重放都会影响真实时间，不能直接视为实测倍数。

如果普通 GRPO 中 actor update 占单步时间比例 `f`，假设其它部分时间不变，粗估：

```text
new_step_time / baseline_step_time ≈ 1 + (2 - 1) * f
f=20% → 1.20x
f=40% → 1.40x
```

例如仅作为换算示例，原来 60 秒/step 且 `f` 在上述区间，对应约 72–84 秒/step；
这不是本项目测得的绝对时长。生成仍是 16 条，但拆成 A/B 两次调用以及 cache reset
可能改变吞吐。top-k 节省分布缓存，交叉目标及 teacher 复用减少模型计算；两者作用不同。

在目标 GPU 主机上，比较 clean control、no-aux 两 mask control 和默认 recipe 的
`timing_s/step`、`timing_s/update_actor`、`timing_s/mlp_rdrop_auxiliary_step` 和
`timing_s/mlp_rdrop_teacher_capture_step`。最后一项单独记录 PPO 前向内的 teacher
分布缓存构建；这部分从旧版辅助阶段移到了主损失前向，比较时不能遗漏。
阶段指标为主机 wall-clock 耗时，受 CUDA 异步执行影响；整体时间以 trainer 的
step/update_actor 为准，避开初始化/编译 warmup，再据实测的秒数规划训练时长。
