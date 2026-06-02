# KML-ARIS Auto Research Plan: LinearGM Dataset Distillation for Class-Incremental Learning

## 1. Goal

在当前 `featmatch` 仓库的 MACIL 基础上，完成一个自动科研闭环：

- 研究主题：基于梯度匹配的数据集蒸馏 + 类增量学习。
- 目标数据集：ImageNet-A 10 tasks, 即本仓库 `exps/imga10.json` 对应的 IMA-10 设置。
- 核心目标：在 IMA-10 上超过参考原文 MACIL 的 ImageNet-A 10-task 结果。
- 参考基线：`drift_cil.pdf` 中 ImageNet-A 10-task 结果，MACIL/Ours 为 `ALast=64.14±0.58`, `AAvg=71.45±1.35`。
- 验收标准：三 seed 平均值同时满足 `ALast > 64.14` 且 `AAvg > 71.45`，并提供可复现实验日志、配置、结果表和 claim ledger。

## 2. Literature Constraints

### 2.1 Linear Gradient Matching

`distill.pdf` 提出 Linear Gradient Matching: 对预训练特征提取器上的随机线性分类头，匹配真实数据与合成数据诱导的分类梯度。关键实现约束：

- 随机线性 probe 每轮重新采样，降低固定头过拟合。
- 匹配对象优先使用线性分类头梯度或 feature-gradient 的低维近似。
- 使用 differentiable augmentation，多视图合成批次提升泛化。
- 使用图像正则，如 TV/L2，以及更强的 multi-scale/pyramid 表示作为第二阶段创新。
- 1 image per class 是强压缩设置，但 CIL 中可扩展到 1/2/4 IPC 做效率-精度曲线。

### 2.2 Semantic Drift Calibration

`drift_cil.pdf` 的 MACIL 机制已经是本仓库主体：

- ViT-B/16-IN21K frozen backbone + task-specific LoRA。
- 分类损失使用 angular penalty。
- 对旧类执行 mean shift compensation 和 covariance calibration。
- patch-token feature self-distillation 抑制遗忘。
- ImageNet-A/CUB/ImageNet-R 按 10 tasks, 20 classes/task 设置。

### 2.3 ARIS Harness

ARIS 论文 `https://huggingface.co/papers/2605.03042` 的可落地约束：

- 使用执行层、编排层、保证层三层结构。
- 长周期科研不信任单 agent，采用跨角色 adversarial review。
- 每个中间产物必须进入 persistent research state。
- 实验结论必须经过 integrity verification, result-to-claim mapping, claim auditing。
- Auto Review Loop 每轮输出 action items，Producer 修复，Critic 给出 verdict。

## 3. Current Repository Baseline

当前仓库已经具备以下基础：

- `methods/macil.py`: MACIL 主实现，包含 replay loader、influence replay sampling、replay gradient guidance、LinearGM replay generation/refinement、set-level LinearGM。
- `trainer.py`: 记录 per-task top1 curve、accuracy matrix、Average Accuracy、Last Accuracy。
- `exps/imga10.json`: ImageNet-A 10-task 原始目标配置。
- `exps/imgr5.json`: 已加入 set-level GM 的可参考配置。

因此本项目不重写 CIL 框架，而是在现有 MACIL 上做三类最小侵入增强：

1. Dataset distillation replay: 让每个任务结束后生成/刷新旧类蒸馏图像。
2. Set-level LinearGM: 按类组共同优化 replay set，使 batch gradient 匹配真实类组。
3. Drift-aware replay: replay 目标结合 MACIL 的 mean/cov/patch-token 漂移约束。

## 4. Proposed Innovation

方法名建议：`Drift-Aware Set LinearGM Replay (DAS-LGM)`

核心创新点：

- Set-level gradient matching: 不再逐类单图像优化，而是按 `replay_set_size` 个类组成 balanced mini-set，匹配真实 batch 的线性 probe 梯度。
- Drift-aware target refresh: 对旧类 replay 图像，用旧模型特征和新模型边界共同约束，降低旧类语义漂移。
- Prototype-Mahalanobis anchoring: replay 生成时加入类均值和协方差约束，避免合成样本只贴近新分类头而偏离旧类分布。
- Influence-guided replay sampling: 训练时优先采样与新任务梯度冲突更大的旧类 replay 图像。
- Evidence-driven hyperparameter search: 8xH200 上并行进行 staged ablation，Critic 只允许基于日志和表格接受 claim。

## 5. Three-Agent Workflow

三个 KML agent 对应 ARIS 的 executor/reviewer/harness roles：

- `KML CEO`: 总控与研究账本维护。负责目标拆解、轮次调度、claim ledger、资源分配、最终决策。
- `KML Producer`: 实现与实验执行。负责代码修改、配置生成、训练脚本、GPU 任务提交、日志解析、表格生成。
- `KML Critic`: 独立审查。负责代码 review、实验完整性检查、结果到 claim 映射、消融充分性、是否超过原文的 verdict。

每一轮固定输出：

- Producer artifact: patch/config/log/table。
- Critic verdict: `pass`, `revise`, or `reject`。
- CEO decision: `advance`, `rerun`, `rollback`, or `freeze`.

## 6. 8xH200 Execution Plan

### Phase 0: Environment and Data

目标：在 8xH200 机器上跑通单 seed 单 task sanity check。

Commands:

```bash
conda create -n featmatch python=3.9 -y
conda activate featmatch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm numpy scipy pillow tqdm matplotlib pandas
git clone https://github.com/yanchengli3372637603-art/featmatch.git
cd featmatch
```

数据目录要求：

```text
ImageNet-A/
  train/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

Sanity command:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config exps/imga10_kml_lgm_cil.json --device 0
```

### Phase 1: Baseline Reproduction

目标：复现原文 IMA-10。

Runs:

- `exps/imga10.json`, seeds `1993,1996,1997`。
- 指标：`ALast`, `AAvg`, forgetting, per-task curve。
- Critic 检查：class order、seed、data split、backbone、epoch、batch size 是否与论文一致。

Gate:

- 若 baseline 与 `64.14/71.45` 差距超过 1.5 点，先修环境和数据，不进入创新实验。

### Phase 2: DAS-LGM Main Run

目标：运行目标配置 `exps/imga10_kml_lgm_cil.json`。

关键开关：

- `enable_replay=true`
- `generate_replay=true`
- `replay_gradmatch=true`
- `replay_set_gm=true`
- `replay_gm_views=2`
- `replay_gm_num_heads=4`
- `replay_proto_lambda=0.15`
- `replay_teacher_kd=true`
- `train_global_ce=true`
- `lora_ortho_lambda=0.05`

8xH200 调度：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config exps/imga10.json --device 0
CUDA_VISIBLE_DEVICES=1 python main.py --config exps/imga10_kml_lgm_cil.json --device 0
CUDA_VISIBLE_DEVICES=2 python main.py --config exps/imga10_kml_lgm_cil_ipc2.json --device 0
CUDA_VISIBLE_DEVICES=3 python main.py --config exps/imga10_kml_lgm_cil_set16.json --device 0
CUDA_VISIBLE_DEVICES=4 python main.py --config exps/imga10_kml_lgm_cil_noset.json --device 0
CUDA_VISIBLE_DEVICES=5 python main.py --config exps/imga10_kml_lgm_cil_noproto.json --device 0
CUDA_VISIBLE_DEVICES=6 python main.py --config exps/imga10_kml_lgm_cil_nokd.json --device 0
CUDA_VISIBLE_DEVICES=7 python main.py --config exps/imga10_kml_lgm_cil_fast.json --device 0
```

### Phase 3: Ablation and Claim Audit

最小消融矩阵：

| Run | Purpose |
| --- | --- |
| MACIL baseline | 原文复现 |
| DAS-LGM full | 主方法 |
| no set-level GM | 证明 batch-level gradient matching 有效 |
| no prototype anchor | 证明漂移约束有效 |
| no teacher KD | 证明旧模型分布约束有效 |
| no influence sampling | 证明训练期 replay 采样有效 |
| IPC=2/4 | 画压缩率-精度曲线 |

Claim ledger 必须记录：

- claim 文本。
- 支撑日志路径。
- 支撑表格路径。
- seed 列表。
- 反例或失败运行。
- Critic verdict。

### Phase 4: Close Loop

闭环定义：

1. 环境配置可复现。
2. 代码和配置进入 GitHub。
3. 三 seed 实验完成。
4. 结果表自动生成。
5. Critic 完成 evidence audit。
6. 达到 `ALast > 64.14` 和 `AAvg > 71.45`。
7. 生成 final report，包括方法、实现差异、主表、消融、限制。

## 7. Implementation Backlog

P0:

- 添加 `exps/imga10_kml_lgm_cil.json`。
- 添加日志解析脚本，自动抽取 `Average Accuracy`, `Last Accuracy`, `Accuracy Matrix`。
- 添加 evidence ledger 模板。

P1:

- 给 set-level GM 增加 pyramid/image-parameterization 选项，复刻 LinearGM 的隐式正则。
- 增加 replay image visualization grid。
- 增加 per-task replay quality probe: 用 replay-only head 训练并在当前 test set 上评估。

P2:

- 支持 multi-backbone evaluator，如 DINOv2/CLIP 特征 probe，用于证明蒸馏 replay 不只过拟合 MACIL backbone。
- 自动生成论文图表和 LaTeX 表格。

## 8. Risk Register

| Risk | Mitigation |
| --- | --- |
| Git HTTP 被网络重置 | 使用 GitHub API 或服务器端直接 push |
| ImageNet-A 类不均衡导致 replay 真实目标噪声大 | `replay_set_real_per_class=2`, EMA target, influence sampling |
| 合成 replay 图像过拟合分类头 | random probe reinit, multi-head, multi-view augmentation, TV/L2/pyramid |
| 8xH200 显存浪费 | 单 GPU 多 seed/多 config 并行，不做 DDP |
| 超不过原文 | 按 ablation ledger 定位瓶颈，优先调 `replay_lambda`, `replay_gradmatch_lambda`, `replay_set_size`, `replay_ipc`, `train_global_ce` |

## 9. Success Report Template

```text
Method: DAS-LGM
Dataset: ImageNet-A 10 tasks
Backbone: ViT-B/16-IN21K
Seeds: 1993, 1996, 1997

Baseline MACIL:
  ALast:
  AAvg:

DAS-LGM:
  ALast:
  AAvg:

Delta:
  ALast:
  AAvg:

Critic verdict:
Evidence:
```

