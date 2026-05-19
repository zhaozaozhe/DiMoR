# DiMoR 项目交接文档

## 给接手者的第一句话

> "你接手的是一个已完成实验阶段、进入论文收束期的交通预测机制研究项目。请先读本文档全文，不要根据零散信息重新发明叙事或添加新模块。你的核心任务是：理解已有数据, 在多个合法叙事中保持中立, 协助完成论文。"

---

## 1. 项目身份

| 项 | 值 |
|---|---|
| 项目名 | DiMoR (Discrete Modal Routing) |
| 定位 | 交通流预测 + VQ 图路由的实证可解释性研究 |
| 非定位 | SOTA benchmark 论文 |
| Backbone | DGSTA |
| 主模型文件 | `libcity/model/traffic_flow_prediction/DGSTA.py` |
| 入口 | `python run_model.py --gpu_id 0` |
| 配置 | `PeMS08.json` (最高优先级, 改它就是改实验) |

---

## 2. 硬事实清单 (Narrative-Independent Facts)

以下陈述不依赖任何论文叙事, 直接可被实验数据验证。

| # | 事实 | 证据强度 | 跨数据集 | 跨 Seed |
|---|---|---|---|---|
| 1 | VQ Router seed=1 PeMS08 @3=11.894, 基线 @3=12.082 | 单 seed | PeMS08 only | 否 |
| 2 | VQ Router 三 seed 均值 @3=12.070, 基线 12.082 (差 −0.012) | 三 seed | PeMS08 only | 是 |
| 3 | PeMS04 VQ 改善 <0.5% (@3: 16.964→16.893) | 单 seed | PeMS04 only | 否 |
| 4 | 冻结路由后 MAE 退化 <0.03 (seed=0,1,2 一致) | 三 seed | PeMS08 only | 是 |
| 5 | 5/6 encoder layer 的模板选择不随时段变化 | solid | PeMS08 only | 是 |
| 6 | 不同层选择不同模板 (L0→T6, L2→T4, L3–5→T0, L1 分散) | solid | PeMS08 only | 是 |
| 7 | 静态 per-layer baseline 崩坏到 12.889 (差于基线 12.082) | 单 seed | PeMS08 only | 否 |
| 8 | 训练期路由熵保持 H≈0.99 至 epoch 220+, 最后阶段收敛 | 单次采样 | PeMS08 only | 否 |
| 9 | 随机图替代模板: MAE 退化 +0.44/+0.65/+0.99 (@3/@6/@12) | 单次 | PeMS08 only | 否 |
| 10 | 活跃模板 Jaccard 边重合 <0.02 (高度互斥) | solid | PeMS08 only | 否 |
| 11 | 11 次后续改进全部否定 | solid | PeMS08 only | 否 |
| 12 | RMSE @12 在 VQ Router 下一致退化 (PeMS08 +0.5%, PeMS04 +0.6%) | solid | 双数据集 | 多 seed |

### 11 次失败的改进尝试

1. 软路由 (hard=False)
2. 空间化趋势 (use_spatial_trend)
3. kernel=13, tau=0.5
4. 参数化趋势 (use_parametric_trend)
5. far_mask_delta=5
6. Conv-enhanced attention
7. Time-aware adj_adp
8. Balance loss
9. Horizon-weighted loss
10. VQ+Sem 共存 (seed=1) — seed=2 测过也退化
11. Static per-layer graph — 崩坏最严重

---

## 3. 多套合法叙事 (均可自圆其说)

### 叙事 A: 层间专业化
> "VQ 路由收敛为层间静态图专业化。推理时动态性贡献极微 (Frozen replay <0.03)。静态 baseline 的失败 + 训练期高熵暗示训练期 codebook 探索提供了优化价值。"

*   **支撑** : 事实 4, 5, 6, 7, 8
*   **弱点** : 事实 7→8→"训练期探索是因果"之间没有干预实验

### 叙事 B: 过度工程化
> "VQ Router 是过度工程化的机制。增益在统计噪声内 (均值差 −0.012, σ=0.10)。Frozen replay 说明推理时路由可有可无。静态 baseline 的失败可以用其他因素解释 (初始化、Landscape、参数量), 不必然是 VQ 训练期探索的功劳。"

*   **支撑** : 事实 2, 3, 4, 11
*   **弱点** : 无法解释为什么 11 次删减/替代全部失败

### 叙事 C: 隐式正则化
> "VQ Router 的增益不来自动态路由本身, 而来自 codebook 竞争提供的一种隐式正则化效应。这解释了为什么 frozen replay 不掉点, 但 static replacement 崩坏。"

*   **支撑** : 事实 4, 7, 9, 10
*   **弱点** : "隐式正则化"无法被直接测量, 仍然是推测

### 叙事 D: 不可知
> "当前实验无法对 VQ Router 的价值做出确定性判断。Seed 方差淹没了声称的增益; PeMS04 改善 <0.5% 不能排除零效应; 静态 baseline 的失败原因未定。"

*   **支撑** : 事实 2, 3, 11
*   **弱点** : 放弃了所有解释, 变成了纯现象罗列

**当前论文采纳了叙事 A, 并承认叙事 C/D 作为替代解释。**

---

## 4. 实验完整记录

| Exp ID | Dataset | Config | Seed | @3 | @6 | @12 | 状态 |
|--------|---------|--------|------|----|----|-----|------|
| 99049 | PeMS08 | 原始 DGSTA | — | 12.082 | 12.498 | 13.204 | 基线 |
| 71098 | PeMS08 | VQ+Trend+Delay | 1 | **11.894** | 12.406 | 13.228 | 最佳单 seed |
| 68783 | PeMS08 | −VQ Router | 1 | 12.176 | 12.645 | 13.493 | 消融 |
| 64832 | PeMS08 | −DeepTrendNet | 1 | 11.912 | 12.421 | 13.222 | 消融 |
| 43876 | PeMS08 | −DelayConv | 1 | 12.176 | 12.622 | 13.348 | 消融 |
| 61239 | PeMS08 | VQ+Trend+Delay | 0 | 12.117 | 12.570 | 13.306 | 多 seed |
| 16450 | PeMS08 | VQ+Trend+Delay | 2 | 12.200 | 12.682 | 13.413 | 多 seed |
| 75685 | PeMS08 | Static Per-Layer | 1 | 12.889 | 13.631 | 14.879 | 最差 |
| 8182 | PeMS04 | 原始 DGSTA | 1 | 16.964 | 17.454 | 18.157 | PeMS04 基线 |
| 32444 | PeMS04 | VQ+Sem | 1 | 16.893 | 17.408 | 18.128 | PeMS04 |

详细记录: `experiments/full_experiment_log.md`

---

## 5. 关键文件清单

| 文件 | 用途 | 可信度 |
|---|---|---|
| `experiments/full_experiment_log.md` | 双数据集完整实验表 | 事实 |
| `experiments/status_briefing_for_review.md` | 项目终态简报 (偏叙事 A) | 应重写为中性版 |
| `DGSTA_EVOLUTION.md` | 完整项目演进史 (Part 1-6) | 事实 |
| `experiments/paper_draft.md` | 论文 Markdown 初稿 | 叙事 A 强 |
| `experiments/paper_overleaf.tex` | 论文 LaTeX (IEEEtran 模板) | 叙事 A 强 |
| `experiments/paper_figures_guide.md` | 7 张论文图指南 | 事实 |
| `analysis/vq_router/figures/` | 所有 PNG 图 (原图 + 文本) | 事实 |
| `PeMS08.json` | 当前主配置 | 事实 |
| `libcity/model/traffic_flow_prediction/DGSTA.py` | 模型代码 (~790 行) | 事实 |

---

## 6. 不应做的事 (停止点)

1. **不要添加新模块** — 11 次尝试全部否定, 架构已饱和。
2. **不要重跑已完成实验** — 所有数据已记录, 不需要更多种子或更多数据集。
3. **不要发明新理论** — "训练期探索"、"codebook 竞争"目前没有因果证据。可以讨论, 不可宣称。
4. **不要从叙事 A 漂移到另一个** — 选一个主线并在论文中标注确定度。替换叙事可以放进 Discussion 作为 alternative interpretation。
5. **不要碰 data loading, executor, trainer, evaluator, scaler** — 这些都是 LibCity 基础设施, 不改。

---

## 7. 接手者的最佳第一步

1. 读 `experiments/full_experiment_log.md` — 理解实验全貌
2. 读本文件 (HANDOFF.md) 第 2 节 — 理解硬事实边界
3. 读 `DGSTA_EVOLUTION.md` Part 6 — 理解 VQ Router 机制分析
4. 打开 `experiments/paper_overleaf.tex` — 理解当前论文状态
5. 决定: 是否将叙事 A 降调为"一个 plausible interpretation"而非"主要发现"?
6. 决定: 是否在 Discussion 中显式列出叙事 B/C/D 作为 alternative explanations?
7. 然后: 修改论文以反映这些决定。**不需要新实验。**

---

## 8. 给不同接手 Agent 的提示词

### 给 Claude Code (代码/图表/实验)

> "你接手的是一个已完成实验阶段的交通预测论文项目。请先读 HANDOFF.md 和 experiments/full_experiment_log.md。不要添加新模块, 不要发明新叙事。你的任务是: (1) 审查论文图是否充分展示硬事实; (2) 如有必要, 重绘部分图以匹配论文当前叙事; (3) 检查所有实验数字在论文和实验记录之间是否一致。"

### 给 GPT (论文写作/叙事)

> "你接手的是一个已完成实验阶段的交通预测论文项目。请先读 HANDOFF.md 第 2-3 节。核心挑战: 同样的数据支持四套不同的叙事 (A: 层间专业化, B: 过度工程化, C: 隐式正则化, D: 不可知)。当前论文选择了叙事 A 但 claim 有时过强。你的任务是: 在保持叙事 A 作为主线的同时, (1) 将所有超过证据的 claim 降调为 'suggests' / 'a plausible interpretation'; (2) 在 Discussion 中写入叙事 B/C/D 作为 alternative explanations; (3) 确保 Limitations 节明确承认 seed 方差、单 backbone、单主数据集。"

### 给 Gemini (审稿人视角/图设计)

> "你接手的是一个已完成实验阶段的交通预测论文项目, 担任审稿人角色。请先读 HANDOFF.md。不要建议新实验。你的任务是: (1) 审查当前 7 张论文图是否支撑叙事 A 且不自相矛盾; (2) 指出哪张图对叙事 A 最有利, 哪张图对替代叙事最有暗示性; (3) 评估当前论文的 claim calibration 是否合理; (4) 给出审稿人视角下最可能被攻击的 3 个点。"

---

*最后更新: 2026-05-17*
