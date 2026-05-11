# DiMoR 项目现状简报 —— 供外部审查

## 背景

项目目标：改进 DGSTA 交通流量预测模型，探索 VQ Router、DeepTrendNet、DelayConv 的增益，并最终形成可发表论文。

代码从原始 DGSTA 基线出发，进行了 7 轮架构改进，通过了完整的消融实验和 VQ Router 机制分析。

## 当前最佳性能 (PeMS08, 3 seeds)

| Seed | Config | @3 MAE | @6 MAE | @12 MAE |
|---|---|---|---|---|
| 1 | Full | 11.894 | 12.406 | 13.228 |
| 0 | Full | 12.117 | 12.570 | 13.306 |
| 2 | Full | 12.200 | 12.682 | 13.413 |
| **Mean±Std** | | **12.070±0.13** | **12.553±0.11** | **13.316±0.08** |

原始 DGSTA 基线: @3=12.082, @6=12.498, @12=13.204

**Full 模型的均值性能与基线几乎无差异。** Seed=1 是最乐观的单次结果（@3=11.894），但 seed 间方差高达 0.3 MAE。

## 消融实验 (均基于 seed=1)

| Config | @3 | @6 | @12 |
|---|---|---|---|
| Full (VQ+Trend+Delay) | 11.894 | 12.406 | 13.228 |
| -VQ Router | 12.176 (+2.4%) | 12.645 (+1.9%) | 13.493 (+2.0%) |
| -DeepTrendNet | 11.912 (+0.2%) | 12.421 (+0.1%) | 13.222 (≈0) |
| -DelayConv | 12.176 (+2.4%) | 12.622 (+1.7%) | 13.348 (+0.9%) |

**注意：消融结论基于 seed=1 单次。** 如果 VQ Router ablation 的 seed 间方差与 Full 类似，则 VQ Router 的"2% 增益"结论不成立。

## 机制分析发现（独立于指标，可能更有价值）

1. **Codebook 多样性健康**：10 个图模板 pairwise cosine similarity 接近 0
2. **Hard routing 严重坍缩**：5/6 层 100% 固定到单一模板，不同层选不同模板
3. **Layer specialization**：VQ Router 实际行为是"层间图专业化"而非"时变图路由"
4. **冻结重放实验**：锁死路由后性能退化 <0.2%，说明推理时动态性贡献极小
5. **Soft/hard routing 脱钩**：Balance loss 能拉平 soft entropy（0.91），但 hard routing 依然坍缩，MAE 退化
6. **DeepTrendNet 推理贡献接近零**：DGSTA attention backbone 已隐式吸收趋势信息
7. **所有后续改进尝试均失败**：软路由、空间化趋势、参数化趋势、conv-enhanced attention、adj_adp 时变、balance loss——无一有效

## 当前困境

- **单 seed 显示的改进被多 seed 均值抹平**
- **已尝试 7+ 种改进方向，0 次成功突破**
- **消融实验基于单 seed，未多 seed 验证**
- **论文定位摇摆：该走"SOTA 改进"还是"机制分析"路线？**

## 项目资产

- 完整代码，所有模块 config-gated，可复现
- 7 轮消融实验 + 完整日志 + checkpoint
- VQ Router 逐层逐时行为分析脚本
- 冻结重放实验（裁决实验）
- 两个文档：DGSTA_EVOLUTION.md + SETUP_NEW_MACHINE.md（LLM 可秒懂）
- GitHub 仓库：zhaozaozhe/DiMoR, 分支 main

## 请求审查

1. 在当前指标现实下，最可行的论文路线是什么？
2. 是否需要补全 -VQ 的多 seed 消融？还是单 seed 表格已够用？
3. 机制分析发现（layer specialization 等）是否独立于指标足以支撑一篇论文？
4. 如果继续实验，最该跑的方向是什么？（跨数据集？static-per-layer graph？训练动力学？）
5. 对当前项目叙事（"DiMoR: Discrete Modal Routing"）的批评和建议
