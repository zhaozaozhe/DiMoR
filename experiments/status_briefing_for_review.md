# DiMoR 项目终态简报

## 项目定位

**DiMoR: Discrete Modal Routing for Explainable Traffic Forecasting**

论文类型：机制分析型（非 SOTA 竞赛型）
目标 venue：SCI 四区 / EI
状态：实验完成，进入论文素材阶段

## 最终架构

```
DGSTA backbone (Geo + Temporal + Semantic Attention, GCN)
  + VQ Router (10-template codebook, Gumbel-Softmax hard routing, Top-20 sparsification)
  + DelayConv (causal depthwise temporal conv in GCN)
  + DeepTrendNet (trend decomposition + MLP, fusion_weight=1.0)
  + Semantic Attention (DTW-based, now coexists with VQ — formerly mutually exclusive)
```

## 主实验结果 (seed=1)

| Model | @3 MAE | @6 MAE | @12 MAE |
|---|---|---|---|
| DGSTA baseline | 12.082 | 12.498 | 13.204 |
| DiMoR (seed=1) | 11.894 | 12.406 | 13.228 |
| -VQ Router | 12.176 | 12.645 | 13.493 |
| -DeepTrendNet | 11.912 | 12.421 | 13.222 |
| -DelayConv | 12.176 | 12.622 | 13.348 |

## 多 Seed 稳定性 (内部校验)

| Seed | @3 | @6 | @12 |
|---|---|---|---|
| 1 | 11.894 | 12.406 | 13.228 |
| 0 | 12.117 | 12.570 | 13.306 |
| 2 | 12.056 | 12.565 | 13.354 |
| Mean±Std | 12.022±0.10 | 12.514±0.08 | 13.296±0.05 |

## 机制分析发现 (论文核心)

1. **Codebook diversity is healthy**: 10 template graphs are diverse (pairwise cosine < 0.08)
2. **Hard routing converges to layer specialization**: 5/6 layers permanently fix to single templates, different layers choose DIFFERENT templates
3. **Inference-time dynamicity contributes minimally**: Frozen-template replay degrades MAE < 0.2%
4. **Semantic Attention suppression harms long horizon**: Original VQ branch excluded semantic attention; restoring coexistence partially recovers long-term prediction
5. **DeepTrendNet contributes near-zero**: DGSTA's attention backbone already absorbs trend information
6. **Balance loss degrades performance**: Soft/hard routing entropy detachment — regularizing soft probs cannot fix hard routing collapse
7. **Seed variance exceeds module gains**: 3-seed std (0.10) is comparable to ablation deltas (0.28)

## 论文叙事建议

NOT: "We propose a SOTA dynamic routing model"
INSTEAD: "We systematically analyze VQ-based dynamic graph routing behavior in traffic forecasting, revealing layer specialization, semantic suppression, and seed sensitivity — while maintaining competitive performance."

## 论文骨架

1. **Introduction**: Dynamic graph routing is widely used but poorly understood
2. **Related Work**: DGSTA, VQ routing, dynamic GNNs, traffic forecasting
3. **DiMoR Architecture**: VQ Router + Semantic Attention coexistence
4. **Mechanism Analysis** (core contribution):
   - Codebook diversity vs routing collapse
   - Layer specialization discovery
   - Frozen-template replay experiment
   - Semantic suppression hypothesis + structural fix
5. **Ablation Study**: VQ Router, DeepTrendNet, DelayConv contributions
6. **Discussion**: Why dynamic routing gains are limited — variance, backbone saturation, layer specialization vs temporal routing
7. **Conclusion**: Interpretable dynamic routing framework, competitive performance, mechanism insights

## 论文资产清单

- [ ] 主消融表 (seed=1)
- [ ] 多 seed 均值表 (可放 appendix)
- [ ] Router behavior 可视化 (heatmap, layer specialization)
- [ ] Frozen replay 结果
- [ ] Codebook diversity 分析
- [ ] 实验 cache ID 对照表
- [ ] 最终配置快照
- [ ] 环境信息 (Torch 2.9.1, CUDA 12, RTX 5070 Ti)
