# DiMoR 论文配图指南

## 论文定位
普通但扎实的交通预测可解释性分析。不宣称革命性发现，不发明理论名词。
核心信息："DiMoR achieves competitive performance while providing interpretable routing behaviors."

## 最终入选 5 张图

### Figure 3: Routing Behavior by Layer and Time Period
文件: `fig2_time_routing.png`
| 论文位置: Section 4.2 Routing Behavior Analysis
Caption: "Dominant template selection across encoder layers and time periods.
Each cell shows the most frequently selected graph template for a given layer-period
combination. Routed template choice is predominantly stable: 5 of 6 layers select
the same template regardless of time period, while Layer 1 exhibits limited
day/night variation."

### Figure 4: Per-Layer Template Utilization
文件: `fig1_layer_specialization.png`
| 论文位置: Section 4.2
Caption: "Template usage distribution per encoder layer. Different layers
consistently prefer different templates, suggesting that VQ routing converges
to depth-wise graph specialization. Layer 1 shows the most diverse template usage
among all layers."

### Figure 5: Frozen Replay Validation
文件: `fig5_frozen_replay.png`
| 论文位置: Section 4.3 Validation
Caption: "Frozen template replay experiment. When each layer's routing is locked
to its dominant template and inference is re-run without any routing decisions,
performance degrades by less than 0.03 MAE across all horizons. This indicates
that inference-time routing dynamicity contributes minimally to prediction
accuracy in the converged model."

### Figure 6: Learned Graph Templates
文件: `fig6_template_viz.png`
| 论文位置: Section 4.4 Template Analysis
Caption: "Visualization of learned graph templates. Top row: templates actively
used by specific encoder layers. Bottom row: templates that were learned but
rarely selected. Actively used templates exhibit clearer structural patterns
along the diagonal."

### Figure 7: Routing vs Semantic Branch Horizon Trade-off
文件: `fig7_semantic_suppression.png`
| 论文位置: Section 4.5 Discussion
Caption: "Performance comparison across prediction horizons. The semantic attention
branch (without VQ routing) achieves better long-horizon accuracy (@12), while
VQ routing improves short-horizon prediction (@3, @6). This suggests complementary
temporal characteristics between the two mechanisms."

## 淘汰的图（不放论文正文）

| 图 | 原因 |
|---|---|
| fig3_codebook_diversity | 模板不坍缩是预期行为，不是发现 |
| fig4_static_vs_dynamic | 与 fig1/fig2 信息重叠 |
| fig8_routing_entropy | 单统计量不够支撑独立叙事 |
| fig9_layer_perturbation | 效应量太小(<0.02)，缺乏说服力 |

## 论文可用指标表格

| Model | @3 MAE | @6 MAE | @12 MAE |
|---|---|---|---|
| DGSTA (baseline) | 12.082 | 12.498 | 13.204 |
| DiMoR (Ours) | 11.894 | 12.406 | 13.228 |
| −VQ Router | 12.176 | 12.645 | 13.493 |
| −DeepTrendNet | 11.912 | 12.421 | 13.222 |
| −DelayConv | 12.176 | 12.622 | 13.348 |

Note: Results reported on single seed. Multi-seed mean and std provided in Appendix.
