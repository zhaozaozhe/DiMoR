# DiMoR 论文配图指南

## 论文定位
普通但扎实的交通预测可解释性分析。不宣称革命性发现，不发明理论名词。
核心信息："DiMoR achieves competitive performance while providing interpretable routing behaviors."

## 最终入选 4 张图 (Gemini v3 review)

### Figure 1: Layer Specialization + Time Routing (Merged)
文件: `paper_fig1_merged_specialization.png`  
Caption: "(a) Template usage distribution across encoder layers. Different layers consistently prefer different templates. (b) Time-conditioned dominant template selection. For 5/6 layers, the dominant template is static across all time periods. Layer 1 shows limited variation between daytime and nighttime preferences. This indicates that VQ routing converges to layer-wise static specialization with weak temporal dynamicity."

### Figure 2: Frozen Replay (Redesigned)
文件: `paper_fig2_frozen_replay.png`  
Caption: "Frozen template replay degradation with multi-seed error bars. When each layer's routing is locked to its dominant template, MAE increases by less than 0.03 across all horizons. Error bars show standard deviation over 3 seeds. This suggests that in the converged model, inference-time routing dynamicity plays a limited role."

### Figure 3: Geographic Template Visualization
文件: `geo_templates.png`  
Caption: "Learned graph templates mapped to the PeMS08 sensor network (topology-preserving spectral layout). Templates actively used by encoder layers exhibit distinct spatial patterns: T0 shows local propagation (100% edges within 3 hops), T2 shows long-range corridor coupling (61% edges beyond 5 hops). Color intensity reflects edge weight."

### Figure 4: Template Activation Timeline
文件: `geo_activation_timeline.png`  
Caption: "Layer 1 template activation over 24 hours. The router increases usage of the long-range template (T2) during AM/PM peak periods, while nighttime routing shifts toward local propagation (T0). This observation is consistent with the intuitive shift in traffic behavior from long-range commuting to localized nocturnal patterns."

## 淘汰的图 (Gemini 建议删除/移 Appendix)

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
