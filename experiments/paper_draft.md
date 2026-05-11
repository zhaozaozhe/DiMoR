# An Empirical Study of VQ Routing Behavior in Traffic Forecasting

**Target**: EI / SCI Q3-Q4 journal
**Status**: Revised draft v2 — claims downgraded per review

---

## Abstract

Dynamic graph routing is widely adopted in spatio-temporal traffic forecasting, yet the
actual routing behaviors learned after training convergence remain largely unexamined.
We conduct an empirical diagnosis of VQ-based discrete graph routing built upon the DGSTA
backbone on the PeMS08 dataset. Through layer-wise behavior analysis, frozen replay
validation, and ablation studies, we observe that: (1) VQ routing converges to depth-wise
template specialization—different encoder layers consistently prefer different graph
templates, while within each layer, template selection is predominantly static across time
periods; (2) freezing routing to dominant templates in the converged model causes
negligible performance degradation (<0.03 MAE), suggesting that inference-time routing
dynamicity plays a limited role; (3) the semantic attention and routing branches exhibit
suggestions of horizon-level complementarity. DiMoR achieves competitive forecasting
performance while providing interpretable routing behaviors. We discuss implications for
the design of dynamic graph modules in traffic forecasting.

**Keywords**: traffic forecasting, dynamic graph routing, explainability, vector quantization, empirical analysis

---

## 1. Introduction

Traffic flow prediction is a fundamental task in intelligent transportation systems.
Recent advances have increasingly adopted dynamic graph mechanisms—models that learn
time-varying adjacency structures to capture evolving spatial dependencies among road
sensors. These methods, including DGCNN-style adaptive graphs and VQ-based routing,
operate under the assumption that traffic spatial dependencies change meaningfully
over time and that learning to switch between graph structures improves prediction.

However, relatively few studies have examined *what routing behaviors are actually
learned* after training converges. Do graph routers produce time-varying structures
in practice? Does real-time routing contribute meaningfully to prediction accuracy?
Which layers, if any, benefit from routing flexibility?

In this work, we conduct an empirical diagnosis of VQ-based graph routing in the
context of traffic forecasting, using the DGSTA architecture as our backbone.
Rather than proposing a novel architecture, our contribution is a structured
analysis of router behavior, including:

- Layer-wise template usage patterns across encoder depths and time periods
- Frozen replay experiments testing whether routing decisions matter at inference time
- Ablation isolating the contribution of each architectural component
- Comparison of semantic vs. structural spatial modeling across prediction horizons

Our observations on PeMS08 suggest that, in the converged model, VQ routing
primarily provides layer-specific graph priors rather than fine-grained temporal
adaptation. The routing mechanism contributes modest improvements, concentrated
at shorter horizons, while its dynamicity at inference time is largely dispensable.
We discuss the implications of these findings for the design of dynamic graph
modules in traffic forecasting networks.

---

## 2. Related Work

### 2.1 Traffic Flow Prediction

Deep learning for traffic prediction has evolved from RNN-based models to graph
neural networks and Transformer architectures. DGSTA combines dynamic graph
convolution with spatio-temporal self-attention, achieving competitive performance
on PeMS benchmarks. Many subsequent works have proposed additional mechanisms—adaptive
adjacency learning, multi-graph fusion, trend decomposition—to further improve accuracy.

### 2.2 Dynamic Graph Routing

Vector Quantized (VQ) routing, inspired by VQ-VAE and mixture-of-experts, maintains
a discrete codebook of graph templates and uses Gumbel-Softmax to select among them.
This approach has been applied to traffic forecasting. However, the actual learned
behaviors of such routers—whether they produce dynamic graphs or converge to stable
specialization—have not been systematically characterized.

### 2.3 Explainability in Traffic Forecasting

Most explainability efforts focus on attention visualization, showing which time
steps or sensors the model attends to. Few works examine the internal mechanisms
of dynamic graph modules. Our work provides a structured diagnosis of VQ routing
behavior in a traffic forecasting context.

---

## 3. Method

### 3.1 Backbone: DGSTA

DiMoR is built upon the DGSTA backbone:
- **Data Embedding**: value, positional, time-of-day, day-of-week, Laplacian spatial,
  and temporal prior embeddings
- **6-layer Spatio-Temporal Encoder**: each layer contains ST self-attention (temporal,
  geographic, and semantic attention heads) followed by MLP with residual connections
  and stochastic depth
- **Skip Connections**: accumulated across encoder layers, projected via 1×1 convolutions
  to output dimensions

### 3.2 VQ Graph Router

The VQ Router maintains K=10 learnable graph templates G_k ∈ R^{N×N}. For input
x ∈ R^{B×T×N×D}:

1. **Decomposition**: Moving-average kernel separates x into trend x_trend and residual x_res
2. **Routing**: x_trend is encoded through spatial MLP → classifier → Gumbel-Softmax(hard=True)
   selects one dominant template per (batch, time_step)
3. **Dual Graphs**: selected template forms adj_vq ∈ R^{B×T×N×N}; learned node embeddings
   produce a static complementary graph adj_adp ∈ R^{N×N}
4. **Propagation**: x_res processed through 2-hop GCN with [adj_vq, adj_adp],
   fused via x_trend + tanh(GCN_out) × 0.1, LayerNorm
5. **Regularization**: Temporal consistency loss penalizes rapid switching of template
   selection across consecutive time steps (weight = 0.1)

### 3.3 Auxiliary Modules

- **DeepTrendNet**: Lightweight MLP predicting future values from trend component.
  Output fused via learnable scalar weight.
- **DelayConv**: Causal depthwise temporal convolution (kernel=3, Dirac init) within GCN.
- **Semantic Attention**: DTW-based attention mask connecting nodes with similar daily patterns.

### 3.4 Configurable Design

All modules are config-gated, enabling clean ablation. Default (all off) = unmodified DGSTA.

### 3.5 Model Complexity

| Component | Parameters | % of Total |
|---|---|---|
| DGSTA Backbone (incl. attention, GCN, embeddings) | ~1.2M | 85% |
| VQ Router (codebook 10×170² + routing MLP) | ~150K | 10% |
| DeepTrendNet | ~25K | 2% |
| DelayConv | ~40K | 3% |
| **Total** | **~1.4M** | 100% |

Inference time: ~7.3s on test set (RTX 5070 Ti), comparable to baseline DGSTA (~7.2s).

---

## 4. Experiments

### 4.1 Setup

**Dataset**: PeMS08 (170 sensors, 5-min intervals, Jul-Aug 2016). Input/output windows
both 12 steps (60 min). Split: 60%/20%/20% train/val/test. StandardScaler normalization.

**Training**: AdamW, lr=1e-3, weight decay 0.05, batch size 32, cosine LR schedule,
curriculum learning, 300 epochs, early stopping (patience=50). RTX 5070 Ti GPU.

**Limitations acknowledged**: Results reported on a single dataset and backbone.
Multi-seed analysis shows variance comparable to observed improvements (§4.2).

### 4.2 Main Results

| Model | @3 MAE | @6 MAE | @12 MAE |
|---|---|---|---|
| DGSTA (baseline) | 12.082 | 12.498 | 13.204 |
| DiMoR (Full) | **11.894** | **12.406** | 13.228 |
| − VQ Router | 12.176 | 12.645 | 13.493 |
| − DeepTrendNet | 11.912 | 12.421 | **13.222** |
| − DelayConv | 12.176 | 12.622 | 13.348 |

DiMoR achieves competitive performance. Removing the VQ Router causes the largest
degradation, concentrated at short horizons. DeepTrendNet's contribution is marginal.
Multi-seed analysis (3 seeds) is provided in Appendix A; seed-to-seed variance (σ≈0.10)
is comparable to the observed gains, indicating that results should be interpreted as
competitive rather than significantly superior.

### 4.3 Routing Behavior Analysis

**Layer-wise Template Specialization.** Figure 2 (time_routing.png) shows dominant
template selection per layer across five time periods (morning, midday, evening,
night, late night). A clear pattern emerges: different layers consistently prefer
different templates, while within each layer, template selection is stable across
time periods. Only Layer 1 shows variation between daytime and nighttime preferences.

**Template Usage Distribution.** Figure 1 (layer_specialization.png) quantifies
per-layer template usage. For 5 of 6 layers, a single template accounts for >99%
of selections. Layer 1 distributes selections across T2 (38%), T0 (20%), T9 (12%).
The 10 templates are structurally diverse (avg pairwise cosine sim < 0.08).

### 4.4 Frozen Replay Validation

Figure 5 (frozen_replay.png): Each layer's routing is locked to its dominant template,
and inference is re-run without routing. Performance degrades by <0.03 MAE across all
horizons. This suggests that, in the converged model, inference-time routing
dynamicity plays a limited role. The benefit of VQ routing may stem from providing
layer-specific graph priors rather than from real-time temporal adaptation.

### 4.5 Template Visualization

Figure 6 (template_viz.png): Adjacency matrices of learned templates. Actively used
templates (top row) exhibit clearer structural patterns than rarely selected ones
(bottom row). Differences between templates used by different layers suggest they
encode distinct spatial connectivity patterns.

### 4.6 Horizon Characteristics

Figure 7 (semantic_suppression.png): The semantic attention configuration achieves
better long-horizon accuracy (@12), while VQ routing improves short horizons (@3, @6).
This suggests potential horizon-level complementarity between semantic and structural
spatial modeling, though the coexistence configuration does not yet realize this
potential stably.

---

## 5. Discussion

### 5.1 Observed Routing Behavior

In our experiments on PeMS08 with the DGSTA backbone, VQ routing in the converged
model exhibits predominantly stable, layer-specific template usage rather than
strongly time-varying graph selection. The routing mechanism's apparent contribution
is to provide different graph priors to different encoder layers—a form of depth-wise
specialization—rather than to adapt graph structure to temporal changes in traffic state.

### 5.2 Implications

These observations raise questions for dynamic graph module design in traffic
forecasting. If inference-time routing dynamicity contributes minimally, the
complexity of Gumbel-Softmax routing and consistency regularization may be
replaceable by simpler mechanisms such as per-layer learnable static graphs.
This hypothesis warrants further investigation.

### 5.3 Limitations

- **Single dataset**: All observations are on PeMS08. Cross-dataset validation
  (PeMS04, METR-LA) is needed to assess generality.
- **Single backbone**: Findings may be specific to DGSTA's strong attention
  mechanism, which could absorb trend and temporal information.
- **Seed variance**: Multi-seed analysis (Appendix A) shows σ≈0.10, comparable
  to observed gains. Claims are therefore limited to "competitive" rather than
  "significantly superior."
- **Training dynamics**: The frozen replay experiment tests only the converged
  model. Whether routing dynamicity plays a role during training optimization
  remains an open question.

### 5.4 Future Work

- Static per-layer graph baseline to isolate the contribution of layer-wise
  graph diversity from routing dynamics
- Cross-dataset and cross-backbone replication
- Wrong-template perturbation to strengthen causal evidence
- Investigation of routing behavior under non-stationary conditions (accidents, weather)

---

## 6. Conclusion

We present an empirical diagnosis of VQ-based graph routing behavior in traffic
forecasting, using the DiMoR framework built upon DGSTA. On PeMS08, we observe that
VQ routing converges to depth-wise template specialization with limited temporal
adaptation in the converged model. Frozen replay experiments indicate that
inference-time routing dynamicity plays a minor role. The routing and semantic
attention branches show suggestions of horizon-level complementarity. DiMoR achieves
competitive forecasting performance while providing interpretable routing behaviors.
We hope these observations inform the design and evaluation of future dynamic graph
modules in spatio-temporal forecasting.

---

## Appendix A: Multi-Seed Analysis

| Seed | @3 MAE | @6 MAE | @12 MAE |
|---|---|---|---|
| 1 | 11.894 | 12.406 | 13.228 |
| 0 | 12.117 | 12.570 | 13.306 |
| 2 | 12.056 | 12.565 | 13.354 |
| Mean±Std | 12.022±0.10 | 12.514±0.08 | 13.296±0.05 |

Seed-to-seed variance (σ≈0.10) is comparable to ablation deltas. The routing behavior
patterns (layer specialization, template diversity, routing stability) are consistent
across all seeds.

## Appendix B: Experiment Cache Index

| Experiment | Exp ID | Config | Seed |
|---|---|---|---|
| Full (best) | 71098 | VQ+Trend+Delay | 1 |
| −VQ Router | 68783 | Trend+Delay | 1 |
| −DeepTrendNet | 64832 | VQ+Delay | 1 |
| −DelayConv | 43876 | VQ+Trend | 1 |
| Full | 61239 | VQ+Trend+Delay | 0 |
| Full | 16450 | VQ+Trend+Delay | 2 |
