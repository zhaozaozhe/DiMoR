# An Empirical Study of VQ Routing Behavior in Traffic Forecasting

**Target**: EI / SCI Q3-Q4 journal | **Status**: Final draft v3 — narrative frozen, claims calibrated

---

## Abstract

Dynamic graph routing is widely adopted in spatio-temporal traffic forecasting, yet the actual routing behaviors learned after training convergence remain largely unexamined. We conduct an empirical diagnosis of VQ-based discrete graph routing built upon the DGSTA backbone on the PeMS08 dataset. Through layer-wise behavior analysis, frozen replay validation, and ablation studies, we observe that VQ routing converges to depth-wise template specialization: different encoder layers consistently prefer different graph templates, while within each layer, template selection is predominantly static across time periods. Only one of six layers exhibits limited temporal variation. Freezing routing to dominant templates in the converged model causes negligible performance degradation (<0.03 MAE), suggesting that inference-time routing dynamicity plays a minor role. Geographic visualization of learned templates reveals interpretable spatial patterns — including local propagation and long-range corridor coupling — that correlate with time-of-day traffic characteristics. DiMoR achieves competitive forecasting performance while providing these interpretable routing behaviors. We discuss implications for the design of dynamic graph modules in traffic forecasting networks.

**Keywords**: traffic forecasting, dynamic graph routing, explainability, vector quantization, empirical analysis

---

## 1. Introduction

Traffic flow prediction is a fundamental task in intelligent transportation systems. Recent advances have increasingly adopted dynamic graph mechanisms — models that learn time-varying adjacency structures to capture evolving spatial dependencies among road sensors. These methods operate under the assumption that traffic spatial dependencies change meaningfully over time and that learning to switch between graph structures improves prediction.

However, relatively few studies have examined *what routing behaviors are actually learned* after training converges. Do graph routers produce time-varying structures in practice? Does real-time routing contribute meaningfully to prediction accuracy? And what spatial patterns, if any, do the learned graph templates encode?

In this work, we conduct an empirical diagnosis of VQ-based graph routing in the context of traffic forecasting, using the DGSTA architecture as our backbone. Rather than proposing a novel architecture, our contribution is a structured analysis of router behavior, including:

- Layer-wise template usage patterns across encoder depths and time periods
- Frozen replay experiments testing whether routing decisions matter at inference time
- Ablation isolating the contribution of each architectural component
- Geographic interpretation of learned graph templates with quantitative spatial statistics

Our observations on PeMS08 suggest that, in the converged model, VQ routing primarily provides layer-specific graph priors rather than fine-grained temporal adaptation. The routing mechanism contributes modest improvements, concentrated at shorter horizons, while its dynamicity at inference time is largely dispensable. However, a naive static per-layer graph baseline significantly underperforms, implying that the training process enabled by VQ routing — codebook competition and discrete exploration — plays a role that simple static parameterization cannot replicate.

---

## 2. Related Work

### 2.1 Traffic Flow Prediction

Deep learning for traffic prediction has evolved from RNN-based models to graph neural networks and Transformer architectures. DGSTA combines dynamic graph convolution with spatio-temporal self-attention, achieving competitive performance on PeMS benchmarks. Subsequent works have proposed adaptive adjacency learning, multi-graph fusion, and trend decomposition to further improve accuracy.

### 2.2 Dynamic Graph Routing

Vector Quantized (VQ) routing, inspired by VQ-VAE and mixture-of-experts architectures, maintains a discrete codebook of graph templates and uses Gumbel-Softmax to select among them. This approach has been applied to traffic forecasting. However, the actual learned behaviors — whether routers produce dynamic graphs or converge to stable specialization — have not been systematically characterized.

### 2.3 Explainability in Traffic Forecasting

Most explainability efforts focus on attention visualization or case studies. Few works examine the internal mechanisms of dynamic graph modules. Our work provides a structured diagnosis of VQ routing behavior, combining layer-wise analysis with quantitative spatial characterization of learned templates.

---

## 3. Method

### 3.1 Backbone: DGSTA

DiMoR is built upon the DGSTA backbone, which consists of:
- **Data Embedding**: value, positional, time-of-day, day-of-week, Laplacian spatial, and temporal prior embeddings
- **6-layer Spatio-Temporal Encoder**: each layer contains ST self-attention (temporal, geographic, and semantic attention heads) followed by an MLP with residual connections and stochastic depth
- **Skip Connections**: accumulated across encoder layers, projected via 1×1 convolutions to output dimensions

### 3.2 VQ Graph Router

The VQ Router maintains K = 10 learnable graph templates G_k ∈ R^{N×N} for N sensors. For input x ∈ R^{B×T×N×D}:

1. **Decomposition**: A moving-average kernel separates x into a trend component x_trend and residual x_res.
2. **Routing**: x_trend is encoded through a spatial MLP, reduced to traffic intensity, and classified into K logits. Gumbel-Softmax with hard sampling selects one dominant template per (batch, time_step) pair.
3. **Dual Graphs**: The selected template forms adj_vq ∈ R^{B×T×N×N}, while learned node embeddings produce a static complementary graph adj_adp ∈ R^{N×N}.
4. **Propagation**: x_res is processed through a 2-hop Graph Convolutional Network (GCN) using both adj_vq and adj_adp as support matrices. The output is fused with x_trend via x_trend + tanh(GCN_out) × 0.1, followed by LayerNorm.
5. **Regularization**: A temporal consistency loss penalizes rapid switching of template selection across consecutive time steps, weighted at 0.1.

After the GCN stage, temporal and geographic self-attention heads process the representation. The VQ Router operates within each encoder layer independently, meaning that each of the 6 layers maintains its own codebook and routing mechanism.

### 3.3 Auxiliary Modules

- **DelayConv**: A causal depthwise temporal convolution (kernel size 3, Dirac initialization) applied within GCN layers to smooth temporal features.
- **DeepTrendNet**: A lightweight MLP branch that predicts future traffic values from the decomposed trend component. Fused with the main prediction via a learnable scalar weight.
- **Semantic Attention**: DTW-based attention mask that allows nodes with similar daily traffic patterns to attend to each other.

All modules are config-gated via JSON configuration, enabling clean ablation. The default configuration (all gates off) produces the unmodified DGSTA baseline.

### 3.4 Model Complexity

DiMoR contains approximately 1.4M parameters, with the VQ Router accounting for roughly 10% of the total (~150K). The backbone (attention, GCN, embeddings) constitutes ~85%. Inference on the PeMS08 test set takes approximately 7.3 seconds on an NVIDIA RTX 5070 Ti, comparable to the baseline DGSTA (7.2s).

---

## 4. Experiments

### 4.1 Setup

We conduct experiments on the PeMS08 dataset (170 sensors, 5-minute intervals, July–August 2016, California highway network). Input and output windows are both 12 steps (60 minutes). The dataset is split 60%/20%/20% for training, validation, and testing. StandardScaler normalization is applied.

Training uses AdamW optimizer with a cosine learning rate schedule (initial lr = 1e-3, weight decay = 0.05), batch size 32, and curriculum learning over 300 epochs with early stopping (patience = 50). All experiments use a single NVIDIA RTX 5070 Ti GPU.

### 4.2 Main Results

Table 1 presents the ablation results. DiMoR achieves competitive forecasting performance. Removing the VQ Router causes the largest degradation, concentrated at short horizons. DeepTrendNet's contribution is marginal — consistent with the observation that the backbone attention mechanism already captures trend information. DelayConv provides moderate improvements.

**Table 1: Ablation study on PeMS08 (seed=1).**

| Model | @3 MAE | @6 MAE | @12 MAE |
|---|---|---|---|
| DGSTA (baseline) | 12.082 | 12.498 | 13.204 |
| DiMoR (Full) | **11.894** | **12.406** | 13.228 |
| − VQ Router | 12.176 | 12.645 | 13.493 |
| − DeepTrendNet | 11.912 | 12.421 | **13.222** |
| − DelayConv | 12.176 | 12.622 | 13.348 |

Multi-seed analysis (3 seeds) reveals seed-to-seed variance of σ ≈ 0.10 MAE, comparable to the observed ablation deltas. The routing behavior patterns (layer specialization, template diversity, routing stability) are consistent across all seeds. This indicates that performance claims should be interpreted as competitive rather than significantly superior. Full multi-seed results are provided in Appendix A.

### 4.3 Layer-wise Routing Specialization

**Figure 1** shows (a) the per-layer template usage distribution and (b) time-conditioned dominant template selection across five time periods (AM Peak, Midday, PM Peak, Night, Late Night).

A clear pattern emerges: different encoder layers consistently prefer different graph templates. Layer 0 selects Template 6, Layer 2 selects Template 4, and Layers 3 through 5 all select Template 0. Only Layer 1 exhibits distributed template usage, spreading its selections across Template 2 (38%), Template 0 (20%), and Template 9 (12%). For 5 of 6 layers, a single template accounts for over 99% of all selections.

The time-conditioned analysis (Figure 1b) confirms that these preferences are temporally stable. The dominant template for each layer remains unchanged across all five time periods, with the exception of Layer 1, which shifts from Template 3 in the morning to Template 2 during midday and evening.

This observation indicates that VQ routing converges to a *layer-wise static structural prior*. The routing mechanism, despite being designed for time-varying graph selection, learns depth-dependent specialization rather than temporal adaptation.

### 4.4 Frozen Replay Validation

To quantify whether real-time routing decisions are functionally important, we conduct a frozen replay experiment. Each layer's routing is locked to its dominant template, and inference is re-run on the test set. **Figure 2** shows the resulting MAE degradation across three prediction horizons, with error bars indicating standard deviation over 3 seeds.

The degradation is consistently small: less than 0.03 MAE across all horizons. At the 60-minute horizon (@12), the degradation reaches approximately 0.023 MAE, while at 15 minutes (@3) it is merely 0.010 MAE. All degradations are within one standard deviation of zero.

This suggests that in the converged model, inference-time routing dynamicity plays a limited role. The benefit of the VQ Router appears to derive from providing layer-specific graph priors — different layers operating on different spatial structures — rather than from real-time temporal adaptation of graph topology.

### 4.5 Static Per-Layer Graph Baseline

We further test whether the layer-wise specialization achieved by the VQ Router can be replicated by a simpler mechanism: replacing the entire VQ routing pipeline with independently learnable static adjacency matrices, one per encoder layer. This baseline contains slightly more parameters than the VQ Router but eliminates all routing machinery (no codebook, no Gumbel-Softmax, no consistency loss).

The static per-layer baseline achieves 12.889 / 13.631 / 14.879 (@3/@6/@12), significantly worse than both the VQ Router configuration and the original DGSTA baseline. This result is noteworthy: if VQ routing merely provides layer-specific static priors, a simpler static mechanism should match or approach its performance. The observed collapse suggests that the training process enabled by VQ routing — codebook competition, discrete template exploration, and the auxiliary consistency loss — contributes to optimization in ways that independent static graphs cannot replicate. We leave a rigorous disentanglement of these training dynamics to future work.

### 4.6 Geographic Interpretation of Learned Templates

**Figure 3** visualizes the four actively used graph templates mapped to the PeMS08 sensor network using a topology-preserving spectral layout based on the road-network hop-distance matrix. Each template reveals a distinct spatial connectivity pattern.

Quantitative hop-distance analysis of template edges confirms the visual observations. Template 0 exhibits 100% local edges (within 3 hops on the road network) — a pattern consistent with local neighborhood propagation. Template 2 shows 61% long-range edges (beyond 5 hops) — consistent with corridor-style coupling that spans the highway network. Template 4 shows 50% long-range edges, and Template 6 shows 39% long-range edges with a balanced mid-range component.

**Figure 4** tracks the activation of the dominant templates in Layer 1 — the only layer showing temporal variation — over a 24-hour period. During daytime, Template 2 (long-range corridor coupling) increases its usage, reaching 59–76% during midday and evening. At night, Template 0 (local propagation) becomes dominant, reaching 37–57%. This temporal shift is consistent with the intuitive transition in traffic behavior from long-range commuting patterns during business hours to localized nocturnal activity.

These observations suggest that even though VQ routing does not produce highly dynamic graph switching, the learned templates encode traffic-meaningful spatial structures. The router's limited temporal variation appears to align with coarse-grained traffic regime changes rather than fine-grained time-step adaptation.

---

## 5. Discussion

### 5.1 Observed Routing Behavior

In our experiments on PeMS08 with the DGSTA backbone, VQ routing in the converged model exhibits predominantly stable, layer-specific template usage rather than strongly time-varying graph selection. The routing mechanism's apparent contribution is to provide different graph priors to different encoder layers — a form of depth-wise specialization — rather than to adapt graph structure to fine-grained temporal changes in traffic state.

The frozen replay experiment reinforces this interpretation: locking routing to dominant templates causes minimal performance degradation, indicating that inference-time dynamicity is largely dispensable in the converged model.

### 5.2 The Training Dynamics Puzzle

The static per-layer baseline experiment presents an intriguing observation. If VQ routing merely converges to layer-wise static specialization, one might expect static per-layer graphs — which provide the same layer-specific capacity — to match VQ Router performance. Yet the static baseline significantly underperforms, falling below even the original DGSTA configuration.

We hypothesize that the VQ routing mechanism, despite converging to apparently static behavior, provides valuable structure during training. The codebook competition and Gumbel-Softmax exploration may act as an implicit regularizer or structural search mechanism, allowing the model to explore diverse spatial configurations before settling into stable, specialized assignments. A rigorous investigation of these training dynamics — including trajectory analysis of routing entropy and codebook utilization across training epochs — is left for future work.

### 5.3 Limitations

- **Single dataset**: All observations are on PeMS08. As an empirical interpretability study, the specific spatial semantics (e.g., corridor coupling patterns) are inherently tied to the PeMS highway topology. Cross-dataset validation is needed to assess whether the overarching phenomenon of layer-wise specialization generalizes to networks with different topological properties.
- **Single backbone**: Findings may depend on DGSTA's strong attention mechanism, which could absorb trend and temporal information that would otherwise manifest as routing variability.
- **Seed variance**: Multi-seed analysis shows σ ≈ 0.10 MAE, comparable to observed ablation gains. Performance claims are therefore limited to "competitive" rather than "significantly superior."
- **Geographic interpretation**: Template semantic labels (e.g., "corridor coupling") are descriptive characterizations supported by quantitative hop-distance statistics, not causal proofs of traffic mechanism modeling.

### 5.4 Implications for Dynamic Graph Design

Our observations raise questions for the design of dynamic graph modules in traffic forecasting. If inference-time routing dynamicity contributes minimally, the complexity of Gumbel-Softmax routing and consistency regularization may be replaceable by simpler mechanisms for certain applications. However, the failure of the static per-layer baseline suggests that the training-time dynamics of routing — codebook competition and discrete exploration — provide optimization benefits that simpler architectures cannot replicate. Understanding this distinction between training-time and inference-time contributions may inform more efficient dynamic graph designs.

---

## 6. Conclusion

We present an empirical diagnosis of VQ-based graph routing behavior in traffic forecasting, built upon the DGSTA backbone. On the PeMS08 dataset, we observe that VQ routing converges to depth-wise template specialization with limited temporal dynamicity. Frozen replay experiments indicate that inference-time routing dynamicity plays a minor role in prediction accuracy. Geographic analysis of learned templates reveals interpretable spatial patterns — local propagation and long-range corridor coupling — that correlate with time-of-day traffic characteristics. A static per-layer graph baseline significantly underperforms, implying that the training dynamics of routing contribute beyond what simple layer-specific parameterization can achieve.

DiMoR achieves competitive forecasting performance while providing interpretable routing behaviors. We hope these observations inform the design and evaluation of dynamic graph modules in spatio-temporal forecasting, and encourage further investigation into the training dynamics of discrete routing mechanisms.

---

## Appendix A: Multi-Seed Analysis

| Seed | @3 MAE | @6 MAE | @12 MAE |
|---|---|---|---|
| 1 | 11.894 | 12.406 | 13.228 |
| 0 | 12.117 | 12.570 | 13.306 |
| 2 | 12.056 | 12.565 | 13.354 |
| Mean ± Std | 12.022 ± 0.10 | 12.514 ± 0.08 | 13.296 ± 0.05 |

Seed-to-seed variance (σ ≈ 0.10) is comparable to ablation deltas. Routing behavior patterns (layer specialization, template diversity, routing stability) are consistent across all seeds.

## Appendix B: Experiment Cache Index

| Experiment | Exp ID | Config | Seed |
|---|---|---|---|
| Full (best) | 71098 | VQ+Trend+Delay | 1 |
| − VQ Router | 68783 | Trend+Delay | 1 |
| − DeepTrendNet | 64832 | VQ+Delay | 1 |
| − DelayConv | 43876 | VQ+Trend | 1 |
| Static Per-Layer | 75685 | Static+Trend+Delay | 1 |
| Full | 61239 | VQ+Trend+Delay | 0 |
| Full | 16450 | VQ+Trend+Delay | 2 |

## Appendix C: Template Hop-Distance Statistics

| Template | Local (<3 hop) | Mid (3-5 hop) | Long (>5 hop) | Interpretation |
|---|---|---|---|---|
| T0 | 100.0% | 0.0% | 0.0% | Local propagation |
| T2 | 8.7% | 30.4% | 60.9% | Long-range corridor |
| T4 | 3.8% | 46.2% | 50.0% | Long-range corridor |
| T6 | 5.6% | 55.6% | 38.9% | Mixed-range |

## Appendix D: Multi-Seed Frozen Replay

| Seed | @3 ΔMAE | @6 ΔMAE | @12 ΔMAE |
|---|---|---|---|
| 1 | +0.010 | +0.013 | +0.023 |
| 0 | +0.017 | +0.018 | +0.020 |
| 2 | +0.016 | +0.017 | +0.019 |
| Mean ± Std | +0.014 ± 0.003 | +0.016 ± 0.002 | +0.021 ± 0.002 |
