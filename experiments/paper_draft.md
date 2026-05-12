# An Empirical Study of VQ Routing Behavior in Traffic Forecasting

**Target**: EI / SCI Q3-Q4 journal | **Status**: v5 — reviewer edit: narrative restructured, tension highlighted

---

## Abstract

Dynamic graph routing is widely adopted in spatio-temporal traffic forecasting, yet the actual routing behaviors learned after training convergence remain largely unexamined. We conduct an empirical diagnosis of VQ-based discrete graph routing built upon the DGSTA backbone on the PeMS08 dataset. Our analysis reveals that VQ routing converges to depth-wise template specialization: different encoder layers consistently prefer different graph templates, while within each layer, template selection is predominantly static across time periods. Freezing routing to dominant templates in the converged model causes negligible performance degradation (<0.03 MAE), suggesting that inference-time routing dynamicity plays a minor role. This creates an apparent paradox: the router's real-time decisions are largely dispensable, yet replacing the VQ mechanism with independently learned per-layer static graphs causes performance to collapse below the original baseline. Training trajectory analysis offers a potential resolution: the router maintains near-maximum entropy (~0.99) for the vast majority of training, indicating an extended phase of broad structural exploration that static graphs bypass entirely. Geographic visualization of learned templates reveals interpretable spatial patterns — including local propagation and long-range corridor coupling — with quantitative fidelity metrics showing 30–50× larger degradation under random template substitution. Taken together, these observations suggest that the VQ router's primary contribution may lie in training-time structural exploration rather than inference-time graph switching. DiMoR achieves competitive forecasting performance while providing these interpretable routing behaviors.

**Keywords**: traffic forecasting, dynamic graph routing, explainability, vector quantization, empirical analysis

---

## 1. Introduction

Traffic flow prediction is a fundamental task in intelligent transportation systems. Recent advances have increasingly adopted dynamic graph mechanisms — models that learn time-varying adjacency structures to capture evolving spatial dependencies among road sensors. These methods operate under the implicit assumption that traffic spatial dependencies change meaningfully over time and that learning to switch between graph structures improves prediction.

However, relatively few studies have examined *what routing behaviors are actually learned* after training converges. Do graph routers produce time-varying structures in practice? Does real-time routing contribute meaningfully to prediction accuracy? And what spatial patterns, if any, do the learned graph templates encode?

In this work, we conduct an empirical diagnosis of VQ-based graph routing in the context of traffic forecasting, using the DGSTA architecture as our backbone. Rather than proposing a novel architecture, our contribution is a structured analysis of router behavior, spanning layer-wise template usage, frozen replay validation, ablation studies, and geographic interpretation of learned graph templates with quantitative spatial statistics.

Our observations on PeMS08 reveal an apparent paradox. On one hand, VQ routing converges to predominantly static, layer-specific template assignments — freezing these assignments at inference time causes negligible degradation (<0.03 MAE). On the other hand, simplifying the router into independently learned per-layer static graphs causes performance to collapse below the original DGSTA baseline. This tension — real-time routing decisions matter little, yet the routing mechanism appears essential during training — motivates the central investigation of this work. Training trajectory analysis suggests a potential resolution: the router undergoes an extended high-entropy exploration phase (~220 epochs) before converging to specialization, a structural search process that static parameterization cannot replicate.

---

## 2. Related Work

### 2.1 Traffic Flow Prediction

Deep learning for traffic prediction has evolved from RNN-based models to graph neural networks and Transformer architectures. DGSTA combines dynamic graph convolution with spatio-temporal self-attention, achieving competitive performance on PeMS benchmarks. Subsequent works have proposed adaptive adjacency learning, multi-graph fusion, and trend decomposition to further improve accuracy.

### 2.2 Dynamic Graph Routing

Vector Quantized (VQ) routing, inspired by VQ-VAE and mixture-of-experts architectures, maintains a discrete codebook of graph templates and uses Gumbel-Softmax to select among them. This approach has been applied to traffic forecasting. However, the actual learned behaviors — whether routers produce dynamic graphs or converge to stable specialization — have not been systematically characterized.

### 2.3 Explainability in Traffic Forecasting

Most explainability efforts focus on attention visualization or case studies. Few works examine the internal mechanisms of dynamic graph modules. Our work provides a structured diagnosis of VQ routing behavior, combining layer-wise analysis with quantitative spatial characterization of learned templates and XAI-aligned fidelity metrics.

---

## 3. Method

### 3.1 Backbone: DGSTA

DiMoR is built upon the DGSTA backbone, which consists of a data embedding layer (value, positional, time-of-day, day-of-week, Laplacian spatial, and temporal prior embeddings), a 6-layer spatio-temporal encoder (each layer containing ST self-attention with temporal, geographic, and semantic attention heads followed by an MLP with residual connections and stochastic depth), and skip connections accumulated across encoder layers and projected via 1×1 convolutions to output dimensions.

### 3.2 VQ Graph Router

The VQ Router maintains K = 10 learnable graph templates G_k ∈ R^{N×N} for N sensors. For input x ∈ R^{B×T×N×D}:

1. **Decomposition**: A moving-average kernel separates x into a trend component x_trend and residual x_res.
2. **Routing**: x_trend is encoded through a spatial MLP, reduced to traffic intensity, and classified into K logits. Gumbel-Softmax with hard sampling selects one dominant template per (batch, time_step) pair.
3. **Dual Graphs**: The selected template forms adj_vq ∈ R^{B×T×N×N}, while learned node embeddings produce a static complementary graph adj_adp ∈ R^{N×N}.
4. **Propagation**: x_res is processed through a 2-hop GCN using both adj_vq and adj_adp as support matrices. The output is fused with x_trend via x_trend + tanh(GCN_out) × 0.1, followed by LayerNorm.
5. **Regularization**: A temporal consistency loss penalizes rapid switching of template selection across consecutive time steps.

The VQ Router operates within each encoder layer independently, meaning each of the 6 layers maintains its own codebook and routing mechanism. After the GCN stage, temporal and geographic self-attention heads process the representation.

### 3.3 Auxiliary Modules

**DelayConv**: A causal depthwise temporal convolution (kernel size 3, Dirac initialization) applied within GCN layers. **DeepTrendNet**: A lightweight MLP branch predicting future values from the trend component, fused via a learnable scalar weight. **Semantic Attention**: A DTW-based attention mask connecting nodes with similar daily traffic patterns.

All modules are config-gated via JSON configuration. The default configuration (all gates off) produces the unmodified DGSTA baseline.

### 3.4 Model Complexity

DiMoR contains approximately 1.4M parameters, with the VQ Router accounting for ~10% (~150K). Inference takes ~7.3s on an NVIDIA RTX 5070 Ti, comparable to the baseline DGSTA (~7.2s).

---

## 4. Experiments

### 4.1 Setup

We use the PeMS08 dataset (170 sensors, 5-minute intervals, July–August 2016, California highway network). Input and output windows are both 12 steps (60 minutes), split 60%/20%/20% for train/val/test. StandardScaler normalization is applied. Training: AdamW optimizer, cosine LR schedule (initial lr=1e-3, weight decay=0.05), batch size 32, curriculum learning, 300 epochs, early stopping (patience=50). Single NVIDIA RTX 5070 Ti GPU.

### 4.2 Main Results

While Table 1 establishes that the VQ Router contributes meaningfully to prediction accuracy, it does not reveal *how* this contribution is achieved. We next examine the router's internal behavior.

**Table 1: Ablation study on PeMS08 (seed=1).**

| Model | @3 MAE | @6 MAE | @12 MAE |
|---|---|---|---|
| DGSTA (baseline) | 12.082 | 12.498 | 13.204 |
| DiMoR (Full) | **11.894** | **12.406** | 13.228 |
| − VQ Router | 12.176 | 12.645 | 13.493 |
| − DeepTrendNet | 11.912 | 12.421 | **13.222** |
| − DelayConv | 12.176 | 12.622 | 13.348 |

Removing the VQ Router causes the largest degradation, concentrated at short horizons. DeepTrendNet's contribution is marginal — consistent with the backbone attention mechanism already capturing trend information. DelayConv provides moderate improvements. Multi-seed analysis (3 seeds, Appendix A) reveals seed-to-seed variance (σ≈0.10 MAE) comparable to the observed ablation deltas, indicating that performance should be interpreted as competitive rather than significantly superior. Routing behavior patterns are consistent across all seeds.

### 4.3 Routing Behavior: Layer-wise Static Specialization

**Figure 1** shows (a) per-layer template usage distribution and (b) time-conditioned dominant template selection across five periods. Different encoder layers consistently prefer different templates: Layer 0 selects T6, Layer 2 selects T4, and Layers 3–5 all select T0. Only Layer 1 exhibits distributed usage (T2: 38%, T0: 20%, T9: 12%). For 5 of 6 layers, a single template accounts for >99% of all selections. The time-conditioned analysis confirms these preferences are temporally stable — the dominant template remains unchanged across all five periods for all layers except Layer 1.

To test whether these real-time routing decisions matter, we lock each layer's routing to its dominant template and re-run inference. **Figure 2** shows the resulting MAE degradation: <0.03 across all horizons, with multi-seed error bars. Together, these findings indicate that VQ routing converges to a layer-wise static structural prior — the routing mechanism, despite being designed for time-varying graph selection, learns depth-dependent specialization rather than temporal adaptation, and its inference-time dynamicity is largely dispensable.

### 4.4 The Central Puzzle: Static Replacement Fails, Yet Training Dynamics Persist

The observation that routing is predominantly static raises a natural question: if the router does not meaningfully switch between templates, why not replace it with simpler per-layer static graphs? We test this by replacing the entire VQ routing pipeline with independently learnable static adjacency matrices — one per encoder layer — eliminating all routing machinery while retaining slightly more parameters.

The static per-layer baseline achieves 12.889 / 13.631 / 14.879 (@3/@6/@12) — significantly worse than both the VQ Router and the original DGSTA baseline. This is noteworthy: if VQ routing merely provides layer-specific static priors, a simpler static mechanism should approach its performance. The observed collapse suggests that the VQ mechanism contributes something during training that independent static parameterization cannot replicate.

**Figure 5** offers a potential clue. Tracking routing entropy at 15 sampled checkpoints across training reveals an unexpected trajectory: the router maintains near-maximum entropy (H_norm ≈ 0.99) for over 220 epochs before entering a final convergence phase. This extended high-entropy phase indicates broad structural exploration — the model evaluates diverse graph configurations through codebook competition before committing to specialized assignments. Static graphs, initialized randomly and optimized independently, bypass this exploration entirely. We leave rigorous characterization of this phase transition — including whether entropy collapses abruptly or gradually in the final epochs — for future work.

### 4.5 Interpretability Analysis

**Geographic Characterization.** **Figure 3** maps the four actively used templates to the PeMS08 sensor network using a topology-preserving layout. Quantitative hop-distance analysis (Appendix C) confirms: T0 exhibits 100% local edges (within 3 hops), consistent with neighborhood propagation; T2 shows 61% long-range edges (beyond 5 hops), consistent with corridor coupling. **Figure 4** tracks Layer 1 template activation over 24 hours: T2 (long-range corridor coupling) dominates during daytime (59–76%), while T0 (local propagation) dominates at night (37–57%). **Figure 7** provides a micro-geographic zoom confirming these patterns at the individual sensor level.

**Structural Disentanglement.** **Figure 6** presents the Jaccard similarity index of edge overlaps among the four active templates. The near-zero off-diagonal values (≤0.018, mean 0.009) demonstrate that the router extracts highly disjoint, mutually complementary topological priors — not slight variations of a single base graph. Each template contributes a structurally distinct edge set, assigned to different encoder depths through per-layer specialization.

**Quantitative Fidelity.** To align with XAI evaluation standards, **Table 2** reports three interpretability metrics. (1) *Fidelity*: substituting active templates with random graphs of identical sparsity causes MAE degradation of +0.44/+0.65/+0.99 — approximately 30–50× larger than frozen replay degradation (+0.01/+0.01/+0.02). This quantitatively demonstrates that learned templates encode faithful spatial dependencies that random graphs cannot replicate. (2) *Disentanglement*: the mean off-diagonal Jaccard index of 0.009 indicates mutually exclusive template structures. (3) *Locality*: hop-distance distributions (T0: 100% <3 hops; T2: 61% >5 hops) quantify the receptive-field characteristics that Figures 3 and 7 qualitatively suggest.

**Table 2: Quantitative Metrics for VQ Routing Interpretability.**

| Metric | Measurement | Value |
|---|---|---|
| Fidelity | ΔMAE (Random − Original) / (Frozen − Original) | +0.44/+0.65/+0.99 / +0.01/+0.01/+0.02 |
| Disentanglement | Mean Off-diagonal Jaccard | 0.009 |
| Locality (T0 / T2) | Edges <3 hops / >5 hops | 100.0% / 60.9% |

---

## 5. Discussion

Taken together, our experiments reveal an apparent paradox: inference-time routing decisions are largely dispensable (Section 4.3, Figure 2), yet removing the VQ routing mechanism during training causes the static per-layer baseline to underperform even the original DGSTA (Section 4.4).

The training trajectory observed in Figure 5 offers a plausible interpretation: the VQ router undergoes an extended phase of broad structural exploration — maintaining near-maximum entropy for the vast majority of training — before converging to stable layer-wise specialization. This prolonged exploration phase, enabled by codebook competition and Gumbel-Softmax routing, may serve as an implicit structural search mechanism. Static graphs, initialized randomly and optimized independently, bypass this exploration entirely, which may explain their failure to match VQ Router performance. We emphasize that this interpretation remains a plausible hypothesis; rigorous causal isolation of the training dynamics — including whether the entropy collapse occurs abruptly or gradually — is left for future work.

**Limitations.** (1) *Single dataset*: All observations are on PeMS08. The specific spatial semantics are inherently tied to the PeMS highway topology. Cross-dataset validation is needed to assess whether the overarching phenomenon of layer-wise specialization generalizes. (2) *Single backbone*: Findings may depend on DGSTA's strong attention mechanism. (3) *Seed variance*: σ≈0.10 MAE, comparable to observed gains; performance claims are therefore "competitive" rather than "significantly superior." (4) *Geographic interpretation*: Template semantic labels are descriptive characterizations supported by quantitative hop-distance statistics, not causal proofs.

**Implications.** Our observations raise questions for dynamic graph module design. If inference-time routing dynamicity contributes minimally, simpler mechanisms may suffice for certain applications. However, the failure of the static baseline suggests that the training-time dynamics of codebook competition provide optimization benefits that static parameterization cannot replicate. Understanding this distinction may inform more efficient designs — for instance, retaining codebook-based exploration during training while collapsing to static assignments at inference.

---

## 6. Conclusion

We present an empirical diagnosis of VQ-based graph routing behavior in traffic forecasting. On PeMS08, we observe that VQ routing converges to depth-wise template specialization with limited temporal dynamicity — inference-time routing decisions are largely dispensable, yet the routing mechanism proves essential during training. Geographic analysis reveals that learned templates encode interpretable spatial patterns (local propagation, corridor coupling) with high fidelity. We hope these observations — and the tension they highlight between training-time and inference-time contributions — inform the design and evaluation of dynamic graph modules in spatio-temporal forecasting.

---

## Appendix A: Multi-Seed Analysis

| Seed | @3 MAE | @6 MAE | @12 MAE |
|---|---|---|---|
| 1 | 11.894 | 12.406 | 13.228 |
| 0 | 12.117 | 12.570 | 13.306 |
| 2 | 12.056 | 12.565 | 13.354 |
| Mean ± Std | 12.022 ± 0.10 | 12.514 ± 0.08 | 13.296 ± 0.05 |

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
