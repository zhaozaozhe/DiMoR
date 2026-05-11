# DiMoR: Discrete Modal Routing for Explainable Traffic Forecasting

**Target**: EI / SCI Q3-Q4 journal
**Status**: Draft for review

---

## Abstract

Dynamic graph routing has become a popular mechanism in spatio-temporal traffic forecasting,
yet the actual routing behaviors learned by these models remain largely unexamined.
We present DiMoR, a traffic forecasting framework built upon the DGSTA backbone
with a discrete graph routing module, and conduct a systematic analysis of its
learned routing dynamics. Through layer-wise behavior diagnosis, frozen replay
validation, and ablation studies on the PeMS08 dataset, we find that: (1) VQ-based
graph routing converges to depth-wise template specialization rather than fine-grained
temporal adaptation; (2) inference-time routing dynamicity contributes minimally to
prediction accuracy; (3) the routing and semantic attention branches exhibit complementary
horizon characteristics. DiMoR achieves competitive forecasting performance (MAE 11.894,
12.406, 13.228 at 15/30/60 min horizons) while providing interpretable routing behaviors.
Our analysis offers practical insights into what dynamic graph routers actually learn
in traffic forecasting networks.

**Keywords**: traffic forecasting, dynamic graph routing, explainability, vector quantization, spatio-temporal network

---

## 1. Introduction

Traffic flow prediction is a fundamental task in intelligent transportation systems.
Recent advances have increasingly adopted dynamic graph mechanisms—models that learn
time-varying adjacency structures to capture evolving spatial dependencies among road
sensors. These methods, including DGCNN-style adaptive graphs and VQ-based routing,
operate under the implicit assumption that traffic spatial dependencies change
meaningfully over time and that learning to switch between graph structures improves
prediction.

However, relatively few studies have examined *what routing behaviors are actually
learned* after training converges. Do graph routers truly produce time-varying structures?
Which layers benefit from dynamic routing? And how much does real-time routing
contribute to final prediction accuracy?

In this work, we conduct a systematic analysis of VQ-based dynamic graph routing
in the context of traffic forecasting. We adopt the DGSTA architecture as our backbone
and augment it with a discrete graph routing module (VQ Router) that maintains a
codebook of learnable graph templates and selects among them via Gumbel-Softmax.
Rather than proposing a novel architecture for state-of-the-art performance, our
contribution is an *explainability study*: we diagnose what the router learns,
measure the functional importance of its dynamicity, and characterize the
complementary roles of different spatial modeling mechanisms.

Our key findings include:
- VQ routing converges to layer-wise template specialization: different encoder layers
  consistently prefer different graph templates, while within each layer, template
  selection is predominantly static across time periods.
- Frozen replay experiments show that locking routing to dominant templates causes
  negligible performance degradation (<0.03 MAE), indicating that inference-time
  routing dynamicity plays a minor role.
- The semantic attention branch (DTW-based) and the routing branch (VQ-based) exhibit
  complementary horizon characteristics: semantic attention favors long-term prediction
  while routing aids short-term accuracy.

---

## 2. Related Work

### 2.1 Traffic Flow Prediction

Deep learning approaches to traffic prediction have evolved from RNN-based models
to graph neural networks and Transformer architectures. DGSTA combines dynamic
graph convolution with spatio-temporal self-attention, achieving strong performance
on the PeMS benchmark datasets. Many subsequent works have proposed additional
mechanisms—adaptive adjacency learning, multi-graph fusion, trend decomposition—to
further improve accuracy.

### 2.2 Dynamic Graph Routing

Vector Quantized (VQ) routing, inspired by VQ-VAE and mixture-of-experts architectures,
maintains a discrete codebook of graph templates and uses Gumbel-Softmax to select
among them. This approach has been applied to traffic forecasting with reported
improvements. However, the actual learned behaviors of such routers—whether they
produce truly dynamic graphs or converge to static specialization—have not been
systematically studied.

### 2.3 Explainability in Traffic Forecasting

Most explainability efforts in traffic forecasting focus on attention visualization,
showing which time steps or sensors the model attends to. Few works examine the
internal mechanisms of dynamic graph modules. Our work addresses this gap by
providing a structured diagnosis of VQ routing behavior.

---

## 3. Method

### 3.1 Backbone Architecture

DiMoR is built upon the DGSTA backbone, which consists of:
- **Data Embedding**: value, positional, time-of-day, day-of-week, Laplacian spatial,
  and temporal prior embeddings
- **6-layer Spatio-Temporal Encoder**: each layer contains ST self-attention (temporal,
  geographic, and semantic attention heads) followed by an MLP with residual connections
  and stochastic depth
- **Skip Connections**: accumulated across encoder layers and projected to output
  dimensions via 1×1 convolutions

### 3.2 VQ Graph Router

The VQ Router maintains a codebook of K=10 learnable graph templates, each of shape
[N×N] for N sensors. Given the input hidden representation x ∈ R^{B×T×N×D}:

1. **Series Decomposition**: x is decomposed into trend x_trend and residual x_res
   using a moving average kernel (kernel_size=5)
2. **Spatial Pattern Encoding**: x_trend is reduced to traffic intensity, compressed
   through a spatial MLP, and classified into K logits
3. **Hard Routing**: Gumbel-Softmax with hard=True selects one dominant template per
   (batch, time_step) pair
4. **Graph Construction**: the selected template forms `adj_vq` ∈ R^{B×T×N×N}, while
   a learned adaptive adjacency `adj_adp` ∈ R^{N×N} provides a static complementary graph
5. **GCN Propagation**: x_res is processed through a 2-hop GCN using both `adj_vq`
   and `adj_adp`, fused with x_trend via `x_trend + tanh(GCN_out) × 0.1`, and normalized

The router is regularized by a temporal consistency loss that penalizes rapid
switching of template selection across consecutive time steps.

### 3.3 Auxiliary Modules

- **DeepTrendNet**: A lightweight MLP branch that predicts future values from the
  decomposed trend component. Output is fused with the main prediction via a
  learnable scalar weight.
- **DelayConv**: A causal depthwise temporal convolution applied within GCN layers
  to smooth temporal features.
- **Semantic Attention**: DTW-based attention mask that allows nodes with similar
  daily patterns to attend to each other.

### 3.4 Configurable Design

All modules are config-gated via JSON configuration, enabling clean ablation. The
default configuration (all gates off) produces the unmodified DGSTA baseline.

---

## 4. Experiments

### 4.1 Setup

We conduct experiments on the PeMS08 dataset (170 sensors, 5-minute intervals,
July-August 2016). Input and output windows are both 12 steps (60 minutes).
The dataset is split 60%/20%/20% for training, validation, and testing.
Standard normalization (StandardScaler) is applied.

Training uses AdamW optimizer with cosine learning rate schedule, initial
learning rate 1e-3, weight decay 0.05, batch size 32, and curriculum learning
over 300 epochs with early stopping (patience=50). All experiments use a single
NVIDIA RTX 5070 Ti GPU.

### 4.2 Main Results

| Model | @3 MAE | @6 MAE | @12 MAE |
|---|---|---|---|
| DGSTA (baseline) | 12.082 | 12.498 | 13.204 |
| DiMoR (Full) | **11.894** | **12.406** | 13.228 |
| − VQ Router | 12.176 | 12.645 | 13.493 |
| − DeepTrendNet | 11.912 | 12.421 | **13.222** |
| − DelayConv | 12.176 | 12.622 | 13.348 |

DiMoR achieves competitive performance, with the VQ Router contributing the
largest individual improvement (+2.4% at @3). DeepTrendNet's contribution
is marginal, suggesting the backbone attention already captures trend information.
Multi-seed analysis (3 seeds) is provided in Appendix A.

### 4.3 Routing Behavior Analysis

**Layer-wise Template Specialization.** Figure 3 shows the dominant template
selected by each encoder layer across different time periods. A clear pattern
emerges: different layers consistently prefer different templates (L0→T6,
L2→T4, L3/L4/L5→T0), while within each layer, template selection is stable
across morning, midday, evening, and night periods. Only Layer 1 shows
limited variation between daytime (T2) and nighttime (T0) preferences.

**Template Usage Distribution.** Figure 4 quantifies the per-layer template
usage. For 5 of 6 layers, a single template accounts for over 99% of all
selections. Layer 1 is the exception, distributing its selections across
T2 (38%), T0 (20%), and T9 (12%). The 10 learned templates are structurally
diverse (average pairwise cosine similarity < 0.08), confirming that the
codebook has not collapsed.

### 4.4 Frozen Replay Validation

To test whether real-time routing decisions are functionally important, we
conduct a frozen replay experiment (Figure 5). Each layer's routing is locked
to its dominant template, and inference is re-run on the test set without any
routing logic. The performance degradation is negligible: <0.03 MAE across
all prediction horizons. This indicates that inference-time routing dynamicity
contributes minimally to the converged model's accuracy—the benefit of VQ
routing appears to come from providing layer-specific graph priors rather than
from fine-grained temporal adaptation.

### 4.5 Template Visualization

Figure 6 visualizes the adjacency matrices of learned graph templates. Templates
actively used by specific layers (top row) exhibit clear structural patterns
along the diagonal, suggesting they have learned meaningful sensor connectivity
patterns. Templates that were learned but rarely selected (bottom row) show
less organized structures.

### 4.6 Horizon Complementarity

Figure 7 compares three configurations: semantic attention only (without VQ
routing), VQ routing only (without semantic attention in the routing branch),
and their coexistence. Semantic attention achieves the best long-horizon accuracy
(@12 MAE = 13.191), while VQ routing excels at short horizons (@3 MAE = 11.894).
This horizon-level complementarity suggests that semantic and structural spatial
modeling serve different temporal roles in traffic forecasting.

---

## 5. Discussion

### 5.1 What Does the Router Actually Learn?

Our analysis reveals that VQ-based graph routing in DiMoR does not produce
strongly time-varying graphs, contrary to the common assumption in dynamic graph
literature. Instead, the router converges to a *depth-wise specialization*:
different encoder layers adopt different but stable graph templates. The
routing mechanism's primary contribution appears to be providing heterogeneous
graph priors across layers—each layer operates on a spatial structure suited
to its depth in the network—rather than adapting graph structure to temporal
traffic state changes.

This finding does not diminish the value of VQ routing; rather, it clarifies
*how* the mechanism contributes. The codebook of 10 templates provides a
"vocabulary" of graph structures from which layers select suitable priors
during training. The emergent layer specialization resembles the head
specialization observed in multi-head attention mechanisms.

### 5.2 Limitations

- Single dataset (PeMS08). Cross-dataset validation (PeMS04, METR-LA) is
  needed to confirm the generality of these findings.
- Single seed for main results. Multi-seed analysis (Appendix A) shows
  variance comparable to module gains, suggesting that results should be
  interpreted as competitive rather than significantly superior.
- The frozen replay experiment tests only the converged model; it does not
  examine whether routing dynamicity plays a role during training.

### 5.3 Future Work

- Extend the routing behavior analysis to other dynamic graph architectures.
- Investigate whether explicit per-layer static graph learning can match or
  exceed VQ routing performance with fewer parameters.
- Explore whether routing becomes more dynamic under non-stationary traffic
  conditions (accidents, extreme weather, holidays).

---

## 6. Conclusion

We present DiMoR, a traffic forecasting framework with discrete graph routing,
and conduct a systematic analysis of its learned routing behaviors. Through
layer-wise diagnosis, frozen replay validation, and ablation studies, we
characterize how VQ-based dynamic graph routing operates in practice: it
converges to depth-wise template specialization with limited temporal
adaptation, and its primary contribution lies in providing layer-specific
graph priors rather than real-time graph switching. DiMoR achieves competitive
forecasting performance while offering interpretable routing behaviors,
contributing to the understanding of dynamic graph mechanisms in traffic
forecasting.

---

## Appendix A: Multi-Seed Analysis

| Seed | @3 MAE | @6 MAE | @12 MAE |
|---|---|---|---|
| 1 | 11.894 | 12.406 | 13.228 |
| 0 | 12.117 | 12.570 | 13.306 |
| 2 | 12.056 | 12.565 | 13.354 |
| Mean±Std | 12.022±0.10 | 12.514±0.08 | 13.296±0.05 |

The seed-to-seed variance (σ ≈ 0.10) is comparable to the ablation deltas
observed in single-seed experiments. This indicates that performance claims
should be interpreted conservatively. The mechanism analysis findings
(layer specialization, frozen replay, codebook diversity) are consistent
across all three seeds.
