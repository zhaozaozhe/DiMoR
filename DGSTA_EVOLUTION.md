# DGSTA 项目：从原始基线到当前最佳配置的完整演进记录

本文档供 LLM 阅读。描述 DGSTA 项目的原始架构、每次代码改动、当前最佳配置和实验记录。全部基于 git 历史和实际代码，不包含推测。

---

## Part 1: 原始 DGSTA 基线架构

### 1.1 项目概述

- **模型名**: DGSTA — Dynamic Graph Convolution and Spatio-Temporal Self-Attention Network
- **框架**: LibCity 交通预测框架
- **任务**: traffic_state_pred（交通流量预测）
- **主数据集**: PeMS08（170 个传感器，5 分钟间隔，输入 12 步 → 输出 12 步）
- **环境**: Python 3.10, PyTorch 2.9.1, CUDA 12, WSL Ubuntu 24.04
- **入口**: `run_model.py --task traffic_state_pred --model DGSTA --dataset PeMS08 --config_file PeMS08 --gpu_id 0`

### 1.2 核心文件

| 文件 | 作用 |
|---|---|
| `libcity/model/traffic_flow_prediction/DGSTA.py` | 模型定义（~540 行） |
| `libcity/data/dataset/dgsta_dataset.py` | DGSTADataset：DTW 计算、KShape 聚类、adj_mx/sd_mx/sh_mx |
| `libcity/executor/dgsta_executor.py` | DGSTAExecutor：lap_mx 计算、训练循环、early stopping |
| `libcity/config/model/traffic_state_pred/DGSTA.json` | 模型默认超参数 |
| `libcity/config/data/DGSTADataset.json` | 数据集默认配置 |
| `libcity/config/executor/DGSTAExecutor.json` | 训练控制默认配置 |
| `libcity/config/task_config.json` | 任务路由（model → executor → dataset） |
| `PeMS08.json`（项目根目录） | 最高优先级用户配置覆盖 |
| `run_model.py` | CLI 入口 |

### 1.3 配置合并优先级（低 → 高）

```
libcity/config/data/TrafficStateDataset.json
→ TrafficStatePointDataset.json
→ DGSTADataset.json
→ executor/TrafficStateExecutor.json
→ DGSTAExecutor.json
→ model/DGSTA.json
→ task_config.json
→ ./PeMS08.json（用户覆盖）
→ CLI --参数
```

### 1.4 原始 DGSTA 模型架构（DGSTA.py 核心组件）

#### 类定义顺序（由上到下）

1. `drop_path()` — 随机深度函数
2. `TokenEmbedding` — 输入特征线性投影（`feature_dim → embed_dim`）
3. `PositionalEncoding` — 正弦位置编码
4. `LaplacianPE` — Laplacian 特征向量投影（`lape_dim → embed_dim`）
5. `DataEmbedding` — 组合嵌入层：value + position + time-of-day + day-of-week + spatial(Laplacian) + temporal-prior(tempp)
6. `DropPath` — drop_path 的 Module 封装
7. `Chomp2d` — Conv2d 填充裁剪（定义但未被使用）
8. `nconv` — 图卷积传播：`einsum('ncvl,nwv→ncwl', x, A)`
9. `linear` — 1×1 Conv2d
10. `gcn` — 多阶图卷积网络（order=2, support_len=1）
11. `STSelfAttention` — **核心**：DGCNN 动态图 + GCN + 时序注意力 + 地理注意力
12. `Mlp` — GELU MLP
13. `STEncoderBlock` — 6 层编码器块（STSelfAttention + MLP + DropPath + LayerNorm）
14. `count_parameters()` — 参数计数工具
15. `norm_embedding()` — 对 tempp 矩阵做 top-5 稀疏化
16. `DGSTA` — 主模型类

#### DGCNN 动态图构造（STSelfAttention 内）

原始 DGSTA 的核心创新——时变图：

```python
# 构造时（STSelfAttention.__init__）
self.nodevec_p1 = nn.Parameter(randn(288, 40))    # 288 个 5 分钟槽的嵌入
self.nodevec_p2 = nn.Parameter(randn(N, 40))      # 源节点嵌入
self.nodevec_p3 = nn.Parameter(randn(N, 40))      # 目标节点嵌入
self.nodevec_pk = nn.Parameter(randn(40, 40, 40)) # 核心张量

# 前向（STSelfAttention.forward）
def dgconstruct(self, time_emb, src_emb, tgt_emb, core_emb):
    adp = einsum('ai,ijk→ajk', time_emb, core_emb)
    adp = einsum('bj,ajk→abk', src_emb, adp)
    adp = einsum('ck,abk→abc', tgt_emb, adp)
    return softmax(relu(adp), dim=2)   # → [B, N, N] 时变邻接矩阵
```

#### 原始 forward 流程

```
batch['X'][B,12,N,F] → DataEmbedding(x, lap_mx, tempp) → enc[B,12,N,64]
  → 6 × STEncoderBlock(enc, ind, geo_mask):
      STSelfAttention:
        1. DGCNN(ind) → adp [B,N,N]
        2. GCN(x, [adp]) → x [B,12,N,64]
        3. Temporal Attention (2 heads, 25% dim) → t_x [B,12,N,16]
        4. Geographic Attention (4 heads, 50% dim) → geo_x [B,12,N,32]
        5. concat(t_x, geo_x) → proj(48→64)  ← 注意：缺少 Semantic Attention
      → Mlp → DropPath → residual
  → skip_convs → end_conv1(12→12) → end_conv2(256→1) → output[B,12,N,1]
```

**关键缺陷**：`sem_num_heads=2` 被声明，DTW 矩阵被加载，但 Semantic Attention 在 forward 中从未计算。`proj` 输入维度为 `int(dim*3/4)=48` 而非 `dim=64`，因为只拼接了 temporal(25%) + geo(50%)。

### 1.5 原始关键超参数（PeMS08 运行时）

| 参数 | 值 | 来源 |
|---|---|---|
| embed_dim | 64 | DGSTA.json |
| skip_dim | 256 | DGSTA.json |
| enc_depth | 6 | DGSTA.json |
| geo_num_heads | 4 | PeMS08.json |
| sem_num_heads | 2 | PeMS08.json |
| t_num_heads | 2 | PeMS08.json |
| type_ln | "pre" | PeMS08.json |
| type_short_path | "hop" | DGSTA.json |
| far_mask_delta | 7 | PeMS08.json |
| dtw_delta | 5 | DGSTA.json |
| set_loss | "huber" | PeMS08.json |
| huber_delta | 2 | PeMS08.json |
| learner | "adamw" | DGSTA.json |
| learning_rate | 1e-3 | DGSTA.json |
| step_size | 2776 | PeMS08.json |
| use_curriculum_learning | true | DGSTA.json |

### 1.6 原始 PeMS08 基线性能

```
@3:  MAE=12.082, RMSE=20.654, masked_MAE=12.099, masked_MAPE=7.899, masked_RMSE=20.557
@6:  MAE=12.498, RMSE=21.690, masked_MAE=12.517, masked_MAPE=8.175, masked_RMSE=21.602
@12: MAE=13.204, RMSE=23.173, masked_MAE=13.224, masked_MAPE=8.701, masked_RMSE=23.086
```

---

## Part 2: 代码演进时间线

### 修改 1: tempp.npy 路径修复

**日期**: 2026-05-06（commit 前）
**文件**: `DGSTA.py`
**改动**: 1 文件，+2 −1

**问题**: `tempp.npy` 加载路径硬编码为 `/libcity/cache/dataset_cache/...`，在当前环境不存在。

**修复**:
```python
# 旧
tempp = np.load("/libcity/cache/dataset_cache/" + self.dataset + "/tempp.npy")

# 新
tempp_path = os.path.join(os.path.dirname(__file__), "..", "..",
                          "cache", "dataset_cache", self.dataset, "tempp.npy")
tempp = np.load(tempp_path)
```
同时新增 `import os`。

---

### 修改 2: Semantic Attention 补全（commit 4a23188）

**日期**: 2026-05-06
**文件**: `DGSTA.py`
**改动**: 1 文件，+27 −8

**问题**: STSelfAttention 中声明了 `sem_num_heads=2`、加载了 `dtw_matrix`，但：
- 无 `sem_q/k/v_conv` 投影层
- forward 中无语义注意力计算
- `proj = Linear(48, 64)` 只接收 temporal + geo 的拼接

**修复内容**:
1. `STSelfAttention.__init__`: 新增 `sem_q_conv`, `sem_k_conv`, `sem_v_conv`, `sem_attn_drop`
2. `STSelfAttention.forward`: 新增完整语义注意力计算块（仿照 geo_attn），新增 `sem_mask` 参数
3. `STSelfAttention.forward`: `torch.cat([t_x, geo_x, sem_x])` → `proj = Linear(64, 64)`
4. `STEncoderBlock.forward`: 签名新增 `sem_mask` 参数，传递给 STSelfAttention
5. `DGSTA.__init__`: 基于 `dtw_matrix >= dtw_delta` 构造 `sem_mask`
6. `DGSTA.forward`: 传递 `sem_mask` 给 encoder blocks

---

### 修改 3: DeepTrendNet 趋势分支（commit 2d9b01b）

**日期**: 2026-05-05
**文件**: `DGSTA.py` + `PeMS08.json`
**改动**: 2 文件

**新增模块**（从参考代码 `DGSTA_claude_modified.py` 提取）:

1. `MovingAvg` 类 — 移动平均池化
2. `SeriesDecomposition` 类 — 序列分解为 residual + trend
3. `DeepTrendNet` 类 — MLP 趋势预测网络（3 层全连接，mlp3 零初始化）

**DGSTA 改动**:
- `__init__`: 读取 `use_deep_trend` 配置（默认 false），条件构建 `input_decomp` 和 `deep_trend_net`
- `forward`: 当 `use_deep_trend=True` 时，提取 `x_flow = x[..., :feat_dim]`，分解为趋势分量，经 DeepTrendNet 预测 `trend_pred`，与 `main_pred` 融合
- `trend_fusion = nn.Parameter(torch.tensor(0.1))` — 可学习融合权重，初值 0.1

**PeMS08.json**: 新增 `"use_deep_trend": false`

---

### 修改 4: VQ Router + DelayConv 双引擎架构（commit 3b64792）

**日期**: 2026-05-07
**文件**: `DGSTA.py` + `PeMS08.json`
**改动**: 2 文件，+251 −103

**这是最大的一次改动。**从参考代码提取 VQ Router 和 DelayConv，与当前基线形成条件双引擎架构。

#### 新增模块

1. `DelayConv` — 深度可分离因果时域卷积，`kernel=(3,1)`, `groups=channels`, Dirac 初始化
2. `SpatialPatternRouter` — VQ 图路由器：
   - 10 个可学习图模板 `graph_codebook[10,N,N]`
   - Gumbel-Softmax 硬路由选择图模板
   - `adj_vq` (时变，来自 codebook) + `adj_adp` (静态，来自 node embeddings)
   - 时间一致性 loss: `mean(diff(logits[:,1:], logits[:,:-1]) ** 2)`
3. `sparsify_graph` — Top-20 稀疏化 + 温度缩放 + re-softmax

#### 类重排

`MovingAvg` 和 `SeriesDecomposition` 从 STEncoderBlock 之后移到 `Chomp2d` 之后（`gcn` 之前），因为 STSelfAttention 需要在 `__init__` 中引用它们。

#### STSelfAttention 双分支改造

**构造函数**:
```python
if use_vq_router:
    # VQ 管线：SeriesDecomp, VQ Router, LayerNorm, res_scale=0.1
    # support_len=2, proj=Linear(48,64)  ← 无 Semantic Attention
else:
    # 基线管线：DGCNN (nodevec_p1..pk), sem QKV
    # support_len=1, proj=Linear(64,64)  ← 有 Semantic Attention
gconv = gcn(32, 32, 0.3, support_len=support_len, use_delay_conv=use_delay_conv)
```

**forward VQ 分支**:
```
x → SeriesDecomp → x_res, x_trend
x_trend → VQ Router → adj_vq[B,T,N,N], adj_adp[N,N]
sparsify_graph(adj_vq), sparsify_graph(adj_adp)
x_res → reshape1 → GCN(x_res, [adj_vq, adj_adp]) → reshape2 → tanh
x_trend + GCN_out * 0.1 → LayerNorm
→ Temporal attn → Geo attn → concat → proj(48→64)
```

**forward 基线分支**: 完全保持修改 2 的 Semantic Attention 补全版行为。

#### gcn 改造

```python
gcn(c_in, c_out, dropout, support_len, order, use_delay_conv=False):
    if use_delay_conv:
        self.delay_conv = DelayConv(c_in_original, kernel=3)
        self.out_proj = Conv2d(c_out, c_out, 1)
    # forward: 条件应用 delay_conv + out_proj
```

#### 配置传递链

```
PeMS08.json
  "use_vq_router": false
  "use_delay_conv": false
    ↓
DGSTA.__init__ → STEncoderBlock → STSelfAttention → gcn / SpatialPatternRouter
```

#### 新增配置键

| 键 | 默认值 | 作用 |
|---|---|---|
| `use_vq_router` | false | 启用 VQ Router 完整管线 |
| `use_delay_conv` | false | 在 gcn 中启用 DelayConv |
| `consistency_weight` | 0.1 | VQ Router 一致性 loss 的权重 |

#### PeMS08.json 状态（commit 3b64792）

```json
{
    "far_mask_delta": 7,
    "use_deep_trend": false,
    "use_vq_router": false,
    "use_delay_conv": false
}
```

---

### 修改 5: nconv einsum → matmul 修复

**日期**: 2026-05-07（commit 3b64792 之后，未提交）
**文件**: `DGSTA.py`
**改动**: 6 行

**问题**: 原始 `nconv` 使用 `einsum('ncvl,nwv→ncwl', x, A)`。当 VQ Router 激活时，`adj_vq` 形状为 `[B,T,N,N]`（4 维），einsum 期望 A 为 3 维，运行时崩溃。

**修复**: 替换为参考代码的 matmul 版本，处理 2/3/4 维邻接矩阵：
```python
def forward(self, x, A):
    x = x.permute(0, 3, 2, 1)
    if A.dim() == 3:
        A = A.unsqueeze(1)
    x = torch.matmul(A, x)
    return x.permute(0, 3, 2, 1).contiguous()
```

---

### 修改 6: trend_fusion 条件初始化

**日期**: 2026-05-07（commit 3b64792 之后，未提交）
**文件**: `DGSTA.py`
**改动**: 1 行

**问题**: `trend_fusion = 0.1` 在所有模式下相同。实验发现 VQ Router 模式（DeepTrendNet + VQ + DelayConv 全开）下 `trend_fusion=0.1` 压制了 DeepTrendNet 的贡献，导致长期预测（@12）退化。

**修复**:
```python
# 旧
self.trend_fusion = nn.Parameter(torch.tensor(0.1))

# 新
self.trend_fusion = nn.Parameter(torch.tensor(1.0 if self.use_vq_router else 0.1))
```

**实验验证**: 此改动是唯一带来显著增益的单行改动（@3: 11.970→11.894, @6: 12.484→12.406, @12: 13.332→13.228）。

---

### 修改 7: adj_adp 时变图修复

**日期**: 2026-05-09（commit 3b64792 之后，未提交）
**文件**: `DGSTA.py` + `PeMS08.json`
**改动**: 2 文件，+20 −5

**问题**: VQ Router 的 `adj_adp = relu(nodevec1 @ nodevec2)` 生成的是静态 [N,N] 矩阵，不随时间变化。原始 DGCNN 的核心贡献就是时变图，VQ Router 的 adj_vq 已通过 codebook 路由实现时变，但 adj_adp 退化成了死图。

**修复**:

SpatialPatternRouter:
```python
# __init__ 新增
self.time_embed = nn.Embedding(288, embed_dim)  # 288 个 5 分钟槽

# forward 新增参数 ind=None
if self.use_time_aware and ind is not None:
    time_idx = ind % 288
    time_emb = self.time_embed(time_idx)         # [B, embed_dim]
    nodevec1_t = nodevec1 * time_emb.unsqueeze(1) # [B, N, embed_dim]
    adj_adp = relu(matmul(nodevec1_t, nodevec2.unsqueeze(0)))  # [B, N, N]
else:
    adj_adp = relu(mm(nodevec1, nodevec2))        # 原始静态 [N, N]
```

STSelfAttention.forward VQ 分支: `vq_router(x_trend, ind=ind, training=...)`

新配置键: `use_time_aware_adp`（默认 true）。

---

## Part 3: 当前最佳配置

### 3.1 当前 PeMS08.json（完整）

```json
{
    "dataset_class": "DGSTADataset",
    "input_window": 12,
    "output_window": 12,
    "train_rate": 0.6,
    "eval_rate": 0.2,
    "batch_size": 32,
    "add_time_in_day": true,
    "add_day_in_week": true,
    "step_size": 2776,
    "max_epoch": 300,
    "bidir": true,
    "far_mask_delta": 7,
    "geo_num_heads": 4,
    "sem_num_heads": 2,
    "t_num_heads": 2,
    "cluster_method": "kshape",
    "cand_key_days": 21,
    "seed": 1,
    "type_ln": "pre",
    "set_loss": "huber",
    "huber_delta": 2,
    "mode": "average",
    "use_deep_trend": true,
    "use_vq_router": true,
    "use_delay_conv": true,
    "use_time_aware_adp": true
}
```

### 3.2 当前 DGSTA.py 中所有 config-gated 模块

| 模块 | 配置键 | 默认值 | 当前值 | 激活时行为 |
|---|---|---|---|---|
| DeepTrendNet | `use_deep_trend` | false | true | SeriesDecomp + DeepTrendNet + trend_fusion 融合 |
| VQ Router | `use_vq_router` | false | true | 替代 DGCNN，VQ 双图路由 + SeriesDecomp + tanh 融合 |
| DelayConv | `use_delay_conv` | false | true | gcn 中应用因果时域卷积 + out_proj |
| Time-Aware adj_adp | `use_time_aware_adp` | true | true | adj_adp 从静态变为随时步变化 |
| Spatial Trend (失效) | `use_spatial_trend` | false | false | DeepTrendNet 中路网邻接空间混合 |
| Parametric Trend (失效) | `use_parametric_trend` | false | false | DeepTrendNet 中二次参数化趋势残差分支 |
| Conv Attention (失效) | `use_conv_attn` | false | false | 时序 Q/K/V 前加深度可分离时域卷积 |

当前 DGSTA.py 共 789 行。

### 3.3 当前 forward pass 完整数据流（全开模式）

```
batch['X'][B,12,N,F] ─┬─→ enc_embed_layer(x, lap_mx, tempp) → enc[B,12,N,64]
                       │      → 6 × STEncoderBlock(enc, ind, geo_mask, sem_mask):
                       │           STSelfAttention (VQ 分支):
                       │             1. SeriesDecomp(enc) → x_res, x_trend
                       │             2. VQ Router(x_trend, ind):
                       │                  Gumbel-Softmax 选图模板 → adj_vq [B,T,N,N]
                       │                  time_embed[ind%288] ⊙ nodevec1 @ nodevec2 → adj_adp [B,N,N]
                       │             3. sparsify_graph → Top-20 + softmax
                       │             4. GCN(x_res, [adj_vq, adj_adp]) → tanh
                       │             5. x_trend + GCN_out × 0.1 → LayerNorm
                       │             6. Temporal Attn → t_x [B,12,N,16]
                       │             7. Geographic Attn → geo_x [B,12,N,32]
                       │             8. concat(t_x, geo_x) → proj(48→64)  [无 Semantic]
                       │           → Mlp → DropPath → residual
                       │      → skip_convs 累加
                       │→ end_conv1(12→12) → end_conv2(256→1) → main_pred[B,12,N,1]
                       │
                       └─→ x_flow = x[..., :feat_dim]
                           → SeriesDecomp(x_flow) → x_trend
                           → DeepTrendNet(x_trend):
                                feature_proj → MLP(12→64→64→12, mlp3零初始化)
                           → trend_pred [B,12,N,1]
                           
main_pred + trend_fusion(1.0) × trend_pred → output [B,12,N,1]
```

### 3.4 与原始基线的差异汇总

| 差异点 | 原始基线 | 当前最佳配置 |
|---|---|---|
| 图构造 | DGCNN (`dgconstruct`, 288 时变) | VQ Router (`adj_vq` codebook 路由 + `adj_adp` 时变) |
| Semantic Attention | 声明但未计算 | VQ 模式下不存在；baseline 模式下已补全 |
| 趋势预测 | 无 | DeepTrendNet (MLP, trend_fusion=1.0) |
| GCN 时域卷积 | 无 | DelayConv (因果, kernel=3) |
| 一致性正则 | 无 | VQ Router consistency_loss (weight=0.1) |
| loss 计算 | curriculum learning only | curriculum learning + consistency_loss |
| proj 维度 | Linear(48,64) [缺 sem] | Linear(48,64) [VQ 模式无 sem] |

---

## Part 4: 完整实验记录

### 4.1 消融实验（论文核心表格）

基准：原始 DGSTA 基线 @3=12.082, @6=12.498, @12=13.204。

| Model | VQ | Trend | Delay | @3 MAE | @6 MAE | @12 MAE | Δ@3 | Δ@6 | Δ@12 |
|---|---|---|---|---|---|---|---|---|---|
| DGSTA (baseline) | — | — | — | 12.082 | 12.498 | 13.204 | — | — | — |
| **Full** | ✓ | ✓ | ✓ | **11.894** | **12.406** | **13.228** | **−1.6%** | **−0.7%** | **+0.2%** |
| − VQ Router | ✗ | ✓ | ✓ | 12.176 | 12.645 | 13.493 | +2.4% | +1.9% | +2.0% |
| − DeepTrendNet | ✓ | ✗ | ✓ | 11.912 | 12.421 | 13.222 | +0.2% | +0.1% | −0.0% |
| − DelayConv | ✓ | ✓ | ✗ | 12.176 | 12.622 | 13.348 | +2.4% | +1.7% | +0.9% |

**消融结论**:
- **VQ Router** 是最大贡献者：关掉后 @3/@6/@12 全面退化 2%+
- **DelayConv** 次之：关掉后 @3 退化 2.4%，@12 退化 0.9%
- **DeepTrendNet** 在推理时贡献接近零：关掉后 @3/+0.2%, @6/+0.1%, @12/−0.0%

### 4.2 参数探索实验（均基于 Full 配置叠加）

| # | 改动 | @3 | @6 | @12 | 结论 |
|---|---|---|---|---|---|
| 1 | trend_fusion 0.1→1.0 | 11.970→11.894 | 12.484→12.406 | 13.332→13.228 | ★ 唯一有效改动 |
| 2 | hard=False（软路由） | 11.900 | 12.455 | 13.324 | ❌ 全面退化 |
| 3 | use_spatial_trend | 11.938 | 12.412 | 13.242 | ❌ 退化 |
| 4 | kernel=13, tau=0.5 | 11.880 | 12.407 | 13.233 | 噪声 |
| 5 | use_parametric_trend | 11.893 | 12.421 | 13.255 | ❌ 退化 |
| 6 | far_mask_delta=5 | 11.947 | 12.447 | 13.263 | ❌ 退化 |
| 7 | use_conv_attn | — | — | — | ❌ 退化 |
| 8 | use_time_aware_adp | — | — | — | ❌ 退化 |
| 9 | use_balance_loss | 11.964 | 12.457 | 13.283 | ❌ 退化 |

### 4.3 实验 cache ID 对照

| 实验 | Exp ID | 配置 |
|---|---|---|
| 原始基线 | — | all off |
| Full (最佳) | 71098 | VQ+Trend+Delay |
| −VQ Router | 68783 | Trend+Delay |
| −DeepTrendNet | 64832 | VQ+Delay |
| −DelayConv | 43876 | VQ+Trend |

---

## Part 5: 已知盲点

### 盲点 1: adj_adp 时变（已实现，待验证）

当前实验 #4 运行中。adj_adp 从静态 [N,N] 变为时变 [B,N,N]。

### 盲点 2: VQ Codebook 多样性

10 个图模板无相互排斥约束，训练中可能坍缩。可加 diversity loss（模板间 cosine similarity 惩罚）。

### 盲点 3: VQ Router 仅以 x_trend 为输入

路由决策只用趋势分量，未利用 residual 分量的短期波动信号。

### 盲点 4: DeepTrendNet 无空间交互

DeepTrendNet 逐节点独立预测趋势，节点间不交换信息。路网邻接矩阵实验（盲点 5 的 spatial_trend）已失败，但 DTW 相似度矩阵可能是更好的空间图。

### 盲点 5: 更多 Codebook 模板或其他改进

当前 10 个模板，可尝试 15/20 个。或尝试不同温度。

---

## 附录 A: 实验命令

```bash
conda activate ai_lab
python run_model.py --gpu_id 0
```

## 附录 B: 从 git 复现任一阶段

```bash
git checkout <commit_hash>                    # 恢复代码
# 手动应用 PeMS08.json 中的配置开关            # 恢复实验配置
python run_model.py --gpu_id 0               # 运行
```

---

## Part 6: VQ Router 机制分析（2026-05-10）

### 6.1 分析动机

VQ Router 为模型带来了 ~1.6% 的 @3 MAE 提升，但其内部工作机制未知。原始假设：
> "VQ Router 通过 Gumbel-Softmax 路由从 10 个可学习图模板中动态选择，实现时变图结构。"

需要验证：(1) 10 个模板是否真正被使用？(2) 路由是否随时步/交通状态变化？

### 6.2 分析方法

在最佳模型 checkpoint（实验 71098，@3=11.894 @6=12.406 @12=13.228）上，运行 `analysis/vq_router/analyze_vq_router.py`，收集：
- **Soft routing probs**: `softmax(logits/tau)` 的每模板概率
- **Hard routing IDs**: `Gumbel-softmax(hard=True).argmax()` 的实际选中模板
- **Time-of-day 标注**: 每个样本对应的小时（0-23）

### 6.3 关键发现

#### 发现 1: Codebook 多样性健康

10 个图模板的 pairwise cosine similarity 接近 0（0.003–0.077），模板本身高度差异。不存在 codebook collapse。

#### 发现 2: Hard Routing 严重坍缩

在无 balance loss 的自然训练状态下，hard routing 行为：
- 每层几乎只使用 1 个模板（100% 选中概率）
- 不同层选择了**不同的**主导模板

```
Layer 0: 永远 T6 (100%)
Layer 1: T2/T0/T3 随时间切换 ← 唯一有动态行为的层
Layer 2: 永远 T4 (100%)
Layer 3: 永远 T0 (100%)
Layer 4: 永远 T0 (100%)
Layer 5: 永远 T0 (100%)
```

#### 发现 3: Soft/Hard Entropy 分离

添加 Balance Loss（λ=0.001）后：
- soft H_norm 保持 0.91（看起来"均匀"）
- 但 4/6 层的 hard H_norm = 0.00（100% 选单个模板）
- MAE 全面退化（@3: +0.07, @6: +0.05, @12: +0.06）

证实了 soft/hard routing 的经典脱钩现象。

#### 发现 4: 动态性仅存在于 1/6 层

时间条件分析显示：
- 5/6 层：路由选择不随时段变化（早高峰=午平峰=深夜）
- Layer 1：白天偏好 T2（59–76%），夜间偏好 T0（37–57%）

### 6.4 机制解释

**原假设（时变图路由）被否证。** VQ Router 的实际工作机制更接近：

> **层间图专业化（Depth-wise Graph Specialization）**

即：10 个模板提供了"图结构词汇表"，6 个编码层各选择一个最适合其感受野的模板。增益来自多模板容量（每层不同图）而非时变适应性（同层不同时刻不同图）。

```
原叙事:  VQ Router → 时变动态图 → 性能提升
修正叙事: VQ Router → 层间异质图先验 → 容量提升 → 性能提升
```

### 6.5 审稿人可能的问题与回答

**Q**: "既然每层固定模板，为什么不直接学 6 个静态邻接矩阵？"

**A**: (1) 多模板机制在训练初期提供了探索空间，模型收敛到层专业化是 emergent behavior 而非预设；(2) Layer 1 保留了时变行为，说明路由器有潜力在需要时变信息的层做出动态选择；(3) 这引出了一个更深刻的问题：交通预测中动态图的需求是否本身具有**层级不均匀性**——某些层需要动态适应，某些层仅需稳定拓扑先验。

### 6.6 对未来工作的启发

1. **显式层专业化图学习**：用 6 个可学习静态图替代 VQ Router 的 10 模板路由，可能以更少参数达到相同效果
2. **选择性动态路由**：仅在特定层（如 Layer 1）保留动态路由，其余层使用固定图
3. **动态性需求的层级分析**：研究不同深度对时变图的需求差异，可能揭示 GNN 深度与时间建模的基本关系

### 6.7 新增分析工具

- `analysis/vq_router/analyze_vq_router.py`: 时间条件路由分析脚本
- `analysis/vq_router/time_conditioned_routing.txt`: 分析报告
- `analysis/vq_router/vq_analysis_report.txt`: 早期分析报告

### 6.8 相关代码改动

为支持机制分析，对 DGSTA.py 做了以下 instrumentation（不影响训练行为）：
- `SpatialPatternRouter.forward`: 新增 `self.routing_probs` (soft probs for analysis)
- `STSelfAttention.forward`: 新增 `self.last_router_weights` (hard selection for analysis)
- `DGSTA.calculate_loss_without_predict`: 新增 `use_balance_loss` 条件块（已关闭）
