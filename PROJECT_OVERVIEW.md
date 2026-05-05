# DGSTA Project — Complete Technical Overview

This file is an LLM-facing technical overview. It is intended for fast orientation and may lag behind code. For implementation decisions, always verify against source files and git diff.

## 1. Project Identity

- **Model name**: DGSTA — **D**ynamic **G**raph Convolution and **S**patio-**T**emporal Self-**A**ttention Network
- **Framework**: Built on LibCity (a traffic prediction research framework)
- **Task**: Traffic flow prediction (`traffic_state_pred`)
- **Target variable**: `traffic_flow` (single output dimension per sensor)
- **Temporal granularity**: 5-minute intervals (`time_intervals: 300` seconds)
- **Input/Output**: 12 historical steps → 12 future steps (i.e., 60 min → 60 min)
- **Primary dataset**: PeMS08 (~170 sensors, California highway traffic)
- **Git repo**: `git@github.com:lzmmm30/DGSTA.git` (branch `claude-dgsta-improvement`; main is `master`)
- **Environment**: Conda env `ai_lab`, Python 3.10, PyTorch 2.9.1, CUDA 12.x, WSL Ubuntu 24.04

---

## 2. Directory Tree (Files You Need to Know About)

```
<project_root>/
├── run_model.py                              # Entry point CLI
├── PeMS08.json                               # Per-dataset hyperparam override (primary)
├── PeMS03.json / PeMS04.json / PeMS07.json / PEMS-BAY.json   # Other dataset configs
├── PeMS08.npy (etc.)                         # Precomputed adjacency matrices per dataset
├── requirements.txt                          # Original frozen deps (outdated, for reference)
├── env_ai_lab_before_codex.yml               # Actual conda env snapshot
│
├── raw_data/                                 # Raw LibCity-format atomic files
│   ├── PeMS08/
│   │   ├── PeMS08.dyna                       # Time-series: [geo_id, type, timestep, traffic_flow, ...]
│   │   ├── PeMS08.geo                        # Geo entities: [geo_id, type, coordinates]
│   │   ├── PeMS08.rel                        # Relations: [origin_id, destination_id, cost]
│   │   └── config.json                       # Data schema + metadata
│   ├── PeMS03/   (same structure, 227 sensors, 404 MB .dyna)
│   ├── PeMS04/   (same structure, 307 sensors, 286 MB .dyna)
│   ├── PeMS07/   (same structure, 883 sensors, 1.1 GB .dyna — traffic_flow only)
│   └── PEMS-BAY/ (same structure, 325 sensors, 727 MB .dyna)
│
├── libcity/
│   ├── config/
│   │   ├── config_parser.py                  # Hierarchical JSON config merger
│   │   ├── task_config.json                  # Task routing table
│   │   ├── model/traffic_state_pred/DGSTA.json   # Model defaults
│   │   ├── data/DGSTADataset.json            # Dataset defaults
│   │   ├── data/TrafficStatePointDataset.json    # Parent dataset defaults
│   │   ├── data/TrafficStateDataset.json     # Grandparent dataset defaults
│   │   ├── executor/DGSTAExecutor.json       # Training control defaults
│   │   ├── executor/TrafficStateExecutor.json    # Parent executor defaults
│   │   └── evaluator/TrafficStateEvaluator.json  # Metrics config
│   │
│   ├── data/
│   │   ├── dataset/
│   │   │   ├── abstract_dataset.py           # Base class
│   │   │   ├── traffic_state_dataset.py      # Middle class
│   │   │   ├── traffic_state_point_dataset.py # Point-based dataset (train/val/test split)
│   │   │   └── dgsta_dataset.py             # DGSTA-specific: DTW, clustering, pattern keys
│   │   ├── batch.py                          # Batch collation
│   │   ├── list_dataset.py                   # List-style dataset wrapper
│   │   └── utils.py                          # Data utilities
│   │
│   ├── executor/
│   │   ├── abstract_executor.py              # Base executor
│   │   ├── traffic_state_executor.py         # GPU management, train/eval loop
│   │   ├── dgsta_executor.py                # DGSTA-specific: lap_mx forwarding
│   │   └── scheduler.py                      # LR scheduler factory
│   │
│   ├── evaluator/
│   │   ├── abstract_evaluator.py             # Base evaluator
│   │   ├── traffic_state_evaluator.py        # Metric computation orchestrator
│   │   └── eval_funcs.py                     # Metric implementations
│   │
│   ├── model/
│   │   ├── abstract_model.py                 # nn.Module subclass with predict/calculate_loss
│   │   ├── abstract_traffic_state_model.py   # Adds data_feature accessor
│   │   ├── loss.py                           # 13 loss functions
│   │   └── traffic_flow_prediction/
│   │       └── DGSTA.py                      # ★ THE CORE MODEL FILE (536 lines)
│   │
│   ├── pipeline/
│   │   └── pipeline.py                       # run_model(), hyper_parameter(), finetune()
│   │
│   ├── utils/
│   │   ├── argument_list.py                  # CLI argument registry
│   │   ├── normalization.py                  # StandardScaler / MinMaxScaler / None
│   │   ├── utils.py                          # get_model(), get_executor(), get_logger(), etc.
│   │   └── distributed.py                    # Distributed training helpers
│   │
│   ├── cache/
│   │   ├── dataset_cache/
│   │   │   ├── dgsta_point_based_PeMS08_*.npz  # Preprocessed train/val/test arrays
│   │   │   ├── PeMS08/tempp.npy               # Temporal prior per dataset
│   │   │   ├── dtw_PeMS08.npy                 # Precomputed DTW distance matrix
│   │   │   └── pattern_keys_kshape_PeMS08_*.npy # KShape clustering output
│   │   └── 99049/                              # Exp ID directory with model_cache/*.tar
│   │
│   └── log/                                    # Historical training logs (~450 .log files)
│
├── experiments/
│   ├── PeMS08_ablation_log.md                  # Ablation experiment tracking template
│   └── logs/                                   # Recent run logs
│
└── references/
    └── claude_modified/                        # Prior-work backups (gitignored)
```

---

## 3. Entry Point & Runtime Flow

### 3.1 Command Line

```bash
# Full PeMS08 training with GPU override
python run_model.py --gpu_id 0

# Equivalent explicit form:
python run_model.py --task traffic_state_pred --model DGSTA \
    --dataset PeMS08 --config_file PeMS08 --train True --saved_model True --gpu_id 0
```

The script (`run_model.py`) is an argparse wrapper that delegates to `libcity.pipeline.run_model()`.

### 3.2 Pipeline Flow

```
CLI args
  │
  ▼
ConfigParser(task, model, dataset, config_file, other_args)
  │  Merges JSONs bottom-up: base data/exec/model JSONs → task_config.json → dataset .json → CLI args
  │
  ▼
get_dataset(config)          # Instantiates DGSTADataset
  │  Loads raw .dyna/.geo/.rel → normalizes → caches as .npz
  │  Computes DTW, adjacency, short-path, KShape clustering
  │
  ▼
get_data_feature()           # Returns dict: scaler, adj_mx, sd_mx, sh_mx, dtw_matrix, pattern_keys, etc.
  │
  ▼
get_model(config, data_feature)  # Instantiates DGSTA model (= DGSTA.__init__)
  │
  ▼
model.to(device)
  │
  ▼
get_executor(config, model, dataset)  # Instantiates DGSTAExecutor
  │
  ▼
executor.train(train_data, eval_data)   # Training loop
  │  For each epoch:
  │    For each batch:
  │      model.calculate_loss(batch, batches_seen, lap_mx=None)
  │        → model.forward(batch) → prediction
  │        → inverse_scale → curriculum_filter → loss_fn → backward
  │
  ▼
executor.evaluate(test_data)   # Compute MAE, RMSE, MAPE (masked + unmasked)
```

### 3.3 Config Merging Priority (lowest to highest)

1. `libcity/config/data/TrafficStateDataset.json` (grandparent defaults)
2. `libcity/config/data/TrafficStatePointDataset.json` (parent defaults)
3. `libcity/config/data/DGSTADataset.json` (dataset defaults)
4. `libcity/config/executor/TrafficStateExecutor.json`
5. `libcity/config/executor/DGSTAExecutor.json`
6. `libcity/config/evaluator/TrafficStateEvaluator.json`
7. `libcity/config/model/traffic_state_pred/DGSTA.json`
8. `libcity/config/task_config.json` (task routing only)
9. `./{config_file}.json` (top-level per-dataset override, e.g. `PeMS08.json`)
10. CLI `other_args` (highest priority)

`ConfigParser.get(key)` walks this chain bottom-up and returns the first hit.

---

## 4. Data Pipeline Details

### 4.1 Raw Data Format (LibCity Atomic Files)

**`.dyna`**: Tab-separated, header: `dyna_id, type, time, entity_id, traffic_flow [, traffic_occupancy, traffic_speed]`
Each row is one sensor at one 5-min timestep.

**`.geo`**: Tab-separated, header: `geo_id, type, coordinates`
One row per sensor.

**`.rel`**: Tab-separated, header: `rel_id, type, origin_id, destination_id, cost`
Pairwise sensor distances (used to build adjacency).

**`config.json`**: Data schema metadata:
```json
{
  "info": {
    "data_col": ["traffic_flow"],
    "weight_col": "cost",
    "output_dim": 1,
    "time_intervals": 300,
    "init_weight_inf_or_zero": "zero",
    "set_weight_link_or_dist": "link",
    "calculate_weight_adj": false,
    "weight_adj_epsilon": 0.1
  }
}
```

### 4.2 DGSTADataset — What It Computes

Inherits: `AbstractDataset → TrafficStateDataset → TrafficStatePointDataset → DGSTADataset`

**During `__init__`**, it:
1. Loads raw `.dyna` → DataFrame → pivot table (timesteps × sensors × features)
2. StandardScaler normalization (on training portion only)
3. Splits: train/eval/test along time axis (e.g. 60%/20%/20%)
4. Builds **adjacency matrix** via Gaussian kernel on pairwise road-network distances
5. Builds **shortest distance matrix** (`sd_mx`) and **shortest hop matrix** (`sh_mx`) via Floyd-Warshall
6. Computes **DTW distance matrix** between all pairs of sensor time series (cached as `.npy`)
7. Runs **KShape clustering** on daily patterns → produces `pattern_keys` (one cluster ID per sensor-day)
8. Computes `tempp.npy` (temporal prior) from the training portion
9. Caches the processed tensors as `.npz` (filename encodes all hyperparams)

**`get_data_feature()` returns**:
```
scaler, adj_mx, sd_mx, sh_mx, ext_dim, num_nodes, feature_dim, output_dim,
num_batches, dtw_matrix, pattern_keys
```

### 4.3 Batch Format

Each batch is a tuple: `(x, y)` where:
- `x`: `(B, input_window=12, N, feature_dim)` — for PeMS08, feature_dim ≈ 3 (flow + time_in_day + day_in_week)
- `y`: `(B, output_window=12, N, 1)` — future traffic flow

The `ind` (time-of-day index, used in dynamic graph) is passed through the batch as a separate item — extracted inside `DGSTA.forward()`.

---

## 5. DGSTA Model Architecture — Complete Breakdown

File: `libcity/model/traffic_flow_prediction/DGSTA.py` (536 lines)

### 5.1 Inheritance
```
nn.Module → AbstractModel → AbstractTrafficStateModel → DGSTA
```

### 5.2 Classes Defined in DGSTA.py

| Class | Lines | Purpose |
|---|---|---|
| `drop_path()` | 16 | Stochastic depth function |
| `TokenEmbedding` | 27 | Linear projection `input_dim → embed_dim` |
| `PositionalEncoding` | 39 | Sinusoidal temporal position encoding |
| `LaplacianPE` | 58 | Linear projection `lape_dim → embed_dim` for spatial position |
| `DataEmbedding` | 68 | **Combines ALL embeddings**: value + pos + time-of-day + day-of-week + spatial(Laplacian) + temporal-prior(tempp) |
| `DropPath` | 107 | Module wrapper for `drop_path` |
| `Chomp2d` | 116 | Conv padding removal (defined, possibly unused) |
| `nconv` | 125 | Graph convolution: `einsum('ncvl,nwv→ncwl', x, A)` |
| `linear` | 134 | 1×1 Conv2d |
| `gcn` | 143 | Multi-order GCN: for each support, computes `A¹ @ x, A² @ x, ... A^order @ x`, concats, then MLP |
| `STSelfAttention` | 167 | **THE CORE**: Dynamic graph + temporal attention + spatial attention |
| `Mlp` | 275 | `Linear → GELU → Drop → Linear → Drop` |
| `STEncoderBlock` | 294 | One encoder block: STSelfAttention + Mlp, residual, LayerNorm, DropPath |
| `count_parameters` | 327 | Utility |
| `norm_embedding` | 330 | Keep top-5 neighbors per row, zero else |
| **`DGSTA`** | 337 | **MAIN MODEL** (see §5.3) |

### 5.3 DGSTA (Main Model) — Detailed Forward Pass

**`__init__`**(Lines 338–446):

1. **Extracts from `data_feature`**: `scaler, num_nodes, feature_dim, ext_dim, dtw_matrix, adj_mx, sd_mx, sh_mx`

2. **Reads from `config`**: `embed_dim=64, skip_dim=256, lape_dim=8, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, enc_depth=6, drop_path=0.3, type_ln='pre', far_mask_delta=5, ...`

3. **Builds geo_mask** (Lines 394–405):
   - If `type_short_path == "dist"`: masks node pairs whose road distance > `far_mask_delta`
   - Else (default): masks node pairs whose hop count ≥ `far_mask_delta` (i.e., nodes > 5 hops away are masked from spatial attention)

4. **Loads tempp** (Lines 439–446):
   ```python
   tempp_path = f"../../cache/dataset_cache/{self.dataset}/tempp.npy"  # relative to __file__
   tempp = np.load(tempp_path)
   tempp = norm_embedding(tempp)          # keep top-5 per row
   self.tempp = cal_lape_emb(tempp)       # Laplacian eigenvectors
   ```

5. **Builds components**:
   - `enc_embed_layer`: `DataEmbedding`
   - `encoder_blocks`: 6 × `STEncoderBlock` with linearly increasing `drop_path` (0 → 0.3)
   - `skip_convs`: 6 × `Conv2d(embed_dim=64 → skip_dim=256, 1×1)` for skip connections
   - `end_conv1`: `Conv2d(input_window=12 → output_window=12, 1×1)` — temporal transform
   - `end_conv2`: `Conv2d(skip_dim=256 → output_dim=1, 1×1)` — output projection

**`cal_lape_emb(adj)`** (Lines 449–463):
1. Converts adj to sparse COO
2. Computes normalized symmetric Laplacian: `L = I - D^{-1/2} A D^{-1/2}`
3. Counts isolated (zero-degree) nodes
4. Eigendecomposition of L
5. Returns first `lape_dim=8` non-trivial eigenvectors (sorted by eigenvalue, skipping isolated-node dims)

**`forward(self, batch, lap_mx=None)`** (Lines 465–478):
```
x, ind = batch                  # x: [B, T_in=12, N, F]
                                # ind: [B] time-of-day indices

enc = enc_embed_layer(x, lap_mx, self.tempp)
    = value_embedding(x)
    + positional_encoding           # sinusoidal, [1, T, 1, D]
    + daytime_embedding             # time-of-day ∈ [0, 1440)
    + weekday_embedding             # day-of-week ∈ {0..6}
    + spatial_embedding(lap_mx)     # Laplacian eigenvectors projected to D
    + tempp_embedding(self.tempp)   # temporal prior projected to D
    + dropout

skip = 0
for i, block in enumerate(encoder_blocks):
    enc = block(enc, ind, geo_mask)          # STEncoderBlock
    skip += skip_convs[i](enc.permute(0,3,2,1))  # accumulate skip: [B, skip_dim, N, T]
    # permute: [B,T,N,D] → [B,D,N,T] so Conv2d works on (N,T) spatial grid

skip = skip.permute(0,3,2,1)          # → [B, T, N, skip_dim]
skip = ReLU(end_conv1(skip))          # → [B, T_out=12, N, skip_dim]
skip = ReLU(end_conv2(skip))          # → [B, T_out=12, N, 1]
return skip
```

### 5.4 STSelfAttention — The Core Attention Module

**Head allocation** (total dim = embed_dim = 64, total heads = 4+2+2 = 8, head_dim = 8):

| Attention type | Heads | Dim fraction | What it does |
|---|---|---|---|
| Geographic (spatial) | 4 | 50% | Per-timestep node-to-node attention |
| Semantic | 2 | 25% | *Declared but NOT separately computed* |
| Temporal | 2 | 25% | Per-node time-to-time attention |

**Forward pass inside STSelfAttention.forward(x, ind, geo_mask)**:

```
1. DYNAMIC GRAPH CONVOLUTION (lines 233–247):
   time_emb = nodevec_p1[ind % 288]             # [B, 40] time-of-day embedding
   adp = dgconstruct(time_emb, nodevec_p2, nodevec_p3, nodevec_pk)
       = ReLU(t @ Core @ src @ tgt^T)            # Tucker-style tensor decomposition
       = softmax(adp, dim=-1)                    # row-normalized → [B, N, N]
   x_gcn = reshape1(x).permute(0,3,2,1)          # [B, 32, N, T]
   x_gcn = gconv(x_gcn, [adp])                   # 2-hop GCN with dynamic adj
   x_gcn = reshape2(x_gcn.permute(...))           # back to [B, T, N, D]

2. TEMPORAL SELF-ATTENTION (lines 250–258):
   Q_t, K_t, V_t = 1×1 Conv2d on temporal_portion of channels
   Reshape to [B*N, t_heads=2, T, head_dim=8]
   attn = softmax(Q_t @ K_t^T / sqrt(8))
   out_t = attn @ V_t                            # [B*N, 2, T, 8]
   out_t = reshape → [B, T, N, int(D*0.25)]

3. GEOGRAPHIC SELF-ATTENTION (lines 260–271):
   Q_g, K_g, V_g = 1×1 Conv2d on geo_portion of channels
   Reshape to [B*T, geo_heads=4, N, head_dim=8]
   If geo_mask: attn = softmax(Q @ K^T / sqrt(8) + geo_mask)  # mask far nodes
   Else:        attn = softmax(Q @ K^T / sqrt(8))
   out_g = attn @ V_g                            # [B*T, 4, N, 8]
   out_g = reshape → [B, T, N, int(D*0.50)]

4. FUSION:
   out = concat([out_t, out_g], dim=-1)          # [B, T, N, int(D*0.75)]
   out = proj(out)                                # Linear → [B, T, N, D]
   out = dropout(out)
   return out
```

**Important architectural note**: The semantic heads (`sem_num_heads=2`, 25% of channels) have `Q_s`, `K_s`, `V_s` projection layers defined but are **never called** in `forward()`. Only temporal (25%) and geographic (50%) projections are used. The remaining 25% of the dimension space passes through untouched before the final `proj` layer. This is either a deliberate simplification or a porting discrepancy from the original paper.

### 5.5 Dynamic Graph Construction (`dgconstruct`)

A Tucker-decomposition-style dynamic adjacency matrix:

```python
# nodevec_p1: [288, 40]  — 288 time-of-day slots (5-min intervals in 24h)
# nodevec_p2: [N, 40]    — source node embeddings
# nodevec_p3: [N, 40]    — target node embeddings
# nodevec_pk: [40, 40, 40] — core tensor

adp = time_emb @ core.reshape(40, 40*40)     # [B, 1600]
adp = adp.reshape(B, 40, 40)                 # [B, 40, 40]
adp = src_emb @ adp                           # [B, N, 40]  (broadcast src over batch)
adp = adp @ tgt_emb.T                         # [B, N, N]
adp = softmax(ReLU(adp), dim=-1)             # row-normalized
```

So the adjacency `adp` is a **time-of-day-dependent, learned** matrix that changes every 5-minute interval. This is the core "dynamic graph" mechanism.

### 5.6 Curriculum Learning

```python
# In calculate_loss_without_predict:
task_level = min(task_level, output_window)   # starts at 0, max 12
if training:
    y_true = y_true[:, :task_level, :, :]     # only first task_level steps
    y_pred = y_pred[:, :task_level, :, :]
    if batches_seen % step_size == 0:
        task_level += 1                        # increase difficulty
```

The model first learns to predict 1 step, then 2, then 3... up to all 12. `step_size` controls how many batches per difficulty level (PeMS08: 2776 steps).

### 5.7 Loss Functions (from `libcity/model/loss.py`)

| Loss | Formula / Notes |
|---|---|
| `masked_mae` | Mean absolute error, zero-valued ground-truth positions masked |
| `masked_mse` | Mean squared error, masked |
| `masked_rmse` | Root MSE, masked |
| `masked_mape` | Mean absolute percentage error, masked |
| `masked_huber` | Huber loss, masked |
| `huber` | Standard Huber (delta from config) |
| `log_cosh` | Log-cosh loss |
| `quantile` | Quantile loss (delta=0.25) |
| `mae`, `mse`, `rmse` | Unmasked variants |
| `r2_score` | R² coefficient of determination |
| `explained_variance` | Explained variance score |

Active loss for PeMS08 baseline: `huber` with `huber_delta=2` (set in `PeMS08.json`).

---

## 6. Config System — Key Parameters & Their Defaults

### 6.1 Model Architecture (`DGSTA.json` + `PeMS08.json` override)

| Param | Default | PeMS08 | Meaning |
|---|---|---|---|
| `embed_dim` | 64 | 64 | Hidden dimension throughout the model |
| `skip_dim` | 256 | 256 | Skip connection dimension (4× embed_dim) |
| `lape_dim` | 8 | 8 | Number of Laplacian eigenvectors |
| `geo_num_heads` | 4 | 4 | Geographic attention heads |
| `sem_num_heads` | 2 | 2 | Semantic attention heads (declared, not used) |
| `t_num_heads` | 2 | 2 | Temporal attention heads |
| `mlp_ratio` | 4 | 4 | MLP hidden = embed_dim × mlp_ratio |
| `enc_depth` | 6 | 6 | Number of ST encoder blocks |
| `drop` | 0 | 0 | General dropout |
| `attn_drop` | 0 | 0 | Attention dropout |
| `drop_path` | 0.3 | 0.3 | Max stochastic depth rate |
| `type_ln` | "post" | "pre" | LayerNorm position: before ("pre") or after ("post") attention/MLP |
| `type_short_path` | "hop" | "hop" | Geo mask basis: "hop" or "dist" |
| `far_mask_delta` | 5 | 7 | Nodes farther than this get masked (hops if hop, distance if dist) |
| `bidir` | false | true | Whether adjacency is treated as bidirectional in Dataset |

### 6.2 Training (`DGSTA.json` + `PeMS08.json` override)

| Param | Default | PeMS08 | Meaning |
|---|---|---|---|
| `batch_size` | 32 | 32 | Batch size |
| `max_epoch` | 300 | 300 | Training epochs |
| `learner` | "adamw" | "adamw" | Optimizer |
| `learning_rate` | 1e-3 | 1e-3 | Initial LR |
| `weight_decay` | 0.05 | 0.05 | AdamW weight decay |
| `lr_scheduler` | "cosinelr" | "cosinelr" | LR schedule |
| `lr_eta_min` | 1e-4 | 1e-4 | Minimum LR |
| `lr_warmup_epoch` | 5 | 5 | Linear warmup epochs |
| `lr_warmup_init` | 1e-6 | 1e-6 | Initial warmup LR |
| `clip_grad_norm` | true | true | Gradient clipping |
| `max_grad_norm` | 5 | 5 | Max gradient norm |
| `use_early_stop` | true | true | Early stopping |
| `patience` | 50 | 50 | Early stop patience (epochs) |
| `use_curriculum_learning` | true | true | Curriculum learning |
| `step_size` | 1562 | 2776 | Batches per curriculum difficulty increase |
| `set_loss` | "masked_mae" | "huber" | Loss function |
| `huber_delta` | 1 | 2 | Huber loss delta |
| `scaler` | "standard" | "standard" | Data normalization |
| `seed` | 0 | 1 | Random seed |

### 6.3 Data (`DGSTADataset.json` + `PeMS08.json` override)

| Param | Default | PeMS08 | Meaning |
|---|---|---|---|
| `train_rate` | 0.7 | 0.6 | Fraction for training |
| `eval_rate` | 0.1 | 0.2 | Fraction for validation |
| `input_window` | 12 | 12 | Input time steps |
| `output_window` | 12 | 12 | Output time steps |
| `add_time_in_day` | false | true | Add time-of-day feature |
| `add_day_in_week` | false | true | Add day-of-week feature |
| `cache_dataset` | true | true | Cache preprocessed data |
| `num_workers` | 0 | 0 | DataLoader workers |
| `cluster_method` | — | "kshape" | Time series clustering method |
| `cand_key_days` | — | 21 | Candidate days for pattern key selection |

---

## 7. Dataset Quick Reference

| Dataset | Sensors | Dyna Size | Unique features |
|---|---|---|---|
| **PeMS03** | 227 | 404 MB | epsilon=0.1 |
| **PeMS04** | 307 | 286 MB | epsilon=0 (unique), far_mask=3 |
| **PeMS07** | 883 | 1.1 GB | flow-only (no speed/occupancy), batch_size=8, grad_accmu=2 |
| **PeMS08** | 170 | 166 MB | **Primary dataset**, cand_key_days=21 |
| **PEMS-BAY** | 325 | 727 MB | Largest .rel (53KB) |

METR-LA is listed in `task_config.json` as allowed but has **no raw_data directory, no top-level config, and no cache** in the current workspace.

---

## 8. Cache & Tempp

### 8.1 What is `tempp.npy`?

A temporal prior matrix computed during `DGSTADataset.__init__()`. It represents a pairwise relationship between sensors based on their temporal patterns. In the model:
1. Loaded as `[N, N]` matrix
2. `norm_embedding`: zeros diagonal, keeps only top-5 values per row → sparse binary-ish matrix
3. `cal_lape_emb`: computes Laplacian eigenvectors of this sparsified matrix
4. Used in `DataEmbedding` as the `tempp_embedding` (temporal prior positional encoding)

### 8.2 DTW & Pattern Keys

- **DTW matrix**: Pairwise Dynamic Time Warping distance between all sensor time series. Used by the Dataset but not directly by the DGSTA forward pass.
- **Pattern keys**: KShape clustering of daily patterns → one cluster ID per (sensor, day) pair. Used during data loading for curriculum/sampling strategies.

---

## 9. Checklist for Running PeMS08 from Scratch

**Prerequisites already met in current workspace:**
- Conda env `ai_lab` with PyTorch 2.9.1 + CUDA 12
- Raw data in `raw_data/PeMS08/`
- Preprocessed cache in `libcity/cache/dataset_cache/`
- Model checkpoints in `libcity/cache/99049/model_cache/`

**Command:**
```bash
conda activate ai_lab
python run_model.py --gpu_id 0
```

**Why `--gpu_id 0` is required:** `DGSTAExecutor.json` defaults `gpu_id` to `[0, 3]`, but the machine has only GPU 0. Passing `--gpu_id 0` overrides this.

**Expected output:**
- `libcity/log/` gets a new log file
- `libcity/cache/{exp_id}/` gets model checkpoints and TensorBoard events
- Console prints per-epoch train loss, eval loss, and metrics (MAE, MAPE, RMSE)

---

## 10. Known Issues & Gotchas

1. **Hardcoded `gpu_id: [0, 3]`** in `DGSTAExecutor.json` → must override via `--gpu_id 0` on single-GPU machines.

2. **`import torch.nn.init`** in DGSTA.py is unused (dead import).

3. **`from torch.nn.functional import cosine_similarity`** is unused (dead import).

4. **`Chomp2d`** class defined but never used in forward pass.

5. **Semantic attention heads** have projection layers defined but are never called — only temporal and geographic attention are computed.

6. **`lap_mx` parameter** passed to `DGSTA.forward()` and `predict()` but not used — the Laplacian is recomputed inside `DataEmbedding`.

7. **Config merging can be confusing**: `PeMS08.json` is loaded as `config_file`, NOT `raw_data/PeMS08/config.json`. The latter is a data schema file loaded by the dataset, not the model config system.

8. **PeMS07** has only `traffic_flow` in its `.dyna` (no occupancy/speed), which is unique among all datasets.

---

## 11. Ablation & Experimentation

The project is set up for **config-switch-based ablation**. Adding a new module follows this pattern:
1. Add a boolean config key (e.g., `use_my_module: true`)
2. In `DGSTA.__init__`, read the key and conditionally build the module
3. In `DGSTA.forward`, conditionally apply the module
4. In `PeMS08.json`, set the key to enable/disable

This preserves the original DGSTA as the default (all switches off = original behavior).

Experiment logs are tracked in `experiments/PeMS08_ablation_log.md`.

---

## 12. File Sizes & Line Counts (for context window estimation)

| File | Lines |
|---|---|
| `DGSTA.py` (core model) | 536 |
| `dgsta_dataset.py` | ~300 |
| `dgsta_executor.py` | ~150 |
| `pipeline.py` | 219 |
| `traffic_state_executor.py` | ~400 |
| `traffic_state_point_dataset.py` | ~300 |

## 13. Local Compatibility Patches

This workspace is based on the official DGSTA code, but the original code assumed an older multi-GPU server environment. The following patches were applied only for local compatibility, not for model innovation:

1. Ray Tune import paths updated from `ray.tune.suggest.*` to `ray.tune.search.*`.
2. `tempp.npy` path changed from absolute `/libcity/...` to project-relative path.
3. Pipeline model placement changed from hard-coded `cuda:2` / `device_ids=[2,3]` to `config.get('device')`.
4. Executors made compatible with both `DataParallel(model)` and ordinary single-GPU `model`.

These changes do not intentionally alter DGSTA's model architecture.

## 14. Clean Baseline Reproduction on PeMS08

- Branch: `claude-dgsta-improvement`
- Exp ID: `99049`
- Command:
  `CUDA_VISIBLE_DEVICES=0 python run_model.py --task traffic_state_pred --model DGSTA --dataset PeMS08 --config_file PeMS08 --gpu_id 0`
- Best epoch loaded: 294
- @3: MAE 12.082, RMSE 20.654, masked_MAPE 7.899
- @6: MAE 12.498, RMSE 21.690, masked_MAPE 8.175
- @12: MAE 13.204, RMSE 23.173, masked_MAPE 8.701
- Paper PeMS08 DGSTA:
  - @3: MAE 12.00, RMSE 20.41, MAPE 7.84
  - @6: MAE 12.46, RMSE 21.47, MAPE 8.13
  - @12: MAE 13.18, RMSE 22.94, MAPE 8.63
- Conclusion: close reproduction, slightly worse than paper, acceptable as local clean baseline.

When providing context to an LLM, the critical files are: `DGSTA.py` (model), `dgsta_dataset.py` (data), `pipeline.py` (entry), and `PeMS08.json` (config).
