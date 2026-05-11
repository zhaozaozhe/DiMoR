# DGSTA 项目 — 新机器部署与 LLM 快速上手指南

本文档供新机器上的 Claude Code 或任何 LLM 快速理解项目全貌和职责。

---

## 1. 项目身份

- **项目名**: DiMoR — **Di**screte **Mo**dal **R**outing Network (原 DGSTA + VQ Router 增强)
- **任务**: PeMS08 交通流量预测 (input 12 步 → output 12 步, 5min 粒度)
- **环境**: WSL Ubuntu + Conda env `ai_lab`, Python 3.10, PyTorch 2.9.1, CUDA 12
- **主模型文件**: `libcity/model/traffic_flow_prediction/DGSTA.py` (~800 行)
- **入口**: `python run_model.py --gpu_id 0`
- **配置**: `PeMS08.json` (最高优先级)
- **GitHub**: `https://github.com/zhaozaozhe/DiMoR` (分支: `main`)
- **冻结 commit**: `c0ad640` (所有多 seed 实验基于此)
- **新机器首次**:
  ```bash
  git clone https://github.com/zhaozaozhe/DiMoR.git
  cd DiMoR
  conda env create -f ai_lab_env.yml
  ```
  或使用 U盘压缩包 `DiMoR_portable.tar.gz` 解压后 `git init` + `git remote add origin https://github.com/zhaozaozhe/DiMoR.git`

---

## 2. 新机器初始化（一次性）

```bash
# 1. 解压项目
tar -xzf DGSTA_portable.tar.gz
cd DGSTA_clean

# 2. 创建 conda 环境
conda env create -f ai_lab_env.yml
conda activate ai_lab

# 3. 验证
python -m py_compile libcity/model/traffic_flow_prediction/DGSTA.py
python run_model.py --gpu_id 0
# Ctrl+C 停止，确认无报错即可
```

### 如果缺少 GPU / CUDA
```bash
# PyTorch CPU 版也可以跑（只是慢）
conda install pytorch cpuonly -c pytorch
```

---

## 3. 项目结构（关键文件）

```
DGSTA_clean/
├── PeMS08.json                          # 实验配置（改这个就行）
├── run_model.py                         # 训练入口
├── DGSTA_EVOLUTION.md                   # ★ 项目演进文档（LLM 必读）
├── CLAUDE.md                            # 项目规则
├── PROJECT_OVERVIEW.md                  # 架构概览
│
├── libcity/
│   ├── model/traffic_flow_prediction/DGSTA.py   # ★ 模型代码（唯一需要改的文件）
│   ├── executor/dgsta_executor.py               # 训练循环
│   ├── data/dataset/dgsta_dataset.py            # 数据预处理
│   └── config/                                   # JSON 默认配置
│
├── raw_data/PeMS08/                      # 原始数据 (.dyna/.geo/.rel)
├── libcity/cache/dataset_cache/          # 预处理缓存 (.npz/.npy)
|                                          #   - PeMS08/tempp.npy
|                                          #   - dtw_PeMS08.npy
|                                          #   - pattern_keys_kshape_*.npy
│
├── analysis/vq_router/                   # VQ Router 机制分析脚本
│   ├── analyze_vq_router.py              # 时间条件路由分析
│   ├── frozen_template_replay.py         # 冻结重放实验
│   ├── time_conditioned_routing.txt      # 分析报告
│   └── vq_analysis_report.txt            # 早期报告
│
├── experiments/
│   ├── configs/                          # 各 seed 的配置快照
│   └── ablation_plan.md                  # 消融实验计划
│
└── libcity/cache/<exp_id>/               # 训练结果（每次实验一个目录）
    ├── model_cache/DGSTA_PeMS08.m        # 最佳模型 checkpoint
    └── evaluate_cache/*.csv              # MAE/RMSE 结果
```

---

## 4. 模型架构（当前最佳配置）

### 激活模块

```json
{
    "use_vq_router": true,       // VQ Router: 10个图模板 + Gumbel-Softmax路由
    "use_deep_trend": true,      // DeepTrendNet: 趋势分解 + MLP预测
    "use_delay_conv": true,      // DelayConv: 因果时域卷积 in GCN
    "use_time_aware_adp": false, // adj_adp时变 (实验无效, 已关闭)
    "use_balance_loss": false,   // VQ balance loss (导致退化, 已关闭)
    "far_mask_delta": 7,
    "seed": 2                    // ← 当前正在跑 seed=2
}
```

### VQ Router 转发流程

```
x → SeriesDecomp → x_res, x_trend
x_trend → VQ Router → Gumbel-Softmax(hard=True) → adj_vq[B,T,N,N]
                    → nodevec1 @ nodevec2 → adj_adp[N,N]
sparsify_graph → Top-20 + re-softmax
GCN(x_res, [adj_vq, adj_adp]) → tanh
x_trend + GCN_out * 0.1 → LayerNorm
→ Temporal Attn → Geo Attn → proj(48→64)
(注意: VQ 模式下 Semantic Attention 被静默关闭)
```

### DeepTrendNet 独立分支

```
x_flow = x[..., :feat_dim]
SeriesDecomp(x_flow) → x_trend
DeepTrendNet(x_trend): MLP(12→64→64→12, mlp3零初始化) → trend_pred
output = main_pred + trend_fusion(1.0) × trend_pred
```

---

## 5. 实验协议（严格遵守）

### 冻结环境
- Commit: `c0ad640`
- PyTorch: 2.9.1+cu128
- GPU: RTX 5070 Ti
- 配置快照: `experiments/configs/full_seed*.json`

### 运行实验
```bash
conda activate ai_lab
python run_model.py --gpu_id 0
# 结果在 libcity/cache/<新exp_id>/evaluate_cache/*.csv
```

### 读取最新结果
```bash
ls -t libcity/cache/[0-9]*/evaluate_cache/*.csv | head -1 | xargs cat
```

### 配置切换
所有实验只需改 `PeMS08.json` 中的布尔开关，不需要改 Python 代码。

---

## 6. 当前论文级实验矩阵

| 状态 | 实验 | Seed | Exp ID | @3 | @6 | @12 |
|---|---|---|---|---|---|---|
| ✅ | Full | 1 | 71098 | 11.894 | 12.406 | 13.228 |
| ✅ | Full | 0 | 61239 | 12.117 | 12.570 | 13.306 |
| 🔄 | Full | 2 | 进行中 | — | — | — |
| ⬜ | -VQ | 0 | 待跑 | — | — | — |
| ⬜ | -VQ | 2 | 待跑 | — | — | — |
| ✅ | -VQ | 1 | 68783 | 12.176 | 12.645 | 13.493 |
| ✅ | -Trend | 1 | 64832 | 11.912 | 12.421 | 13.222 |
| ✅ | -Delay | 1 | 43876 | 12.176 | 12.622 | 13.348 |

---

## 7. CLAUDE.md 项目规则（必须遵守）

1. 不重写整个项目
2. 不修改 data loading / executor / evaluator / trainer / scaler
3. 不碰 raw_data / dataset_cache
4. 所有新模块必须 config-gated，默认 false
5. 默认行为必须完全等价于 baseline
6. 改动前先解释意图，改完后总结 + 验证编译
7. 不声称未经实验日志验证的性能提升
8. 优先小改动、可逆改动、消融友好改动

---

## 8. 当前阶段任务

**Phase: 论文收束（不再添加新模块）**

| 优先级 | 任务 | 状态 |
|---|---|---|
| P1 | Full(seed=2) 跑完 | 🔄 |
| P1 | -VQ(seed=0), -VQ(seed=2) | ⬜ |
| P2 | 多 seed mean±std 表格 | ⬜ |
| P2 | Router 行为可视化 | ⬜ |
| P3 | 论文撰写 | ⬜ |

**禁止**: 新增 attention / loss / graph branch / trend 模块 / 任何未在 DGSTA_EVOLUTION.md Part 5 中列举为"已知盲点"的结构改动。
