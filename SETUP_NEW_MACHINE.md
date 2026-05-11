# DiMoR 项目 — 新机器部署指南

本文档供新机器上的 Claude Code 快速理解项目全貌和自己的职责。

---

## 0. Claude Code: 你被启动后要做的第一件事

这是最高优先级任务。按顺序执行：

### 步骤 A: 确认环境
```bash
which conda || echo "conda not found — tell user to install Miniconda first"
nvidia-smi || echo "no GPU — will run on CPU (slow but works)"
```

### 步骤 B: 创建环境
```bash
conda env create -f ai_lab_env.yml
```
如果这步报错 "environment exists"，改成:
```bash
conda env update -f ai_lab_env.yml
```

### 步骤 C: 验证能跑
```bash
conda run -n ai_lab python -m py_compile libcity/model/traffic_flow_prediction/DGSTA.py
conda run -n ai_lab python run_model.py --gpu_id 0 &
sleep 30 && kill %1   # 跑 30 秒确认不报错就行
```

### 步骤 D: 确认就绪
告诉用户: "环境就绪，DGSTA.py 编译通过，训练命令可以正常运行。现在开始读项目文档。"

### 然后: 继续读下面的 Part 1-6, 理解项目全貌。

---

## 1. 项目身份

- **项目名**: DiMoR — **Di**screte **Mo**dal **R**outing for Explainable Traffic Forecasting
- **任务**: PeMS08 交通流量预测 (input 12 步 → output 12 步, 5min 粒度)
- **论文定位**: 机制分析型（不是 SOTA 竞赛型）
  - 核心贡献：系统诊断 VQ 图路由在时空网络中的真实行为
  - 关键发现：layer specialization > temporal routing, DeepTrendNet 冗余, semantic attention 被 VQ 分支错误排除
- **GitHub**: `https://github.com/zhaozaozhe/DiMoR` (分支: `main`)
- **环境**: Conda env `ai_lab`, Python 3.10, PyTorch 2.9.1, CUDA 12
- **主模型文件**: `libcity/model/traffic_flow_prediction/DGSTA.py` (~790 行)
- **入口**: `python run_model.py --gpu_id 0`
- **配置**: `PeMS08.json` (最高优先级，改它就够)

---

## 2. 你的职责（Claude Code）

按优先级排列：

### P0: 理解项目
1. 读 `DGSTA_EVOLUTION.md` 了解完整演进历史
2. 读 `experiments/status_briefing_for_review.md` 了解当前困境和论文定位
3. 告诉用户你理解了哪些关键信息

### P1: 部署验证
```bash
conda env create -f ai_lab_env.yml
conda activate ai_lab
python -m py_compile libcity/model/traffic_flow_prediction/DGSTA.py
python run_model.py --gpu_id 0   # Ctrl+C 确认不报错即可
```

### P2: 补全实验
当前正在进行 VQ+Sem 共存实验（最后一次结构改动）。需要补齐：

| # | 实验 | seed | 优先级 | 说明 |
|---|---|---|---|---|
| 1 | VQ+Sem(seed=1) | 1 | 最高 | 本机已改好代码，直接跑 |
| 2 | VQ+Sem(seed=0) | 0 | 高 | 改 PeMS08.json `seed:0` |
| 3 | VQ+Sem(seed=2) | 2 | 高 | 改 PeMS08.json `seed:2` |

### P3: 产出论文素材
- router behavior 可视化（heatmap, layer specialization 图）
- 多 seed mean±std 表格
- horizon trade-off 分析

---

## 3. 项目结构（关键文件）

```
DiMoR/
├── PeMS08.json                          # 实验配置（改这个）
├── run_model.py                         # 训练入口
├── DGSTA_EVOLUTION.md                   # ★ 项目演进文档（必读）
├── CLAUDE.md                            # 项目规则
├── experiments/
│   └── status_briefing_for_review.md    # ★ 当前状态简报（必读）
├── libcity/
│   ├── model/traffic_flow_prediction/DGSTA.py   # ★ 模型代码
│   ├── executor/dgsta_executor.py               # 训练循环
│   ├── data/dataset/dgsta_dataset.py            # 数据预处理
│   └── config/                                   # JSON 默认配置
├── raw_data/PeMS08/                      # 原始数据
├── libcity/cache/dataset_cache/          # 预处理缓存
├── analysis/vq_router/                   # VQ Router 分析脚本
└── ai_lab_env.yml                        # conda 环境文件
```

---

## 4. 当前模型架构

### 激活模块
```json
{
    "use_vq_router": true,       // VQ Router: 10个图模板 + Gumbel-Softmax路由
    "use_deep_trend": true,      // DeepTrendNet: 趋势分解 + MLP预测
    "use_delay_conv": true,      // DelayConv: 因果时域卷积
    "far_mask_delta": 7,
    "seed": 1
}
```

### 最新改动：VQ + Semantic Attention 共存
原来 VQ Router 模式会静默关闭 Semantic Attention，proj 只有 48 维。现已修复为两者共存，proj 始终 64 维。这是最后一次结构改动，机制驱动而非盲目堆叠。

### 消融方法
所有实验只需改 `PeMS08.json` 中的布尔开关，不改 Python 代码。

---

## 5. 实验协议

- 冻结代码：当前 git commit (`git log -1`)
- 每次改 seed 时保存配置快照到 `experiments/configs/`
- 结果在 `libcity/cache/<exp_id>/evaluate_cache/*.csv`
- 查最新: `ls -t libcity/cache/[0-9]*/evaluate_cache/*.csv | head -1 | xargs cat`

---

## 6. 规则（来自 CLAUDE.md）

1. 不重写整个项目
2. 不修改 data loading / executor / evaluator / trainer / scaler
3. 所有新模块必须 config-gated，默认 false
4. 改动前先解释意图，改完后总结 + 验证编译
5. 不声称未经实验验证的性能提升
6. 当前阶段：不再添加新模块。论文收束模式。
