# 消融实验计划 —— 论文级数据

## 背景

全模块最佳性能: @3=11.894  @6=12.406  @12=13.228 (exp 71098)

## 实验矩阵

| # | 实验名 | VQ | Trend | Delay | 证明什么 |
|---|---|---|---|---|---|
| 1 | 全模块 (已跑) | on | on | on | 最优性能 |
| 2 | -VQ Router | off | on | on | VQ Router 独立贡献 |
| 3 | -DeepTrendNet | on | off | on | DeepTrendNet 独立贡献 |
| 4 | -DelayConv | on | on | off | DelayConv 独立贡献 |
| 5 | 纯基线 | off | off | off | 原始 DGSTA (已有) |

## 结果模板

| Model | @3 MAE | @6 MAE | @12 MAE | @3 RMSE | @6 RMSE | @12 RMSE | Params |
|---|---|---|---|---|---|---|---|
| DGSTA (baseline) | 12.082 | 12.498 | 13.204 | | | | |
| +VQ Router | | | | | | | |
| +DeepTrendNet | | | | | | | |
| +DelayConv | | | | | | | |
| Full (all three) | 11.894 | 12.406 | 13.228 | | | | |

### 运行方法

每轮只改 PeMS08.json 三个布尔值，然后:
```bash
python run_model.py --gpu_id 0
```

结果从 `libcity/cache/<exp_id>/evaluate_cache/*.csv` 获取。
