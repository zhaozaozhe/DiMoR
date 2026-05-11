# 新机器部署指南

本文档供在新机器上首次设置 Claude Code 会话时读取。

---

## 1. 需要拷贝的内容（U盘）

项目总大小 28GB，**只需拷贝约 3GB**：

### 必须拷贝（通过 U 盘）

```
整个项目目录，但删除以下两个文件夹：
  rm -rf libcity/cache/    # 26GB 训练缓存，不需拷贝，运行时会自动重新生成
  rm -rf libcity/log/      # 16MB 历史日志，不需拷贝
```

拷贝后项目约 3GB（代码 + raw_data + .git）。

### 不需要拷贝

- `libcity/cache/` — 模型 checkpoint 和数据集缓存，新机器上运行时会重新生成
- `libcity/log/` — 训练日志，新机器上自动生成
- `__pycache__/` — Python 字节码缓存
- `.vscode/` — IDE 配置

---

## 2. 新机器环境安装

### 2.1 安装 Miniconda

```bash
# 下载安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 按提示操作：yes → 确认路径 → yes (init)
# 重启终端或 source ~/.bashrc
```

### 2.2 创建环境

```bash
cd DGSTA_clean
conda env create -f env_ai_lab_before_codex.yml
conda activate ai_lab
```

### 2.3 验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m py_compile libcity/model/traffic_flow_prediction/DGSTA.py
python run_model.py --gpu_id 0
```

---

## 3. Git 工作流（两台机器协作）

### 3.1 上传项目到 GitHub（一次性）

```bash
# 在 GitHub 网页上创建新的 private repository，例如: DGSTA_experiments
git remote add origin https://github.com/你的用户名/DGSTA_experiments.git
git push -u origin exp/trend-only
```

### 3.2 日常协作流程

```
机器 A (当前)                      GitHub                        机器 B (新)
─────────                          ──────                        ─────────
git commit -m "..." ──────────→   云端保存   ←────────── git pull
git push                                                      改 PeMS08.json
                                                              python run_model.py
                                                              跑完了
                                                              git add 结果文档
                                                              git commit -m "..."
                                                              git push
git pull  ←─────────────────────────────────────────────────  更新到本地
```

**每次在新机器上开始工作前**：
```bash
git pull                    # 拉取最新代码
conda activate ai_lab       # 激活环境
```

**每次跑完实验后**：
```bash
git add DGSTA_EVOLUTION.md experiments/   # 添加结果
git commit -m "exp: Full seed=2 结果"      # 提交
git push                                  # 推送到 GitHub
```

---

## 4. Claude Code 在新机器上的首次对话

在新机器上打开 Claude Code 后，直接说：

> 请先阅读 CLAUDE.md、DGSTA_EVOLUTION.md、SETUP_NEW_MACHINE.md。
> 然后告诉我这个项目是什么，我接下来应该做什么。

Claude Code 会自行理解：
- 项目是什么（CLAUDE.md）
- 之前的完整实验记录（DGSTA_EVOLUTION.md）
- 当前配置状态（PeMS08.json）
- 接下来该跑哪个实验（看 experiment plan）

---

## 5. 两台机器的分工建议

| 机器 | 任务 |
|---|---|
| 当前机器 | Full(seed=2) — 已在跑 |
| 新机器 | 准备好后跑 -VQ(seed=0) 或论文可视化 |

---

## 6. 当前实验排队

按优先级：

1. ~~Full(seed=0)~~ → exp 61239, @3=12.117
2. ~~Full(seed=1)~~ → exp 71098, @3=11.894
3. Full(seed=2) — 当前机器正在跑
4. -VQ(seed=0) — 新机器跑
5. -VQ(seed=2) — 排队

PeMS08.json 当前配置：`seed=2, use_vq_router=true, use_deep_trend=true, use_delay_conv=true`
