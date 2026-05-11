# CLAUDE.md

## Project
DiMoR — Discrete Modal Routing for Explainable Traffic Forecasting.
Built on DGSTA/LibCity. Phase: paper wrap-up (no new modules).

## Environment
- OS: WSL Ubuntu-24.04
- Conda env: ai_lab
- Main dataset: PeMS08
- Main model file: libcity/model/traffic_flow_prediction/DGSTA.py
- Entry point: run_model.py
- GitHub: https://github.com/zhaozaozhe/DiMoR (branch: main)

## Paper Strategy (updated 2026-05-11)
- NOT SOTA-chasing. This is a mechanism-analysis paper.
- Core finding: VQ Router learns layer-specialized static graphs, not time-varying routing.
- Last structural change: VQ+SemanticAttention coexistence (fixes VQ branch's exclusion of sem attn).
- DeepTrendNet contributes near-zero; documented as a negative finding.
- Multi-seed exposed high variance; claims must be conservative.

## Hard rules
- Do not rewrite the whole project.
- Do not modify data loading, evaluator, executor, scaler, trainer, or raw dataset files.
- Do not touch raw_data or libcity/cache/dataset_cache.
- Keep the original DGSTA baseline behavior available via config switches.
- All new modules must be config-gated, default false.
- Before changing code, explain the intended patch.
- After changing code, summarize and run py_compile.
- Never claim performance improvement without experiment logs.
- **No new modules. Paper wrap-up phase. Only multi-seed experiments + visualization.**

## Experiment protocol
- All experiments via PeMS08.json only (no code changes per run).
- Each experiment: save config snapshot, record exp ID + results.
- Multi-seed: change seed in PeMS08.json, rerun. 3 seeds minimum per config.

## Validation
At minimum, run:
python -m py_compile libcity/model/traffic_flow_prediction/DGSTA.py
python -m py_compile run_model.py
