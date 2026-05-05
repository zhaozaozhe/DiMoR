# CLAUDE.md

## Project
This is a clean DGSTA/LibCity traffic flow prediction project.

## Environment
- OS: WSL Ubuntu-24.04
- Conda env: ai_lab
- Main dataset: PeMS08
- Main model file: libcity/model/traffic_flow_prediction/DGSTA.py
- Entry point: run_model.py

## Hard rules
- Do not rewrite the whole project.
- Do not modify data loading, evaluator, executor, scaler, trainer, or raw dataset files unless explicitly requested.
- Do not touch raw_data or libcity/cache/dataset_cache unless explicitly requested.
- Keep the original DGSTA baseline behavior available.
- Prefer small, reversible, ablation-friendly changes.
- Add config switches for every new module when possible.
- Before changing code, explain the intended patch.
- After changing code, summarize modified files and show how to test.
- Never claim performance improvement without experiment logs.

## Research direction
The goal is not blind accuracy chasing.
Prefer:
1. residual enhancement of the original DGSTA dynamic graph,
2. soft routing rather than hard replacement,
3. lightweight and interpretable changes,
4. clear ablation experiments on PeMS08,
5. default behavior should remain original DGSTA unless a config switch enables the new module.

## Validation
At minimum, run:
python -m py_compile libcity/model/traffic_flow_prediction/DGSTA.py
python -m py_compile run_model.py
