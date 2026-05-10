"""
VQ Router mechanism analysis — time-conditioned routing.
Key question: Does routing change with time of day, or is it static per layer?

Usage:
    conda run -n ai_lab python analysis/vq_router/analyze_vq_router.py --exp_id 71098
"""
import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

project_root = '/home/user/DeepLearning/DGSTA_Codex_Workspace/DGSTA_clean'
sys.path.insert(0, project_root)
os.chdir(project_root)

def load_best_model(exp_id='71098'):
    from libcity.config import ConfigParser
    from libcity.data import get_dataset
    from libcity.model.traffic_flow_prediction.DGSTA import DGSTA

    config = ConfigParser('traffic_state_pred', 'DGSTA', 'PeMS08', 'PeMS08', {'gpu_id': [0]})
    dataset = get_dataset(config)
    train_dl, eval_dl, test_dl = dataset.get_data()

    cache_dir = f'libcity/cache/{exp_id}/model_cache'
    m_files = [f for f in os.listdir(cache_dir) if f.endswith('.m')]
    if m_files:
        checkpoint = torch.load(os.path.join(cache_dir, m_files[0]), map_location='cpu')
    else:
        raise FileNotFoundError(f'No .m file in {cache_dir}')

    data_feature = dataset.get_data_feature()
    model = DGSTA(config, data_feature)
    if isinstance(checkpoint, tuple):
        model.load_state_dict(checkpoint[0], strict=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # lap_mx
    adj_mx = model.get_data_feature().get('adj_mx')
    import scipy.sparse as sp
    adj = sp.coo_matrix(adj_mx)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    L = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    lap_mx = torch.from_numpy(EigVec[:, 1:9]).float().to(device)

    return model, dataset.test_dataloader, lap_mx, device


def collect_time_conditioned(model, loader, lap_mx, device):
    """Collect routing weights with time-of-day info."""

    # Storage: layer_idx → list of (template_id, hour_group)
    records = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            batch.to_tensor(device)
            _ = model(batch, lap_mx=lap_mx)

            ind = batch['ind'].cpu().numpy()  # [B] time index

            # ind%288 gives the 5-min slot of the FIRST input step
            # For each time step t in [0..11], the predicted slot is (ind%288 + t) % 288
            for layer_idx, block in enumerate(model.encoder_blocks):
                if hasattr(block.st_attn, 'vq_router'):
                    weights = block.st_attn.last_router_weights.cpu().numpy()  # [B, T, K]
                    hard_ids = weights.argmax(axis=-1)  # [B, T]
                    B, T = hard_ids.shape
                    for b in range(B):
                        for t in range(T):
                            time_slot = (ind[b] + t) % 288
                            hour = time_slot // 12  # 0..23
                            template_id = int(hard_ids[b, t])
                            records[layer_idx].append((hour, template_id))

    return records


def analyze_time_conditioned(records):
    """Analyze P(template | hour, layer)."""
    lines = []
    lines.append("=" * 70)
    lines.append("TIME-CONDITIONED ROUTING ANALYSIS")
    lines.append("Key question: Does routing change with time of day?")
    lines.append("=" * 70)

    # Hour groups
    groups = {
        'Morning(7-9)':  list(range(7, 10)),
        'Midday(10-16)': list(range(10, 17)),
        'Evening(17-19)': list(range(17, 20)),
        'Night(0-6)':    list(range(0, 7)),
        'Late(20-23)':   list(range(20, 24)),
    }

    for layer_idx in sorted(records.keys()):
        data = records[layer_idx]  # list of (hour, template_id)
        lines.append(f"\n--- Layer {layer_idx} ---")

        # Global distribution
        all_ids = [t for _, t in data]
        global_counts = np.bincount(all_ids, minlength=10)
        global_pct = global_counts / global_counts.sum()
        top3 = np.argsort(global_pct)[-3:][::-1]
        lines.append(f"  Global: top-3={top3}, pct={[f'{global_pct[i]:.3f}' for i in top3]}")

        # Per-period distribution
        lines.append(f"  Per-period P(template|hour_group):")
        for group_name, hours in groups.items():
            group_ids = [t for h, t in data if h in hours]
            if len(group_ids) < 10:
                continue
            counts = np.bincount(group_ids, minlength=10)
            probs = counts / counts.sum()
            top3_g = np.argsort(probs)[-3:][::-1]
            lines.append(
                f"    {group_name:<18s}: top-3={list(top3_g)}, "
                f"pct={[f'{probs[i]:.3f}' for i in top3_g]}"
            )

        # Is routing STATIC or DYNAMIC?
        # Compute per-period dominant template; check if it varies
        dominant_per_period = []
        for group_name, hours in groups.items():
            group_ids = [t for h, t in data if h in hours]
            if len(group_ids) < 10:
                continue
            counts = np.bincount(group_ids, minlength=10)
            dominant_per_period.append((group_name, counts.argmax(), counts.max() / counts.sum()))

        unique_dominants = set(d for _, d, _ in dominant_per_period)
        if len(unique_dominants) == 1:
            lines.append(f"  VERDICT: STATIC — dominant template = {dominant_per_period[0][1]} across all periods")
        else:
            lines.append(f"  VERDICT: DYNAMIC — dominant template varies across periods:")
            for name, dom, pct in dominant_per_period:
                lines.append(f"    {name}: T{dom} ({pct:.1%})")

    # Global summary
    lines.append("\n" + "=" * 70)
    lines.append("GLOBAL SUMMARY")
    static_layers = 0
    dynamic_layers = 0
    for layer_idx in sorted(records.keys()):
        data = records[layer_idx]
        dominant_set = set()
        for group_name, hours in groups.items():
            group_ids = [t for h, t in data if h in hours]
            if len(group_ids) < 10:
                continue
            counts = np.bincount(group_ids, minlength=10)
            dominant_set.add(counts.argmax())
        if len(dominant_set) == 1:
            static_layers += 1
            lines.append(f"  Layer {layer_idx}: STATIC — always T{dominant_set.pop()}")
        else:
            dynamic_layers += 1
            lines.append(f"  Layer {layer_idx}: DYNAMIC — {len(dominant_set)} different dominants across periods")
    lines.append(f"\n  Static layers: {static_layers}, Dynamic layers: {dynamic_layers}")
    lines.append("=" * 70)

    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default='71098')
    args = parser.parse_args()

    print(f"Loading best model from experiment {args.exp_id}...")
    model, loader, lap_mx, device = load_best_model(args.exp_id)

    print("Collecting time-conditioned routing data...")
    records = collect_time_conditioned(model, loader, lap_mx, device)

    print("Analyzing...")
    report = analyze_time_conditioned(records)
    print(report)

    out_path = 'analysis/vq_router/time_conditioned_routing.txt'
    with open(out_path, 'w') as f:
        f.write(report)
    print(f"\nSaved to {out_path}")
