"""
Frozen-template static replay experiment.
Key question: If we lock each layer to its dominant template and disable routing,
does MAE degrade?

Loads best model, replaces VQ routing with fixed codebook templates, runs inference.
No retraining needed.

Usage:
    conda run -n ai_lab python analysis/vq_router/frozen_template_replay.py --exp_id 71098
"""
import os
import sys
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
    checkpoint = torch.load(os.path.join(cache_dir, m_files[0]), map_location='cpu')

    data_feature = dataset.get_data_feature()
    model = DGSTA(config, data_feature)
    if isinstance(checkpoint, tuple):
        model.load_state_dict(checkpoint[0], strict=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # lap_mx
    import scipy.sparse as sp
    adj_mx = model.get_data_feature().get('adj_mx')
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

    return model, dataset.test_dataloader, lap_mx, device, config


def patch_layer_with_frozen_template(layer, template_id):
    """Monkey-patch STSelfAttention.forward to use a fixed codebook template."""
    st_attn = layer.st_attn
    codebook = st_attn.vq_router.graph_codebook  # [10, N, N]
    N = codebook.shape[1]
    frozen_adj = F.relu(codebook[template_id])  # [N, N]

    # Store the original forward
    original_forward = st_attn.forward

    def frozen_forward(x, ind, geo_mask=None, sem_mask=None):
        B, T, N_val, D = x.shape
        device = x.device

        # Use series_decomp (same as original)
        x_res, x_trend = st_attn.series_decomp(x)

        # Build adj_vq from frozen template
        adj_vq = frozen_adj.to(device).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        # Build adj_adp from learned node embeddings (still used, trained)
        raw_adp = F.relu(torch.mm(st_attn.vq_router.nodevec1, st_attn.vq_router.nodevec2))

        # Apply sparsify_graph (same as original)
        adj_vq = st_attn.sparsify_graph(adj_vq)
        adj_adp = st_attn.sparsify_graph(raw_adp)
        new_supports = [adj_vq, adj_adp]

        # GCN on residual (same as original)
        x_gcn_in = x_res.reshape(-1, D)
        x_gcn_in = st_attn.reshape1(x_gcn_in)
        x_gcn_in = x_gcn_in.reshape(B, T, N_val, 32)
        x_gcn_in = x_gcn_in.permute(0, 3, 2, 1)
        x_gcn_out = st_attn.gconv(x_gcn_in, new_supports)
        x_gcn_out = x_gcn_out.permute(0, 3, 2, 1)
        x_gcn_out = x_gcn_out.reshape(-1, 32)
        x_gcn_out = st_attn.reshape2(x_gcn_out)
        x_gcn_out = x_gcn_out.reshape(B, T, N_val, D)
        x_gcn_out = torch.tanh(x_gcn_out)
        x_out = x_trend + x_gcn_out * st_attn.res_scale
        x_out = st_attn.layer_norm(x_out)

        # Temporal + Geo attention (same as original)
        t_q = st_attn.t_q_conv(x_out.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = st_attn.t_k_conv(x_out.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = st_attn.t_v_conv(x_out.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N_val, T, st_attn.t_num_heads, st_attn.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N_val, T, st_attn.t_num_heads, st_attn.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N_val, T, st_attn.t_num_heads, st_attn.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * st_attn.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = st_attn.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N_val, T, int(D * st_attn.t_ratio)).transpose(1, 2)

        geo_q = st_attn.geo_q_conv(x_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_k = st_attn.geo_k_conv(x_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_v = st_attn.geo_v_conv(x_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_q = geo_q.reshape(B, T, N_val, st_attn.geo_num_heads, st_attn.head_dim).permute(0, 1, 3, 2, 4)
        geo_k = geo_k.reshape(B, T, N_val, st_attn.geo_num_heads, st_attn.head_dim).permute(0, 1, 3, 2, 4)
        geo_v = geo_v.reshape(B, T, N_val, st_attn.geo_num_heads, st_attn.head_dim).permute(0, 1, 3, 2, 4)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * st_attn.scale
        if geo_mask is not None:
            geo_attn.masked_fill_(geo_mask, float('-inf'))
        geo_attn = geo_attn.softmax(dim=-1)
        geo_attn = st_attn.geo_attn_drop(geo_attn)
        geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N_val, int(D * st_attn.geo_ratio))

        x = st_attn.proj(torch.cat([t_x, geo_x], dim=-1))
        x = st_attn.proj_drop(x)
        return x

    st_attn.forward = frozen_forward
    return original_forward


def run_evaluation(model, loader, lap_mx, device):
    """Run inference and return per-step MAE."""
    y_truths, y_preds = [], []
    scaler = model._scaler

    with torch.no_grad():
        for batch in loader:
            batch.to_tensor(device)
            output = model.predict(batch, lap_mx=lap_mx)
            y_true = scaler.inverse_transform(batch['y'][..., :model.output_dim])
            y_pred = scaler.inverse_transform(output[..., :model.output_dim])
            y_truths.append(y_true.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())

    y_true = np.concatenate(y_truths, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    mae_per_step = np.mean(np.abs(y_pred - y_true), axis=(0, 2, 3))
    return mae_per_step


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default='71098')
    args = parser.parse_args()

    # Dominant templates per layer (from time-conditioned analysis, exp 71098)
    dominant_templates = {
        0: 6,   # Layer 0 → always T6
        1: 2,   # Layer 1 → global dominant T2 (varies by time but T2 most frequent)
        2: 4,   # Layer 2 → always T4
        3: 0,   # Layer 3 → always T0
        4: 0,   # Layer 4 → always T0
        5: 0,   # Layer 5 → always T0
    }

    print(f"Loading best model from experiment {args.exp_id}...")
    model, loader, lap_mx, device, config = load_best_model(args.exp_id)

    # Step 1: Original model inference
    print("Running ORIGINAL VQ Router model...")
    mae_orig = run_evaluation(model, loader, lap_mx, device)
    print(f"  @3={mae_orig[2]:.4f}  @6={mae_orig[5]:.4f}  @12={mae_orig[11]:.4f}")

    # Step 2: Patch layers with frozen dominant templates
    print("\nPatching layers with frozen dominant templates...")
    originals = {}
    for layer_idx, template_id in dominant_templates.items():
        block = model.encoder_blocks[layer_idx]
        originals[layer_idx] = patch_layer_with_frozen_template(block, template_id)
        print(f"  Layer {layer_idx} → frozen to Template {template_id}")

    # Step 3: Frozen model inference
    print("\nRunning FROZEN-TEMPLATE model (no routing)...")
    mae_frozen = run_evaluation(model, loader, lap_mx, device)
    print(f"  @3={mae_frozen[2]:.4f}  @6={mae_frozen[5]:.4f}  @12={mae_frozen[11]:.4f}")

    # Step 4: Report
    print("\n" + "=" * 60)
    print("FROZEN-TEMPLATE STATIC REPLAY — RESULTS")
    print("=" * 60)
    print(f"{'Step':<8} {'@3':>8} {'@6':>8} {'@12':>8}")
    print(f"{'Original':<8} {mae_orig[2]:>8.4f} {mae_orig[5]:>8.4f} {mae_orig[11]:>8.4f}")
    print(f"{'Frozen':<8} {mae_frozen[2]:>8.4f} {mae_frozen[5]:>8.4f} {mae_frozen[11]:>8.4f}")
    diff = mae_frozen - mae_orig
    print(f"{'Delta':<8} {diff[2]:>+8.4f} {diff[5]:>+8.4f} {diff[11]:>+8.4f}")

    if np.mean(diff[2:12]) < 0.05:
        print("\nINTERPRETATION: Frozen templates maintain performance.")
        print("This supports the hypothesis that inference-time dynamic routing")
        print("contributes minimally — gain is from depth-wise graph specialization.")
    else:
        print("\nINTERPRETATION: Significant degradation when routing is frozen.")
        print("Training-time routing dynamics may still be important.")
    print("=" * 60)
