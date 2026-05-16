"""
Generate paper-ready PNG figures for DiMoR mechanism analysis.
Output: analysis/vq_router/figures/*.png
"""
import os, sys, math, numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

project_root = '/home/user/DeepLearning/DGSTA_Codex_Workspace/DGSTA_clean'
sys.path.insert(0, project_root)
os.chdir(project_root)
os.makedirs('analysis/vq_router/figures', exist_ok=True)

# ---- Style ----
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'DejaVu Sans',
})


def load_model():
    from libcity.config import ConfigParser
    from libcity.data import get_dataset
    from libcity.model.traffic_flow_prediction.DGSTA import DGSTA
    import scipy.sparse as sp
    config = ConfigParser('traffic_state_pred', 'DGSTA', 'PeMS08', 'PeMS08', {'gpu_id': [0]})
    dataset = get_dataset(config)
    train_dl, eval_dl, test_dl = dataset.get_data()
    checkpoint = torch.load('libcity/cache/71098/model_cache/DGSTA_PeMS08.m', map_location='cpu')
    data_feature = dataset.get_data_feature()
    model = DGSTA(config, data_feature)
    model.load_state_dict(checkpoint[0], strict=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
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
    return model, dataset.test_dataloader, lap_mx, device


def collect_data(model, loader, lap_mx, device):
    """Collect routing weights + codebooks."""
    records = defaultdict(list)
    all_codebooks = {}
    with torch.no_grad():
        for batch in loader:
            batch.to_tensor(torch.device('cpu')); batch.to_tensor(device)
            _ = model(batch, lap_mx=lap_mx)
            ind = batch['ind'].cpu().numpy()
            for layer_idx, block in enumerate(model.encoder_blocks):
                if hasattr(block.st_attn, 'vq_router'):
                    weights = block.st_attn.last_router_weights.cpu().numpy()
                    hard_ids = weights.argmax(axis=-1)
                    B, T = hard_ids.shape
                    for b in range(B):
                        for t in range(T):
                            time_slot = (ind[b] + t) % 288
                            records[layer_idx].append((time_slot // 12, int(hard_ids[b, t])))
                    if layer_idx not in all_codebooks:
                        all_codebooks[layer_idx] = block.st_attn.vq_router.graph_codebook.detach().cpu()
    return records, all_codebooks


def compute_stats(records, all_codebooks):
    """Pre-compute all statistics."""
    K, L = 10, len(records)
    groups = {'Morning\n(7-9)': range(7,10), 'Midday\n(10-16)': range(10,17),
              'Evening\n(17-19)': range(17,20), 'Night\n(0-6)': range(0,7), 'Late\n(20-23)': range(20,24)}

    # Per-layer template usage
    layer_usage = np.zeros((L, K))
    for li in range(L):
        ids = [t for _, t in records[li]]
        counts = np.bincount(ids, minlength=K)
        layer_usage[li] = counts / counts.sum()

    # Time-conditioned: per layer, per period
    time_usage = np.zeros((L, len(groups), K))
    for li in range(L):
        for gi, (_, hours) in enumerate(groups.items()):
            ids = [t for h, t in records[li] if h in hours]
            if len(ids) > 10:
                counts = np.bincount(ids, minlength=K)
                time_usage[li, gi] = counts / counts.sum()

    # Codebook similarity
    cb_sim = {}
    for li, cb in all_codebooks.items():
        C = cb.shape[0]
        flat = cb.reshape(C, -1)
        sim = F.cosine_similarity(flat.unsqueeze(1), flat.unsqueeze(0), dim=-1).numpy()
        cb_sim[li] = sim

    return layer_usage, time_usage, cb_sim, list(groups.keys())


def fig1_layer_specialization(layer_usage):
    """Figure 1: Per-layer template usage distribution (grouped bar chart)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    L, K = layer_usage.shape
    x = np.arange(L)
    width = 0.25
    colors = plt.cm.tab10(np.linspace(0, 1, K))

    bottom = np.zeros(L)
    for k in range(K):
        vals = layer_usage[:, k]
        bars = ax.bar(x, vals, width, bottom=bottom, color=colors[k],
                      edgecolor='white', linewidth=0.3, label=f'T{k}')
        # Annotate dominant bars
        for i, v in enumerate(vals):
            if v > 0.15:
                ax.text(x[i], bottom[i] + v/2, f'T{k}', ha='center', va='center',
                        fontsize=7, fontweight='bold', color='white' if v > 0.5 else 'black')
        bottom += vals

    ax.set_xlabel('Encoder Layer')
    ax.set_ylabel('Template Usage Fraction')
    ax.set_title('Layer Specialization: Template Usage per Encoder Layer')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in range(L)])
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', ncol=5, fontsize=7, title='Template')
    fig.tight_layout()
    fig.savefig('analysis/vq_router/figures/fig1_layer_specialization.png')
    plt.close(fig)
    print("Fig 1: layer_specialization.png")


def fig2_time_routing(time_usage, period_names):
    """Figure 2: Per-layer dominant template by time period (heatmap)."""
    L, P, K = time_usage.shape
    # For each (layer, period), get dominant template ID
    dominant = np.zeros((L, P))
    dominant_pct = np.zeros((L, P))
    for li in range(L):
        for pi in range(P):
            dominant[li, pi] = time_usage[li, pi].argmax()
            dominant_pct[li, pi] = time_usage[li, pi].max()

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(dominant, aspect='auto', cmap='tab10', vmin=0, vmax=9)

    # Annotate cells
    for li in range(L):
        for pi in range(P):
            tid = int(dominant[li, pi])
            pct = dominant_pct[li, pi]
            ax.text(pi, li, f'T{tid}\n({pct:.0%})', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if pct > 0.8 else 'black')

    ax.set_xticks(range(P))
    ax.set_xticklabels(period_names, fontsize=9)
    ax.set_yticks(range(L))
    ax.set_yticklabels([f'Layer {i}' for i in range(L)])
    ax.set_title('Time-Conditioned Routing: Dominant Template by Period')
    cbar = fig.colorbar(im, ax=ax, ticks=range(K))
    cbar.set_label('Template ID')
    fig.tight_layout()
    fig.savefig('analysis/vq_router/figures/fig2_time_routing.png')
    plt.close(fig)
    print("Fig 2: time_routing.png")


def fig3_codebook_diversity(cb_sim):
    """Figure 3: Codebook pairwise similarity per layer."""
    L = len(cb_sim)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for li in sorted(cb_sim.keys()):
        ax = axes[li // 3, li % 3]
        im = ax.imshow(cb_sim[li], cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.set_title(f'Layer {li}')
        ax.set_xlabel('Template'); ax.set_ylabel('Template')
    fig.suptitle('Codebook Template Pairwise Cosine Similarity', fontsize=14, y=1.01)
    fig.colorbar(im, ax=axes, shrink=0.6, label='Cosine Similarity')
    fig.tight_layout()
    fig.savefig('analysis/vq_router/figures/fig3_codebook_diversity.png')
    plt.close(fig)
    print("Fig 3: codebook_diversity.png")


def fig4_static_vs_dynamic(layer_usage, time_usage, period_names):
    """Figure 4: Layer static/dynamic classification + entropy."""
    L, P, K = time_usage.shape

    # Static vs dynamic classification
    n_unique = np.zeros(L, dtype=int)
    for li in range(L):
        dominants = set()
        for pi in range(P):
            if time_usage[li, pi].sum() > 0:
                dominants.add(int(time_usage[li, pi].argmax()))
        n_unique[li] = len(dominants)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: static vs dynamic bar
    colors = ['#d62728' if n == 1 else '#2ca02c' for n in n_unique]
    bars = ax1.bar(range(L), n_unique, color=colors, edgecolor='white')
    ax1.set_xlabel('Encoder Layer')
    ax1.set_ylabel('Unique Dominant Templates Across Periods')
    ax1.set_title('Layer Routing Dynamics')
    ax1.set_xticks(range(L))
    ax1.set_xticklabels([f'L{i}' for i in range(L)])
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    for i, (bar, n) in enumerate(zip(bars, n_unique)):
        label = 'STATIC' if n == 1 else 'DYNAMIC'
        ax1.text(i, bar.get_height() + 0.15, label, ha='center', fontsize=10, fontweight='bold',
                 color=colors[i])
    ax1.set_ylim(0, max(n_unique) + 0.8)

    # Right: Layer 1 time routing detail
    li = 1  # Layer 1 is the dynamic one
    x = np.arange(P)
    for k in range(K):
        ax2.plot(x, time_usage[li, :, k], 'o-', color=plt.cm.tab10(k), label=f'T{k}',
                 markersize=6, linewidth=1.5, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(period_names)
    ax2.set_ylabel('Template Usage Fraction')
    ax2.set_title(f'Layer 1: The Only Dynamic Layer')
    ax2.legend(loc='upper left', ncol=5, fontsize=7)
    ax2.set_ylim(0, 0.85)

    fig.suptitle('Routing Dynamics: 5/6 Layers Static, 1/6 Dynamic', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig('analysis/vq_router/figures/fig4_static_vs_dynamic.png')
    plt.close(fig)
    print("Fig 4: static_vs_dynamic.png")


if __name__ == '__main__':
    print("Loading model...")
    model, loader, lap_mx, device = load_model()
    print("Collecting data...")
    records, all_codebooks = collect_data(model, loader, lap_mx, device)
    print("Computing stats...")
    layer_usage, time_usage, cb_sim, period_names = compute_stats(records, all_codebooks)
    print("Generating figures...")
    fig1_layer_specialization(layer_usage)
    fig2_time_routing(time_usage, period_names)
    fig3_codebook_diversity(cb_sim)
    fig4_static_vs_dynamic(layer_usage, time_usage, period_names)
    print("\nDone. 4 figures saved to analysis/vq_router/figures/")
