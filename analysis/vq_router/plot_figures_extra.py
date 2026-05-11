"""
Generate additional paper figures: frozen replay, template visualization, semantic suppression.
"""
import os, sys, math, numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = '/home/user/DeepLearning/DGSTA_Codex_Workspace/DGSTA_clean'
sys.path.insert(0, project_root)
os.chdir(project_root)
os.makedirs('analysis/vq_router/figures', exist_ok=True)

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
                     'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300})

# ============================================================
# Figure 5: Frozen Replay — Original vs Frozen MAE
# ============================================================
def fig5_frozen_replay():
    # Results from frozen_template_replay.py on exp 71098
    steps = list(range(1, 13))
    original = [11.523, 11.768, 11.981, 12.166, 12.329, 12.475,
                12.613, 12.741, 12.864, 12.987, 13.158, 13.304]
    frozen   = [11.532, 11.782, 11.997, 12.182, 12.345, 12.488,
                12.623, 12.751, 12.873, 12.996, 13.169, 13.322]
    # NOTE: above are from frozen_template_replay.py masked_MAE values
    # Replace with actual values from our run:
    # From the frozen replay experiment output:
    original_mae = [12.3644, 13.1542, 14.7871]  # @3, @6, @12 (unmasked MAE from script)
    frozen_mae   = [12.3742, 13.1676, 14.8096]
    diff = [f - o for f, o in zip(frozen_mae, original_mae)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: group bar @3, @6, @12
    x = np.arange(3)
    width = 0.35
    ax1.bar(x - width/2, original_mae, width, label='Original (VQ Router)', color='#1f77b4')
    ax1.bar(x + width/2, frozen_mae, width, label='Frozen (Static Templates)', color='#d62728')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['@3 (15min)', '@6 (30min)', '@12 (60min)'])
    ax1.set_ylabel('MAE')
    ax1.set_title('Frozen Template Replay:\nLocking Routing Causes <0.2% Degradation')
    ax1.legend()

    # Right: delta bar
    colors_diff = ['#d62728' if d > 0 else '#2ca02c' for d in diff]
    ax2.bar(x, diff, color=colors_diff, edgecolor='white', width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['@3', '@6', '@12'])
    ax2.set_ylabel('MAE Change (Frozen - Original)')
    ax2.set_title('Degradation < 0.03 MAE at All Horizons')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    for i, d in enumerate(diff):
        ax2.text(i, d + 0.002, f'{d:+.4f}', ha='center', fontweight='bold',
                 color=colors_diff[i])

    fig.suptitle('Frozen Template Replay: Inference-Time Dynamic Routing Contributes Minimally',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig('analysis/vq_router/figures/fig5_frozen_replay.png')
    plt.close(fig)
    print("Fig 5: frozen_replay.png")


# ============================================================
# Figure 6: Template Adjacency Visualization
# ============================================================
def fig6_template_visualization():
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
    model.eval()

    # Collect codebooks from all layers
    codebooks = {}
    for layer_idx, block in enumerate(model.encoder_blocks):
        if hasattr(block.st_attn, 'vq_router'):
            cb = block.st_attn.vq_router.graph_codebook.detach().cpu().numpy()
            codebooks[layer_idx] = cb

    K, N = codebooks[0].shape[0], codebooks[0].shape[1]  # 10 templates, 170 nodes

    # For each of the 3 dominant layers, show the template they use
    # Layer 0→T6, Layer 1→T2, Layer 2→T4 (dominant), Layer 3→T0
    dom_layers = {0: 6, 1: 2, 2: 4, 3: 0}

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, (layer, tid) in enumerate(dom_layers.items()):
        ax = axes[idx // 4, idx % 4]
        adj = codebooks[layer][tid]  # [N, N]
        # Apply sparsify_graph logic for visualization (top-20 per row)
        top_k = 20
        adj_viz = adj.copy()
        for i in range(N):
            row = adj_viz[i]
            threshold = np.sort(row)[-top_k]
            adj_viz[i, row < threshold] = 0
        im = ax.imshow(adj_viz, cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_title(f'L{layer} → Template {tid}')
        ax.set_xlabel('Node'); ax.set_ylabel('Node')

    # Show 4 other random templates from Layer 0
    other_tids = [t for t in range(K) if t not in dom_layers.values()][:4]
    for idx, tid in enumerate(other_tids):
        ax = axes[1, idx] if idx < 4 else axes[1, 0]
        adj = codebooks[0][tid]
        adj_viz = adj.copy()
        top_k = 20
        for i in range(N):
            row = adj_viz[i]
            threshold = np.sort(row)[-top_k]
            adj_viz[i, row < threshold] = 0
        im = axes[1, idx].imshow(adj_viz, cmap='YlOrRd', aspect='auto', vmin=0)
        axes[1, idx].set_title(f'L0 → Template {tid} (unused)')
        axes[1, idx].set_xlabel('Node')

    fig.suptitle('Graph Template Visualization: Layer-Dominant vs Unused Templates',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig('analysis/vq_router/figures/fig6_template_viz.png')
    plt.close(fig)
    print("Fig 6: template_viz.png")


# ============================================================
# Figure 7: Semantic Suppression Evidence
# ============================================================
def fig7_semantic_suppression():
    horizons = ['@3', '@6', '@12']
    # Pure SemAttn model (exp 41108, no VQ, no Trend, no Delay)
    sem_only = [11.980, 12.468, 13.191]
    # VQ Router model (exp 71098, no sem in VQ branch)
    vq_only  = [11.894, 12.406, 13.228]
    # VQ+Sem coexistence (exp 58301 or seed=2)
    vq_sem   = [12.056, 12.565, 13.354]  # seed=2 result

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    width = 0.25

    ax.bar(x - width, sem_only, width, label='Semantic Attention Only\n(exp 41108, @12 best)',
           color='#2ca02c', edgecolor='white')
    ax.bar(x, vq_only, width, label='VQ Router (no Sem)\n(exp 71098, @3/@6 best)',
           color='#1f77b4', edgecolor='white')
    ax.bar(x + width, vq_sem, width, label='VQ + Sem Coexist\n(seed=2, unstable)',
           color='#9467bd', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(['@3 (15min)', '@6 (30min)', '@12 (60min)'])
    ax.set_ylabel('MAE')
    ax.set_title('Semantic Suppression Evidence:\nPure SemAttn Beats VQ at @12, VQ Beats Sem at @3/@6')

    # Annotate values
    for i, (s, v, c) in enumerate(zip(sem_only, vq_only, vq_sem)):
        ax.text(i - width, s + 0.02, f'{s:.3f}', ha='center', fontsize=8)
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
        ax.text(i + width, c + 0.02, f'{c:.3f}', ha='center', fontsize=8)

    ax.legend(fontsize=9)
    ax.set_ylim(11.5, 13.6)

    # Add insight text
    ax.annotate('SemAttn dominates long horizon',
                xy=(2, sem_only[2]), xytext=(1.5, 12.8),
                arrowprops=dict(arrowstyle='->', color='#2ca02c'), color='#2ca02c', fontsize=10)
    ax.annotate('VQ dominates short horizon',
                xy=(0, vq_only[0]), xytext=(0.5, 13.0),
                arrowprops=dict(arrowstyle='->', color='#1f77b4'), color='#1f77b4', fontsize=10)

    fig.tight_layout()
    fig.savefig('analysis/vq_router/figures/fig7_semantic_suppression.png')
    plt.close(fig)
    print("Fig 7: semantic_suppression.png")


if __name__ == '__main__':
    fig5_frozen_replay()
    fig6_template_visualization()
    fig7_semantic_suppression()
    print("\nDone. 3 extra figures saved to analysis/vq_router/figures/")
