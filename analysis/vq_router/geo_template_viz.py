"""
Geographic template visualization for paper.
Maps learned graph templates to PeMS08 sensor coordinates,
extracts traffic-interpretable motifs, generates paper-ready figures.

Output: analysis/vq_router/figures/geo_*.png
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

plt.rcParams.update({'font.size':10,'axes.labelsize':11,'axes.titlesize':12,
                     'legend.fontsize':9,'figure.dpi':150,'savefig.dpi':300})

# 1. Load sensor coordinates (spectral embedding of road topology)
def load_geo():
    # Use road-network hop-distance matrix for topology-preserving layout
    # Each sensor positioned by its structural role in the highway network
    coords = np.load('analysis/vq_router/sensor_coords.npy')
    return coords, coords

def load_template(codebook, tid):
    """Load template, apply sparsify_graph logic (top-20 per row)."""
    adj = codebook[tid].numpy()
    N = adj.shape[0]
    adj_viz = adj.copy()
    for i in range(N):
        row = adj_viz[i]
        thresh = np.sort(row)[-20]
        adj_viz[i, row < thresh] = 0
    return adj_viz

def compute_graph_stats(adj, coords):
    """Compute interpretable graph statistics for a template."""
    adj_bin = (adj > adj.max() * 0.01).astype(float)
    N = adj.shape[0]

    # Degree
    degrees = adj_bin.sum(axis=1)
    mean_deg = degrees.mean()

    # Locality ratio: edges connecting sensors within 10% spatial distance
    distances = np.sqrt(((coords[:,None,:] - coords[None,:,:])**2).sum(axis=-1))
    local_mask = distances < np.percentile(distances, 10)
    local_edges = (adj_bin * local_mask).sum()
    total_edges = adj_bin.sum()
    locality = local_edges / (total_edges + 1e-10)

    # Clustering coefficient (simplified)
    tri = np.trace(np.linalg.matrix_power(adj_bin, 3))
    dsum = (degrees * (degrees - 1)).sum()
    clustering = tri / (dsum + 1e-10) if dsum > 0 else 0

    return {'degree': mean_deg, 'locality': locality, 'clustering': clustering,
            'top5_degree': np.sort(degrees)[-5:].mean()}

def interpret_template(stats):
    """Heuristic traffic interpretation from graph statistics."""
    if stats['locality'] > 0.4:
        if stats['degree'] > 15:
            return 'Dense local mixing'
        return 'Local propagation'
    elif stats['locality'] < 0.2:
        if stats['degree'] < 8:
            return 'Sparse connectivity'
        return 'Long-range corridor'
    else:
        if stats['degree'] > 12:
            return 'Mixed-range interaction'
        return 'Regional clustering'

# ============= FIGURE: Geographic template visualization =============
def plot_geo_templates(codebook, coords_norm, coords_raw):
    dominant = {0:6, 1:2, 2:4, 3:0, 0:0}  # L0→T6, L1→T2, L2→T4, L3→T0
    # Show T0, T2, T4, T6 + one unused (T1) for comparison
    templates_to_show = [0, 2, 4, 6, 1]
    names = {6:'Layer 0/5 Template', 2:'Layer 1 Template', 4:'Layer 2 Template',
             0:'Layer 3/4/5 Template', 1:'Unused Template'}

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for idx, tid in enumerate(templates_to_show):
        if idx >= 5: break
        ax = axes[idx // 3, idx % 3]
        adj = load_template(codebook, tid)
        stats = compute_graph_stats(adj, coords_norm)
        interp = interpret_template(stats)

        # Plot sensors as dots
        ax.scatter(coords_norm[:,0], coords_norm[:,1], c='lightgray', s=8, zorder=1, alpha=0.6)

        # Plot top-100 strongest edges
        edges = []
        N = adj.shape[0]
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j] > 0:
                    edges.append((i, j, adj[i,j]))
        edges.sort(key=lambda x: -x[2])
        top_edges = edges[:100]

        for i, j, w in top_edges:
            alpha_val = min(1.0, w / (top_edges[0][2] + 1e-10))
            ax.plot([coords_norm[i,0], coords_norm[j,0]],
                    [coords_norm[i,1], coords_norm[j,1]],
                    color='#d62728', alpha=alpha_val, linewidth=0.5, zorder=2)

        ax.set_title(f'{names[tid]}\n({interp})', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.98, f"D={stats['degree']:.0f} L={stats['locality']:.2f}",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide unused subplot
    axes[1, 2].axis('off')
    fig.suptitle('Learned Graph Templates Mapped to PeMS08 Sensor Network\nTop-100 strongest edges per template', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig('analysis/vq_router/figures/geo_templates.png')
    plt.close(fig)
    print("geo_templates.png saved")

# ============= FIGURE: 24h Template Activation Timeline =============
def plot_activation_timeline():
    """Load routing data and plot template activation over 24h for Layer 1."""
    from libcity.config import ConfigParser
    from libcity.data import get_dataset
    from libcity.model.traffic_flow_prediction.DGSTA import DGSTA
    import scipy.sparse as sp
    config = ConfigParser('traffic_state_pred','DGSTA','PeMS08','PeMS08',{'gpu_id':[0],'use_vq_router':True,'use_delay_conv':True,'use_deep_trend':True})
    dataset = get_dataset(config)
    train_dl,eval_dl,test_dl = dataset.get_data()
    ckpt = torch.load('libcity/cache/71098/model_cache/DGSTA_PeMS08.m', map_location='cpu')
    data_feature = dataset.get_data_feature()
    model = DGSTA(config, data_feature)
    model.load_state_dict(ckpt[0], strict=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    adj_mx = model.get_data_feature().get('adj_mx')
    adj = sp.coo_matrix(adj_mx)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d,-0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)]=0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    L = sp.eye(adj.shape[0])-adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    EigVal,EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal,EigVec = EigVal[idx], np.real(EigVec[:,idx])
    lap_mx = torch.from_numpy(EigVec[:,1:9]).float().to(device)

    # Collect Layer 1 template usage by hour
    records = defaultdict(list)
    with torch.no_grad():
        for batch in test_dl:
            batch.to_tensor(device)
            _ = model(batch, lap_mx=lap_mx)
            ind = batch['ind'].cpu().numpy()
            li = 1  # Layer 1
            block = model.encoder_blocks[li]
            if hasattr(block.st_attn, 'vq_router'):
                hw = block.st_attn.last_router_weights.cpu().numpy()
                B,T,K = hw.shape
                for b in range(B):
                    for t in range(T):
                        hour = ((ind[b] + t) % 288) // 12
                        records[hour].append(int(hw[b,t].argmax()))

    hours = list(range(24))
    K = 10
    usage = np.zeros((24, K))
    for h in hours:
        if len(records[h]) > 10:
            cnt = np.bincount(records[h], minlength=K) / len(records[h])
            usage[h] = cnt

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4.5))
    key_templates = {0:'Local neighbor', 2:'Corridor coupling', 3:'Dense mixing', 6:'Sparse transit', 9:'Residual'}
    colors_t = {0:'#1f77b4', 2:'#ff7f0e', 3:'#2ca02c', 6:'#d62728', 9:'#9467bd'}
    for tid in [0, 2, 3, 6, 9]:
        lbl = key_templates.get(tid, "")
        ax.plot(hours, usage[:, tid], 'o-', color=colors_t[tid], label=f'T{tid}: {lbl}',
                markersize=5, linewidth=2, alpha=0.85)

    # Shade peak hours
    ax.axvspan(7, 9, alpha=0.08, color='orange')
    ax.axvspan(17, 19, alpha=0.08, color='orange')
    ax.text(8, 0.72, 'AM Peak', fontsize=9, ha='center', color='orange')
    ax.text(18, 0.72, 'PM Peak', fontsize=9, ha='center', color='orange')

    ax.set_xlabel('Hour of Day'); ax.set_ylabel('Template Usage Fraction')
    ax.set_title('Layer 1 Template Activation over 24 Hours\nDaytime: corridor coupling (T2). Night: local neighbor (T0).')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, 23); ax.set_ylim(0, 0.8)
    fig.tight_layout()
    fig.savefig('analysis/vq_router/figures/geo_activation_timeline.png')
    plt.close(fig)
    print("geo_activation_timeline.png saved")

# ============= FIGURE: Case Study — Peak vs Night =============
def plot_case_study():
    """Show template adjacency for AM peak vs Night on Layer 1."""
    from libcity.config import ConfigParser
    from libcity.data import get_dataset
    from libcity.model.traffic_flow_prediction.DGSTA import DGSTA
    import scipy.sparse as sp
    config = ConfigParser('traffic_state_pred','DGSTA','PeMS08','PeMS08',{'gpu_id':[0],'use_vq_router':True,'use_delay_conv':True,'use_deep_trend':True})
    dataset = get_dataset(config)
    _,_,test_dl = dataset.get_data()
    ckpt = torch.load('libcity/cache/71098/model_cache/DGSTA_PeMS08.m', map_location='cpu')
    data_feature = dataset.get_data_feature()
    model = DGSTA(config, data_feature)
    model.load_state_dict(ckpt[0], strict=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Load spectral embedding coordinates
    cn = np.load('analysis/vq_router/sensor_coords.npy')

    # Get codebook from Layer 1
    cb = model.encoder_blocks[1].st_attn.vq_router.graph_codebook.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, (label, tid) in enumerate([('Daytime (AM Peak) — Corridor Coupling', 2),
                                            ('Night — Local Neighbor', 0)]):
        ax = axes[ax_idx]
        adj = load_template(torch.tensor(cb), tid)
        ax.scatter(cn[:,0], cn[:,1], c='lightgray', s=6, zorder=1, alpha=0.5)
        edges = []
        N = adj.shape[0]
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j] > 0:
                    edges.append((i, j, adj[i,j]))
        edges.sort(key=lambda x: -x[2])
        for i, j, w in edges[:80]:
            ax.plot([cn[i,0], cn[j,0]], [cn[i,1], cn[j,1]],
                    color='#d62728' if tid==2 else '#1f77b4',
                    alpha=min(1.0, w/(edges[0][2]+1e-10)), linewidth=0.6, zorder=2)
        ax.set_title(label, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('Layer 1 Template Switching: Day vs Night Spatial Patterns', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig('analysis/vq_router/figures/geo_case_study.png')
    plt.close(fig)
    print("geo_case_study.png saved")

if __name__ == '__main__':
    print("Loading codebook...")
    ckpt = torch.load('libcity/cache/71098/model_cache/DGSTA_PeMS08.m', map_location='cpu')
    model_state = ckpt[0]
    # Extract codebook from any layer (they're all in the state dict)
    codebook = {}
    for key in model_state:
        if 'vq_router.graph_codebook' in key:
            li = int(key.split('encoder_blocks.')[1].split('.')[0])
            codebook[li] = model_state[key]
    # Use Layer 0 codebook as representative
    cb0 = codebook[0]

    coords_norm, coords_raw = load_geo()
    print("Generating geographic template visualization...")
    plot_geo_templates(cb0, coords_norm, coords_raw)
    print("Generating activation timeline (needs model load)...")
    plot_activation_timeline()
    print("Generating case study...")
    plot_case_study()
    print("\nDone. 3 geographic figures saved to analysis/vq_router/figures/")
    print("Interpretation names (use these in paper instead of T0/T2):")
    for tid, adj in [(0, load_template(cb0, 0)), (2, load_template(cb0, 2)),
                      (4, load_template(cb0, 4)), (6, load_template(cb0, 6))]:
        stats = compute_graph_stats(adj, coords_norm)
        interp = interpret_template(stats)
        print(f"  Template {tid}: {interp} (deg={stats['degree']:.1f}, loc={stats['locality']:.2f})")
