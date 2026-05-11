"""
Layer-wise perturbation study: freeze each layer's routing individually,
measure MAE impact. Tests whether high-entropy layers (L1) are functionally
important or just noisy.

Hypothesis: If L1's routing variability is functional (conditional routing),
freezing it should cause the largest MAE degradation.

Usage:
    conda run -n ai_lab python analysis/vq_router/layer_perturbation.py
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

plt.rcParams.update({'font.size':11,'axes.labelsize':12,'axes.titlesize':13,
                     'legend.fontsize':9,'figure.dpi':150,'savefig.dpi':300})


def load_model():
    from libcity.config import ConfigParser
    from libcity.data import get_dataset
    from libcity.model.traffic_flow_prediction.DGSTA import DGSTA
    import scipy.sparse as sp
    config = ConfigParser('traffic_state_pred','DGSTA','PeMS08','PeMS08',{'gpu_id':[0]})
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
    return model, dataset.test_dataloader, lap_mx, device


def run_eval(model, loader, lap_mx, device):
    scaler = model._scaler
    yt, yp = [], []
    with torch.no_grad():
        for batch in loader:
            batch.to_tensor(device)
            out = model.predict(batch, lap_mx=lap_mx)
            yt.append(scaler.inverse_transform(batch['y'][...,:1]).cpu().numpy())
            yp.append(scaler.inverse_transform(out[...,:1]).cpu().numpy())
    yt = np.concatenate(yt,0); yp = np.concatenate(yp,0)
    return np.mean(np.abs(yp-yt), axis=(0,2,3))  # [12] per-step MAE


# Dominant templates per layer (from layer_specialization analysis)
dominant = {0:6, 1:2, 2:4, 3:0, 4:0, 5:0}

print("Loading model...")
model, loader, lap_mx, device = load_model()

# Baseline (original routing)
print("Evaluating ORIGINAL...")
mae_orig = run_eval(model, loader, lap_mx, device)
print(f"  @3={mae_orig[2]:.4f}  @6={mae_orig[5]:.4f}  @12={mae_orig[11]:.4f}")

# Per-layer perturbation
results = {}
for freeze_li in range(6):
    print(f"Freezing Layer {freeze_li} (→T{dominant[freeze_li]})...")

    # Patch this specific layer
    block = model.encoder_blocks[freeze_li]
    st_attn = block.st_attn
    codebook = st_attn.vq_router.graph_codebook
    frozen_adj = F.relu(codebook[dominant[freeze_li]])

    original_forward = st_attn.forward
    def make_frozen_forward(orig_fn, fa, cb):
        def frozen_fn(x, ind, geo_mask=None, sem_mask=None):
            B,T,N,D = x.shape
            x_res, x_trend = st_attn.series_decomp(x)
            adj_vq = fa.to(x.device).unsqueeze(0).unsqueeze(0).expand(B,T,-1,-1)
            raw_adp = F.relu(torch.mm(st_attn.vq_router.nodevec1,
                                       st_attn.vq_router.nodevec2))
            adj_vq = st_attn.sparsify_graph(adj_vq)
            adj_adp = st_attn.sparsify_graph(raw_adp)
            supp = [adj_vq, adj_adp]
            xg = x_res.reshape(-1,D); xg = st_attn.reshape1(xg)
            xg = xg.reshape(B,T,N,32).permute(0,3,2,1)
            xg = st_attn.gconv(xg, supp).permute(0,3,2,1)
            xg = xg.reshape(-1,32); xg = st_attn.reshape2(xg)
            xg = xg.reshape(B,T,N,D); xg = torch.tanh(xg)
            xo = x_trend + xg*st_attn.res_scale; xo = st_attn.layer_norm(xo)

            tq = st_attn.t_q_conv(xo.permute(0,3,1,2)).permute(0,3,2,1)
            tk = st_attn.t_k_conv(xo.permute(0,3,1,2)).permute(0,3,2,1)
            tv = st_attn.t_v_conv(xo.permute(0,3,1,2)).permute(0,3,2,1)
            tq=tq.reshape(B,N,T,st_attn.t_num_heads,st_attn.head_dim).permute(0,1,3,2,4)
            tk=tk.reshape(B,N,T,st_attn.t_num_heads,st_attn.head_dim).permute(0,1,3,2,4)
            tv=tv.reshape(B,N,T,st_attn.t_num_heads,st_attn.head_dim).permute(0,1,3,2,4)
            ta=(tq@tk.transpose(-2,-1))*st_attn.scale
            ta=ta.softmax(dim=-1); ta=st_attn.t_attn_drop(ta)
            tx=(ta@tv).transpose(2,3).reshape(B,N,T,int(D*st_attn.t_ratio)).transpose(1,2)

            gq=st_attn.geo_q_conv(xo.permute(0,3,1,2)).permute(0,2,3,1)
            gk=st_attn.geo_k_conv(xo.permute(0,3,1,2)).permute(0,2,3,1)
            gv=st_attn.geo_v_conv(xo.permute(0,3,1,2)).permute(0,2,3,1)
            gq=gq.reshape(B,T,N,st_attn.geo_num_heads,st_attn.head_dim).permute(0,1,3,2,4)
            gk=gk.reshape(B,T,N,st_attn.geo_num_heads,st_attn.head_dim).permute(0,1,3,2,4)
            gv=gv.reshape(B,T,N,st_attn.geo_num_heads,st_attn.head_dim).permute(0,1,3,2,4)
            ga=(gq@gk.transpose(-2,-1))*st_attn.scale
            if geo_mask is not None: ga.masked_fill_(geo_mask, float('-inf'))
            ga=ga.softmax(dim=-1); ga=st_attn.geo_attn_drop(ga)
            gx=(ga@gv).transpose(2,3).reshape(B,T,N,int(D*st_attn.geo_ratio))
            x = st_attn.proj(torch.cat([tx,gx],dim=-1))
            return st_attn.proj_drop(x)
        return frozen_fn

    st_attn.forward = make_frozen_forward(st_attn.forward, frozen_adj, codebook)
    mae_f = run_eval(model, loader, lap_mx, device)
    st_attn.forward = original_forward  # restore
    results[freeze_li] = mae_f
    print(f"  @3={mae_f[2]:.4f}  @6={mae_f[5]:.4f}  @12={mae_f[11]:.4f}")

# ---- Visualization ----
entropy_vals = [-0.0000, 0.4747, 0.0002, -0.0000, -0.0000, -0.0000]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: MAE degradation per layer
x = np.arange(6)
mae3_d = [results[i][2]-mae_orig[2] for i in range(6)]
mae6_d = [results[i][5]-mae_orig[5] for i in range(6)]
mae12_d = [results[i][11]-mae_orig[11] for i in range(6)]
width=0.25
ax1.bar(x-width, mae3_d, width, label='@3 (15min)', color='#d62728', edgecolor='white')
ax1.bar(x, mae6_d, width, label='@6 (30min)', color='#1f77b4', edgecolor='white')
ax1.bar(x+width, mae12_d, width, label='@12 (60min)', color='#2ca02c', edgecolor='white')
ax1.set_xticks(x)
ax1.set_xticklabels([f'L{i}\n(T{dominant[i]})' for i in range(6)])
ax1.set_ylabel('MAE Change (Frozen - Original)')
ax1.set_title('Layer Perturbation: Freeze Each Layer Individually')
ax1.axhline(y=0, color='gray', linewidth=0.5)
ax1.legend(fontsize=8)

# Right: entropy vs MAE impact correlation
mae3_mean_impact = [np.mean([results[i][j]-mae_orig[j] for j in [2,5,11]]) for i in range(6)]
ax2.scatter(entropy_vals, mae3_mean_impact, s=200, c=range(6), cmap='tab10',
            edgecolors='black', linewidth=1, zorder=5)
for i in range(6):
    ax2.annotate(f'L{i}', (entropy_vals[i], mae3_mean_impact[i]),
                 textcoords="offset points", xytext=(8,5), fontsize=11, fontweight='bold')
ax2.set_xlabel('Mean Routing Entropy H_norm')
ax2.set_ylabel('Mean MAE Impact (Frozen - Original)')
ax2.set_title('Entropy vs Functional Importance')
ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

# Correlation note
corr = np.corrcoef(entropy_vals, mae3_mean_impact)[0,1]
ax2.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=ax2.transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('Layer Perturbation Study: Does High-Entropy Routing Serve a Function?',
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig('analysis/vq_router/figures/fig9_layer_perturbation.png')
print(f"\nFig 9 saved. Pearson r = {corr:.3f}")
print("\nPer-layer MAE impact:")
for i in range(6):
    avg = np.mean([results[i][j]-mae_orig[j] for j in [2,5,11]])
    print(f"  L{i}: ΔMAE={avg:+.4f}  Entropy={entropy_vals[i]:.4f}")
