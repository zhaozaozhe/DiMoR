import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
import scipy.sparse as sp
from torch.nn.functional import cosine_similarity


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc


class DataEmbedding(nn.Module):
    def __init__(
            self, feature_dim, embed_dim, lape_dim, adj_mx, drop=0.,
            add_time_in_day=False, add_day_in_week=False, device=torch.device('cpu'), num_nodes=170
    ):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.tempp_embedding = nn.Linear(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, lap_mx, tempp):
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x += self.position_encoding(x)
        if self.add_time_in_day:
            x += self.daytime_embedding((origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long())
        if self.add_day_in_week:
            x += self.weekday_embedding(origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3))
        x += self.spatial_embedding(lap_mx)
        x += self.tempp_embedding(tempp).unsqueeze(0).unsqueeze(0)
        x = self.dropout(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :x.shape[2] - self.chomp_size, :].contiguous()


class DelayConv(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.delay_conv = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                                    padding=(0, 0), groups=channels)
        nn.init.dirac_(self.delay_conv.weight)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, (0, 0, self.kernel_size - 1, 0))
        x = self.delay_conv(x)
        return x.permute(0, 2, 3, 1)


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        B, T, N, D = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, 1, T)
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1)
        x_padded = torch.cat([front, x], dim=-1)
        x = self.avg(x_padded)
        x = x.reshape(B, N, D, T).permute(0, 3, 1, 2)
        return x


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = x.permute(0, 3, 2, 1)
        if A.dim() == 3:
            A = A.unsqueeze(1)
        x = torch.matmul(A, x)
        return x.permute(0, 3, 2, 1).contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2, use_delay_conv=False):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in_original = c_in
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order
        self.use_delay_conv = use_delay_conv
        if self.use_delay_conv:
            self.delay_conv = DelayConv(c_in_original, kernel_size=3)
            self.out_proj = nn.Conv2d(c_out, c_out, kernel_size=1)

    def forward(self, x, support):
        if self.use_delay_conv:
            x_perm = x.permute(0, 3, 2, 1)
            x_delay = self.delay_conv(x_perm)
            x_delay = x_delay.permute(0, 3, 2, 1)
            out = [x_delay]
        else:
            out = [x]
        for a in support:
            base = out[0]
            x1 = self.nconv(base, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        if self.use_delay_conv:
            h = self.out_proj(h)
        return h


class SpatialPatternRouter(nn.Module):
    def __init__(self, num_nodes, num_graphs=10, input_dim=64, temperature=1.0, adj_init=None,
                 embed_dim=10, use_time_aware=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.temperature = temperature

        self.graph_codebook = nn.Parameter(torch.randn(num_graphs, num_nodes, num_nodes) * 1e-4)

        if adj_init is not None:
            if isinstance(adj_init, np.ndarray):
                adj_init = torch.from_numpy(adj_init).float()
            if adj_init.max() > 1:
                adj_init = adj_init / (adj_init.max() + 1e-5)
            with torch.no_grad():
                if adj_init.dim() == 2 and adj_init.shape[0] == num_nodes:
                    self.graph_codebook[0].copy_(adj_init)

        self.reduce_dim = nn.Linear(input_dim, 1)
        self.spatial_compressor = nn.Sequential(
            nn.Linear(num_nodes, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.classifier = nn.Linear(32, num_graphs)

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, embed_dim).to(torch.float32), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(embed_dim, num_nodes).to(torch.float32), requires_grad=True)
        nn.init.xavier_normal_(self.nodevec1)
        nn.init.xavier_normal_(self.nodevec2)

        self.use_time_aware = use_time_aware
        if self.use_time_aware:
            self.time_embed = nn.Embedding(288, embed_dim)

        self.consistency_loss = 0.0

    def forward(self, x_trend, ind=None, training=True):
        traffic_intensity = self.reduce_dim(x_trend).squeeze(-1)
        pattern_emb = self.spatial_compressor(traffic_intensity)
        logits = self.classifier(pattern_emb)

        if training:
            diff = logits[:, 1:, :] - logits[:, :-1, :]
            self.consistency_loss = torch.mean(diff ** 2)
        else:
            self.consistency_loss = 0.0

        weights = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
        self.routing_probs = F.softmax(logits / self.temperature, dim=-1)
        adj_vq = torch.einsum('btk, knm -> btnm', weights, self.graph_codebook)
        adj_vq = F.relu(adj_vq)

        if self.use_time_aware and ind is not None:
            time_idx = ind % 288
            time_emb = self.time_embed(time_idx)
            nodevec1_t = self.nodevec1.unsqueeze(0) * time_emb.unsqueeze(1)
            adj_adp = F.relu(torch.matmul(nodevec1_t, self.nodevec2.unsqueeze(0)))
        else:
            adj_adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))

        return adj_vq, adj_adp, weights


class STSelfAttention(nn.Module):
    def __init__(
            self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, qkv_bias=False,
            attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1, num_nodes=170,
            use_vq_router=False, use_delay_conv=False, adj_init=None, use_time_aware_adp=True
    ):
        super().__init__()
        assert dim % (geo_num_heads + sem_num_heads + t_num_heads) == 0
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.t_num_heads = t_num_heads
        self.head_dim = dim // (geo_num_heads + sem_num_heads + t_num_heads)
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.s_attn_size = s_attn_size
        self.t_attn_size = t_attn_size
        self.geo_ratio = geo_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.sem_ratio = sem_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.t_ratio = 1 - self.geo_ratio - self.sem_ratio
        self.output_dim = output_dim
        self.use_vq_router = use_vq_router

        self.geo_q_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_k_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_v_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_attn_drop = nn.Dropout(attn_drop)

        self.sem_q_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_k_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_v_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_attn_drop = nn.Dropout(attn_drop)

        self.t_q_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.reshape1 = nn.Linear(dim, 32)
        self.reshape2 = nn.Linear(32, dim)

        if self.use_vq_router:
            self.series_decomp = SeriesDecomposition(kernel_size=5)
            self.num_graphs = 10
            self.vq_router = SpatialPatternRouter(num_nodes, self.num_graphs, input_dim=dim,
                                                  temperature=1.0, adj_init=adj_init,
                                                  use_time_aware=use_time_aware_adp).to(device)
            self.layer_norm = nn.LayerNorm(dim)
            self.res_scale = nn.Parameter(torch.ones(1) * 0.1)
            self.top_k = 20
            self.adj_temperature = 0.5
            support_len = 2
        else:
            self.days = 288
            dims = 40
            self.supports_len = 1
            torch.cuda.manual_seed_all(1)
            self.nodevec_p1 = nn.Parameter(torch.randn(self.days, dims).to(device), requires_grad=True).to(device)
            self.nodevec_p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
            self.nodevec_p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
            self.nodevec_pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
            support_len = 1

        self.gconv = gcn(32, 32, 0.3, support_len=support_len, order=2, use_delay_conv=use_delay_conv)

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.relu(adp)
        adp = F.softmax(adp, dim=2)
        return adp

    def sparsify_graph(self, adj):
        adj = adj / self.adj_temperature
        if self.top_k < adj.shape[-1]:
            mask = torch.zeros_like(adj)
            topk_val, topk_idx = torch.topk(adj, k=self.top_k, dim=-1)
            mask.scatter_(-1, topk_idx, 1.0)
            adj = adj.masked_fill(mask == 0, -1e9)
        return F.softmax(adj, dim=-1)

    def forward(self, x, ind, geo_mask=None, sem_mask=None):
        B, T, N, D = x.shape

        if self.use_vq_router:
            x_res, x_trend = self.series_decomp(x)

            raw_vq, raw_adp, weights = self.vq_router(x_trend, ind=ind, training=self.training)
            self.last_router_weights = weights.detach()
            adj_vq = self.sparsify_graph(raw_vq)
            adj_adp = self.sparsify_graph(raw_adp)
            new_supports = [adj_vq, adj_adp]

            x_gcn_in = x_res.reshape(-1, D)
            x_gcn_in = self.reshape1(x_gcn_in)
            x_gcn_in = x_gcn_in.reshape(B, T, N, 32)
            x_gcn_in = x_gcn_in.permute(0, 3, 2, 1)

            x_gcn_out = self.gconv(x_gcn_in, new_supports)

            x_gcn_out = x_gcn_out.permute(0, 3, 2, 1)
            x_gcn_out = x_gcn_out.reshape(-1, 32)
            x_gcn_out = self.reshape2(x_gcn_out)
            x_gcn_out = x_gcn_out.reshape(B, T, N, D)
            x_gcn_out = torch.tanh(x_gcn_out)

            x = x_trend + x_gcn_out * self.res_scale
            x = self.layer_norm(x)
        else:
            ind %= self.days
            ind = ind.cpu().numpy()
            adp = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
            new_supports = [adp]

            x = x.reshape(-1, D)
            x = self.reshape1(x)
            x = x.reshape(B, T, N, 32)
            x = x.permute(0, 3, 2, 1)
            x = self.gconv(x, new_supports)
            x = x.permute(0, 3, 2, 1)
            x = x.reshape(-1, 32)
            x = self.reshape2(x)
            x = x.reshape(B, T, N, D)

        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, int(D * self.t_ratio)).transpose(1, 2)

        geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_q = geo_q.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_k = geo_k.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_v = geo_v.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale
        if geo_mask is not None:
            geo_attn.masked_fill_(geo_mask, float('-inf'))
        geo_attn = geo_attn.softmax(dim=-1)
        geo_attn = self.geo_attn_drop(geo_attn)
        geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N, int(D * self.geo_ratio))

        sem_q = self.sem_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_k = self.sem_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_v = self.sem_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_q = sem_q.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_k = sem_k.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_v = sem_v.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        if sem_mask is not None:
            sem_attn.masked_fill_(sem_mask, float('-inf'))
        sem_attn = sem_attn.softmax(dim=-1)
        sem_attn = self.sem_attn_drop(sem_attn)
        sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, N, int(D * self.sem_ratio))
        x = self.proj(torch.cat([t_x, geo_x, sem_x], dim=-1))
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class STEncoderBlock(nn.Module):

    def __init__(
            self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, mlp_ratio=4.,
            qkv_bias=True, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'), type_ln="pre",
            output_dim=1, num_nodes=170,
            use_vq_router=False, use_delay_conv=False, adj_init=None, use_time_aware_adp=True
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.st_attn = STSelfAttention(
            dim, s_attn_size, t_attn_size, geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads,
            t_num_heads=t_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, device=device, output_dim=output_dim, num_nodes=num_nodes,
            use_vq_router=use_vq_router, use_delay_conv=use_delay_conv, adj_init=adj_init,
            use_time_aware_adp=use_time_aware_adp
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, ind, geo_mask=None, sem_mask=None):
        if self.type_ln == 'pre':
            x = x + self.drop_path(
                self.st_attn(self.norm1(x), ind, geo_mask=geo_mask, sem_mask=sem_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1(
                x + self.drop_path(self.st_attn(x, ind, geo_mask=geo_mask, sem_mask=sem_mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def norm_embedding(adj):
    torch.fill_(adj.diagonal(), 0)
    values, indices = torch.topk(adj, 5, dim=1)
    b = torch.zeros_like(adj)
    b.scatter_(1, indices, 1.0)
    return b


class DeepTrendNet(nn.Module):
    def __init__(self, input_dim, input_window, output_window, hidden_dim=64):
        super().__init__()
        self.feature_proj = nn.Linear(input_dim, 1)
        self.mlp1 = nn.Linear(input_window, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, output_window)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        nn.init.zeros_(self.mlp3.weight)
        nn.init.zeros_(self.mlp3.bias)

    def forward(self, x):
        x_val = self.feature_proj(x)
        x_val = x_val.permute(0, 2, 3, 1)
        h1 = self.act(self.mlp1(x_val))
        h2 = self.act(self.mlp2(self.dropout(h1))) + h1
        out = self.mlp3(h2)
        return out.permute(0, 3, 1, 2)


class DGSTA(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get("num_nodes", 1)
        self.feature_dim = self.data_feature.get("feature_dim", 1)
        self.ext_dim = self.data_feature.get("ext_dim", 0)
        self.num_batches = self.data_feature.get('num_batches', 1)
        self.dtw_matrix = self.data_feature.get('dtw_matrix')
        self.adj_mx = data_feature.get('adj_mx')
        sd_mx = data_feature.get('sd_mx')
        sh_mx = data_feature.get('sh_mx')
        self._logger = getLogger()
        self.dataset = config.get('dataset')

        self.embed_dim = config.get('embed_dim', 64)
        self.skip_dim = config.get("skip_dim", 256)
        lape_dim = config.get('lape_dim', 8)
        geo_num_heads = config.get('geo_num_heads', 4)
        sem_num_heads = config.get('sem_num_heads', 2)
        t_num_heads = config.get('t_num_heads', 2)
        mlp_ratio = config.get("mlp_ratio", 4)
        qkv_bias = config.get("qkv_bias", True)
        drop = config.get("drop", 0.)
        attn_drop = config.get("attn_drop", 0.)
        drop_path = config.get("drop_path", 0.3)
        self.s_attn_size = config.get("s_attn_size", 3)
        self.t_attn_size = config.get("t_attn_size", 3)
        enc_depth = config.get("enc_depth", 6)
        type_ln = config.get("type_ln", "pre")
        self.type_short_path = config.get("type_short_path", "hop")

        self.output_dim = config.get('output_dim', 1)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)
        add_time_in_day = config.get("add_time_in_day", True)
        add_day_in_week = config.get("add_day_in_week", True)
        self.device = config.get('device', torch.device('cpu'))
        self.world_size = config.get('world_size', 1)
        self.huber_delta = config.get('huber_delta', 1)
        self.quan_delta = config.get('quan_delta', 0.25)
        self.far_mask_delta = config.get('far_mask_delta', 5)
        self.dtw_delta = config.get('dtw_delta', 5)

        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        self.step_size = config.get('step_size', 2500)
        self.max_epoch = config.get('max_epoch', 200)
        self.task_level = config.get('task_level', 0)
        self.lape_dim = config.get('lape_dim', 200)

        self.use_vq_router = config.get('use_vq_router', False)
        self.use_delay_conv = config.get('use_delay_conv', False)
        self.use_time_aware_adp = config.get('use_time_aware_adp', True)
        self.use_balance_loss = config.get('use_balance_loss', False)
        self.lambda_balance = config.get('lambda_balance', 0.001)
        self.consistency_weight = config.get('consistency_weight', 0.1)

        if self.max_epoch * self.num_batches * self.world_size < self.step_size * self.output_window:
            self._logger.warning('Parameter `step_size` is too big with {} epochs and '
                                 'the model cannot be trained for all time steps.'.format(self.max_epoch))
        if self.use_curriculum_learning:
            self._logger.info('Use use_curriculum_learning!')

        if self.type_short_path == "dist":
            distances = sd_mx[~np.isinf(sd_mx)].flatten()
            std = distances.std()
            sd_mx = np.exp(-np.square(sd_mx / std))
            self.far_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.far_mask[sd_mx < self.far_mask_delta] = 1
            self.far_mask = self.far_mask.bool()
        else:
            sh_mx = sh_mx.T
            self.geo_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.geo_mask[sh_mx >= self.far_mask_delta] = 1
            self.geo_mask = self.geo_mask.bool()

        self.sem_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
        self.sem_mask[self.dtw_matrix >= self.dtw_delta] = 1
        self.sem_mask = self.sem_mask.bool()

        self.enc_embed_layer = DataEmbedding(
            self.feature_dim - self.ext_dim, self.embed_dim, lape_dim, self.adj_mx, drop=drop,
            add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week, device=self.device, num_nodes=self.num_nodes
        )

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]

        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, s_attn_size=self.s_attn_size, t_attn_size=self.t_attn_size,
                geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, t_num_heads=t_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=enc_dpr[i],
                act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), device=self.device, type_ln=type_ln,
                output_dim=self.output_dim, num_nodes=self.num_nodes,
                use_vq_router=self.use_vq_router, use_delay_conv=self.use_delay_conv,
                use_time_aware_adp=self.use_time_aware_adp,
                adj_init=self.adj_mx if self.use_vq_router else None
            ) for i in range(enc_depth)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

        self.end_conv1 = nn.Conv2d(
            in_channels=self.input_window, out_channels=self.output_window, kernel_size=1, bias=True,
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=self.skip_dim, out_channels=self.output_dim, kernel_size=1, bias=True,
        )

        tempp_path = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "cache", "dataset_cache", self.dataset, "tempp.npy"
        )
        tempp = np.load(tempp_path)
        tempp = torch.from_numpy(tempp).to(torch.float32)
        tempp = norm_embedding(tempp)
        self.tempp = self.cal_lape_emb(tempp).to(self.device)

        self.use_deep_trend = config.get('use_deep_trend', False)
        if self.use_deep_trend:
            feat_dim = self.feature_dim - self.ext_dim
            self.input_decomp = SeriesDecomposition(kernel_size=5)
            self.deep_trend_net = DeepTrendNet(feat_dim, self.input_window, self.output_window)
            self.trend_fusion = nn.Parameter(torch.tensor(1.0 if self.use_vq_router else 0.1))

    def cal_lape_emb(self, adj):
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        isolated_point_num = np.sum(np.where(d, 0, 1))
        self._logger.info(f"Number of isolated points: {isolated_point_num}")
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        L = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
        laplacian_pe = torch.from_numpy(EigVec[:, isolated_point_num + 1: self.lape_dim + isolated_point_num + 1]).float()
        laplacian_pe.require_grad = False
        return laplacian_pe

    def forward(self, batch, lap_mx=None):
        x = batch['X']
        ind = batch['ind']

        tempp = self.tempp
        enc = self.enc_embed_layer(x, lap_mx, tempp)
        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            enc = encoder_block(enc, ind, self.geo_mask, self.sem_mask)
            skip += self.skip_convs[i](enc.permute(0, 3, 2, 1))

        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        main_pred = skip.permute(0, 3, 2, 1)

        if self.use_deep_trend:
            feat_dim = self.feature_dim - self.ext_dim
            x_flow = x[..., :feat_dim]
            _, x_trend = self.input_decomp(x_flow)
            trend_pred = self.deep_trend_net(x_trend)
            return main_pred + self.trend_fusion * trend_pred
        else:
            return main_pred

    def get_loss_func(self, set_loss):
        if set_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile', 'masked_mae',
                                    'masked_mse', 'masked_rmse', 'masked_mape', 'masked_huber', 'r2', 'evar']:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        if set_loss.lower() == 'mae':
            lf = loss.masked_mae_torch
        elif set_loss.lower() == 'mse':
            lf = loss.masked_mse_torch
        elif set_loss.lower() == 'rmse':
            lf = loss.masked_rmse_torch
        elif set_loss.lower() == 'mape':
            lf = loss.masked_mape_torch
        elif set_loss.lower() == 'logcosh':
            lf = loss.log_cosh_loss
        elif set_loss.lower() == 'huber':
            lf = partial(loss.huber_loss, delta=self.huber_delta)
        elif set_loss.lower() == 'quantile':
            lf = partial(loss.quantile_loss, delta=self.quan_delta)
        elif set_loss.lower() == 'masked_mae':
            lf = partial(loss.masked_mae_torch, null_val=0)
        elif set_loss.lower() == 'masked_mse':
            lf = partial(loss.masked_mse_torch, null_val=0)
        elif set_loss.lower() == 'masked_rmse':
            lf = partial(loss.masked_rmse_torch, null_val=0)
        elif set_loss.lower() == 'masked_mape':
            lf = partial(loss.masked_mape_torch, null_val=0)
        elif set_loss.lower() == 'masked_huber':
            lf = partial(loss.masked_huber_loss, delta=self.huber_delta, null_val=0)
        elif set_loss.lower() == 'r2':
            lf = loss.r2_score_torch
        elif set_loss.lower() == 'evar':
            lf = loss.explained_variance_score_torch
        else:
            lf = loss.masked_mae_torch
        return lf

    def calculate_loss_without_predict(self, y_true, y_predicted, batches_seen=None, set_loss='masked_mae'):
        lf = self.get_loss_func(set_loss=set_loss)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        total_consistency_loss = 0.0
        if self.use_vq_router and self.training:
            for block in self.encoder_blocks:
                total_consistency_loss += block.st_attn.vq_router.consistency_loss
            total_consistency_loss /= len(self.encoder_blocks)

        if self.training:
            if batches_seen % self.step_size == 0 and self.task_level < self.output_window:
                self.task_level += 1
                self._logger.info('Training: task_level increase from {} to {}'.format(
                    self.task_level - 1, self.task_level))
            if self.use_curriculum_learning:
                pred_loss = lf(y_predicted[:, :self.task_level, :, :], y_true[:, :self.task_level, :, :])
            else:
                pred_loss = lf(y_predicted, y_true)
        else:
            pred_loss = lf(y_predicted, y_true)

        total_loss = pred_loss + self.consistency_weight * total_consistency_loss

        if self.use_balance_loss and self.use_vq_router and self.training:
            all_probs = []
            for block in self.encoder_blocks:
                probs = block.st_attn.vq_router.routing_probs  # [B, T, K]
                all_probs.append(probs)
            all_probs = torch.cat(all_probs, dim=0)  # [layers*B, T, K]
            p = all_probs.mean(dim=(0, 1))  # [K] per-template usage distribution
            L_balance = (p * torch.log(p + 1e-10)).sum()
            total_loss = total_loss + self.lambda_balance * L_balance

            if batches_seen is not None and batches_seen % 5000 == 0:
                H_soft = -(p * torch.log(p + 1e-10)).sum()
                H_soft_norm = H_soft / math.log(p.shape[0])

                # Hard routing stats (from actual Gumbel-softmax hard output)
                lines = [
                    f'[Routing] batch {batches_seen}:',
                    f'  soft H={H_soft_norm.item():.4f} top-3={p.topk(3).indices.tolist()}',
                ]
                for layer_idx, block in enumerate(self.encoder_blocks):
                    hw = block.st_attn.last_router_weights  # [B, T, K] one-hot
                    hw_flat = hw.reshape(-1, hw.shape[-1])  # [B*T, K]
                    h_p = hw_flat.mean(dim=0)  # [K] hard usage frequency
                    H_hard = -(h_p * torch.log(h_p + 1e-10)).sum()
                    H_hard_norm = H_hard / math.log(h_p.shape[0])
                    top_ids = h_p.topk(5).indices.tolist()
                    top_vals = [f'{h_p[i].item():.3f}' for i in top_ids]
                    lines.append(
                        f'  L{layer_idx} hard H={H_hard_norm.item():.3f} '
                        f'top-5={list(zip(top_ids, top_vals))}'
                    )
                self._logger.info(''.join(['\n' + l for l in lines]))

        return total_loss

    def calculate_loss(self, batch, batches_seen=None, lap_mx=None):
        y_true = batch['y']
        y_predicted = self.predict(batch, lap_mx)
        return self.calculate_loss_without_predict(y_true, y_predicted, batches_seen)

    def predict(self, batch, lap_mx=None):
        return self.forward(batch, lap_mx)
