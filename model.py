import math
import os
import torch
_UNPATCHED_TOPK = os.environ.get('GELGT_UNPATCHED_TOPK', '0') == '1'
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MLP
from codebook import VectorQuantizerEMA
from einops import rearrange
from local_module import LocalModule
from torch_frame.data.stats import StatType
from typing import Dict, Any, List
from encoders import NeighborNodeTypeEncoder, NeighborHopEncoder, NeighborTimeEncoder, NeighborTfsEncoder, GNNPEEncoder
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.norm import LayerNorm

class GNNStack(nn.Module):
    pass

    def __init__(self, channels, dropout=0.0, edge_dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(3):
            self.convs.append(SAGEConv(channels, channels))
            self.norms.append(LayerNorm(channels, mode='node'))
        self.dropout = nn.Dropout(dropout)
        self.edge_dropout = edge_dropout
        self.act = nn.GELU()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_index):
        for (i, (conv, norm)) in enumerate(zip(self.convs, self.norms)):
            x_in = x
            if self.training and self.edge_dropout > 0 and (edge_index.size(1) > 0):
                mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.edge_dropout
                eidx = edge_index[:, mask]
            else:
                eidx = edge_index
            x = conv(x, eidx)
            x = norm(x)
            x = self.act(x)
            x = self.dropout(x)
            x = x + x_in
        return x

class RelGTLayer(nn.Module):

    def __init__(self, in_channels, out_channels, local_num_layers, global_dim, num_nodes, heads=1, concat=True, ff_dropout=0.0, attn_dropout=0.0, edge_dim=None, conv_type='local', num_centroids=None, sample_node_len=100, max_time_days=100, edge_dropout=0.0, temporal_bias='gaussian', **kwargs):
        super(RelGTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.local_num_layers = local_num_layers
        self.heads = heads
        self.concat = concat
        self.ff_dropout = ff_dropout
        self.attn_dropout = attn_dropout
        self.edge_dim = edge_dim
        self.conv_type = conv_type
        self.num_centroids = num_centroids
        self._alpha = None
        self.sample_node_len = sample_node_len
        self.use_gnn = True
        self.local_module = LocalModule(seq_len=self.sample_node_len, input_dim=in_channels, n_layers=local_num_layers, num_heads=heads, hidden_dim=out_channels, dropout_rate=ff_dropout, attention_dropout_rate=attn_dropout, max_time_days=max_time_days, temporal_bias=temporal_bias)
        self.layer_norm_local = nn.LayerNorm(out_channels)
        if self.conv_type != 'local':
            self.vq = VectorQuantizerEMA(num_centroids, global_dim, decay=0.99)
            c = torch.randint(0, num_centroids, (num_nodes,), dtype=torch.long)
            self.register_buffer('c_idx', c)
            self.attn_fn = F.softmax
            attn_channels = out_channels // heads
            self.lin_proj_g = Linear(in_channels, global_dim)
            self.lin_key_g = Linear(global_dim, heads * attn_channels)
            self.lin_query_g = Linear(global_dim, heads * attn_channels)
            self.lin_value_g = Linear(global_dim, heads * attn_channels)
            self.layer_norm_global = nn.LayerNorm(out_channels)
        if self.use_gnn:
            self.gnn_module = GNNStack(out_channels, dropout=ff_dropout, edge_dropout=edge_dropout)
            self.layer_norm_gnn = nn.LayerNorm(out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_type != 'local':
            self.lin_proj_g.reset_parameters()
            self.lin_key_g.reset_parameters()
            self.lin_query_g.reset_parameters()
            self.lin_value_g.reset_parameters()
            if hasattr(self, 'vq'):
                self.vq.reset_parameters()
        if hasattr(self.local_module, 'reset_parameters'):
            self.local_module.reset_parameters()
        if self.use_gnn:
            self.gnn_module.reset_parameters()
            self.layer_norm_gnn.reset_parameters()

    def forward(self, x_set, x, node_indices, edge_index=None, neighbor_times=None, neighbor_types=None):
        if self.conv_type == 'local':
            out = self.local_forward(x_set, neighbor_times=neighbor_times)
            out = self.layer_norm_local(out)
        elif self.conv_type == 'global':
            out = self.global_forward(x, node_indices)
            out = self.layer_norm_global(out)
        elif self.conv_type == 'full':
            out_local = self.local_forward(x_set, neighbor_times=neighbor_times)
            out_local = self.layer_norm_local(out_local)
            if edge_index is None:
                raise ValueError('use_gnn is True but edge_index is None')
            out_gnn = self.gnn_forward(x_set, neighbor_types, edge_index)
            out_gnn = self.layer_norm_gnn(out_gnn)
        else:
            raise NotImplementedError
        return (out_gnn, out_local)

    def gnn_forward(self, x_set, neighbor_types, edge_index):
        (B, K, C) = x_set.shape
        device = x_set.device
        x_flat = x_set.reshape(B * K, C)
        x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
        x_gnn = self.gnn_module(x_flat, edge_index)
        seed_pos = torch.arange(0, B * K, K, device=device)
        return x_gnn[seed_pos]

    def global_forward(self, x, batch_idx):
        (d, h) = (self.out_channels, self.heads)
        scale = 1.0 / math.sqrt(d)
        q_x = self.lin_proj_g(x)
        k_buf = self.vq.get_k()
        k_x = k_buf.detach().clone()
        v_buf = self.vq.get_v()
        v_x = v_buf.detach().clone()
        q = self.lin_query_g(q_x)
        k = self.lin_key_g(k_x)
        v = self.lin_value_g(v_x)
        (q, k, v) = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=h), (q, k, v))
        dots = torch.einsum('h i d, h j d -> h i j', q, k) * scale
        (c, c_count) = self.c_idx.unique(return_counts=True)
        centroid_count = torch.zeros(self.num_centroids, dtype=torch.long).to(x.device)
        centroid_count[c.to(torch.long)] = c_count
        dots = dots + torch.log(centroid_count.view(1, 1, -1))
        attn = self.attn_fn(dots, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
        out = torch.einsum('h i j, h j d -> h i d', attn, v)
        out = rearrange(out, 'h n d -> n (h d)')
        if self.training:
            x_idx = self.vq.update(q_x)
            self.c_idx[batch_idx] = x_idx.squeeze().to(torch.long)
        return out

    def local_forward(self, x_set, neighbor_times=None, pretrain_token=False):
        return self.local_module(x_set, neighbor_times=neighbor_times, pretrain_token=pretrain_token)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads}, local_num_layers={self.local_num_layers})'

class RelGT(torch.nn.Module):

    def __init__(self, num_nodes: int, max_neighbor_hop: int, node_type_map: Dict[str, int], col_names_dict: Dict[str, Dict[str, List[str]]], col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]], local_num_layers: int, channels: int, out_channels: int, global_dim: int, heads: int=4, ff_dropout: float=0.0, attn_dropout: float=0.0, conv_type: str='full', ablate: str='none', gnn_pe_dim: int=0, num_centroids: int=4096, sample_node_len: int=100, max_time_days: int=100, edge_dropout: float=0.0, temporal_bias: str='gaussian', args: Any=None):
        super(RelGT, self).__init__()
        self.max_neighbor_hop = max_neighbor_hop
        self.node_type_map = node_type_map
        self.sample_node_len = sample_node_len
        self.type_encoder = NeighborNodeTypeEncoder(embedding_dim=channels, node_type_map=self.node_type_map)
        self.hop_encoder = NeighborHopEncoder(embedding_dim=channels, max_neighbor_hop=self.max_neighbor_hop)
        self.time_encoder = NeighborTimeEncoder(embedding_dim=channels)
        self.tfs_encoder = NeighborTfsEncoder(channels=channels, node_type_map=self.node_type_map, col_names_dict=col_names_dict, col_stats_dict=col_stats_dict)
        self.pe_encoder = GNNPEEncoder(embedding_dim=channels, pe_dim=gnn_pe_dim)
        self.layer_norm_type = nn.LayerNorm(channels)
        self.layer_norm_hop = nn.LayerNorm(channels)
        self.layer_norm_time = nn.LayerNorm(channels)
        self.layer_norm_tfs = nn.LayerNorm(channels)
        self.layer_norm_pe = nn.LayerNorm(channels)
        hidden_channels = channels
        ablate_key_dict = {'type': 0, 'hop': 1, 'time': 2, 'tfs': 3, 'gnn': 4}
        self.ablate_idx = ablate_key_dict.get(ablate, None)
        channel_mult = 5 if self.ablate_idx is None else 4
        self.in_mixture = nn.Sequential(nn.Linear(channel_mult * channels, 2 * channels), nn.ReLU(), nn.Linear(2 * channels, channels))
        self.convs = torch.nn.ModuleList()
        self.ffs = torch.nn.ModuleList()
        _overall_num_layers = 1
        for _ in range(_overall_num_layers):
            self.convs.append(RelGTLayer(in_channels=hidden_channels, out_channels=hidden_channels, local_num_layers=local_num_layers, global_dim=global_dim, num_nodes=num_nodes, heads=heads, ff_dropout=ff_dropout, attn_dropout=attn_dropout, conv_type=conv_type, num_centroids=num_centroids, sample_node_len=sample_node_len, max_time_days=max_time_days, edge_dropout=edge_dropout, temporal_bias=temporal_bias))
            h_times = 2 if conv_type == 'full' else 1
            self.ffs.append(nn.Sequential(nn.BatchNorm1d(hidden_channels * h_times), nn.Linear(h_times * hidden_channels, hidden_channels * 2), nn.GELU(), nn.Dropout(ff_dropout), nn.Linear(hidden_channels * 2, hidden_channels), nn.Dropout(ff_dropout), nn.BatchNorm1d(hidden_channels)))
        self.head = MLP(channels, hidden_channels=channels, out_channels=out_channels, num_layers=2)
        self.weight = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    def reset_parameters(self):
        self.type_encoder.reset_parameters()
        self.hop_encoder.reset_parameters()
        self.time_encoder.reset_parameters()
        self.tfs_encoder.reset_parameters()
        self.pe_encoder.reset_parameters()
        for layer in self.in_mixture:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for ff in self.ffs:
            if hasattr(ff, 'reset_parameters'):
                ff.reset_parameters()
        self.head.reset_parameters()

    def filter_neighbors(self, x_set, neighbor_types, neighbor_hops, neighbor_times, edge_index, target_k=300):
        (B, K_old, C) = x_set.shape
        device = x_set.device
        seed_emb = x_set[:, 0, :]
        sim_scores = torch.sum(x_set * seed_emb.unsqueeze(1), dim=-1)
        final_scores = sim_scores.clone()
        if _UNPATCHED_TOPK:
            mask_seed = neighbor_hops == 0
            mask_hop1 = neighbor_hops == 1
            mask_hop2 = neighbor_hops == 2
            final_scores[mask_seed] = 1000000000.0
            final_scores[mask_hop1] = 1000000000.0
            mask_others = ~(mask_seed | mask_hop1 | mask_hop2)
            final_scores[mask_others] = -1000000000.0
        else:
            mask_seed = torch.zeros_like(neighbor_hops, dtype=torch.bool)
            mask_seed[:, 0] = True
            mask_hop1 = neighbor_hops == 1
            mask_hop2 = neighbor_hops == 2
            mask_valid = mask_seed | mask_hop1 | mask_hop2
            final_scores[mask_seed] = 1000000000.0
            final_scores[mask_hop1] = 1000000000.0
            final_scores[~mask_valid] = -1000000000.0
        (_, sorted_indices) = torch.topk(final_scores, k=target_k, dim=1)
        (sorted_indices, _) = torch.sort(sorted_indices, dim=1)
        gather_index = sorted_indices.unsqueeze(-1).expand(-1, -1, C)
        new_x_set = torch.gather(x_set, 1, gather_index)
        new_neighbor_times = torch.gather(neighbor_times, 1, sorted_indices)
        new_neighbor_types = torch.gather(neighbor_types, 1, sorted_indices)
        batch_offset = (torch.arange(B, device=device) * K_old).unsqueeze(1)
        global_old_indices = (sorted_indices + batch_offset).view(-1)
        total_old_nodes = B * K_old
        mapping_table = torch.full((total_old_nodes,), -1, dtype=torch.long, device=device)
        new_global_indices = torch.arange(B * target_k, device=device)
        mapping_table[global_old_indices] = new_global_indices
        new_row = mapping_table[edge_index[0]]
        new_col = mapping_table[edge_index[1]]
        mask_edges = (new_row != -1) & (new_col != -1)
        new_edge_index = torch.stack([new_row[mask_edges], new_col[mask_edges]], dim=0)
        new_batch = torch.arange(B, device=device).repeat_interleave(target_k)
        return (new_x_set, new_neighbor_types, new_neighbor_times, new_edge_index, new_batch)

    def forward(self, neighbor_types, node_indices, neighbor_hops, neighbor_times, grouped_tf_dict, edge_index=None, batch=None):
        neighbor_tfs = self.layer_norm_tfs(self.tfs_encoder(grouped_tf_dict, neighbor_types))
        neighbor_types_emb = self.layer_norm_type(self.type_encoder(neighbor_types.long()))
        neighbor_hops_emb = self.layer_norm_hop(self.hop_encoder(neighbor_hops.long()))
        neighbor_times_emb = self.layer_norm_time(self.time_encoder(neighbor_times.float()))
        neighbor_subgraph_pe = self.layer_norm_pe(self.pe_encoder(edge_index, batch))
        cat_list = [neighbor_types_emb, neighbor_hops_emb, neighbor_times_emb, neighbor_tfs, neighbor_subgraph_pe]
        if self.ablate_idx is not None:
            cat_list.pop(self.ablate_idx)
        x_set = torch.cat(cat_list, dim=-1)
        x_set = self.in_mixture(x_set)
        target_k = self.sample_node_len
        (x_set, filtered_types, filtered_times, edge_index, batch) = self.filter_neighbors(x_set, neighbor_types, neighbor_hops, neighbor_times, edge_index, target_k=target_k)
        weight = torch.sigmoid(self.weight)
        x = x_set[:, 0, :]
        for (i, conv) in enumerate(self.convs):
            (out_gnn, out_local) = conv(x_set, x, node_indices, edge_index, neighbor_times=filtered_times, neighbor_types=filtered_types)
            x_set = weight * out_gnn + (1 - weight) * out_local
        x_set = self.head(x_set)
        return x_set

    def global_forward(self, x, pos_enc, node_indices):
        raise NotImplementedError
        x = self.fc_in(x)
        for (i, conv) in enumerate(self.convs):
            x = conv.global_forward(x, pos_enc, node_indices)
            x = self.ffs[i](x)
        x = self.fc_out(x)
        return x
