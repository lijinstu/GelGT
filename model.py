import torch
from torch import nn
from torch_geometric.nn import MLP, SAGEConv
from torch_geometric.nn.norm import LayerNorm
from encoders import (GNNPEEncoder, NeighborHopEncoder, NeighborNodeTypeEncoder,
                      NeighborTfsEncoder, NeighborTimeEncoder)
from local_module import LocalModule


class GNNStack(nn.Module):
    def __init__(self, channels, dropout=0.1, edge_dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(channels, channels) for _ in range(3)])
        self.norms = nn.ModuleList([LayerNorm(channels, mode='node') for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
        self.edge_dropout = edge_dropout
        self.act = nn.GELU()

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            edges = edge_index
            if self.training and self.edge_dropout > 0:
                keep = torch.rand(edges.shape[1], device=x.device) > self.edge_dropout
                edges = edges[:, keep]
            x = x + self.dropout(self.act(norm(conv(x, edges))))
        return x


class GelGT(nn.Module):
    def __init__(self, max_neighbor_hop, node_type_map, col_names_dict, col_stats_dict,
                 local_num_layers=1, channels=512, out_channels=1, heads=4,
                 ff_dropout=0.1, attn_dropout=0.1, sample_node_len=300,
                 max_time_days=365, edge_dropout=0.2):
        super().__init__()
        self.sample_node_len = sample_node_len
        self.type_count = len(node_type_map)
        self.type_encoder = NeighborNodeTypeEncoder(node_type_map, channels)
        self.hop_encoder = NeighborHopEncoder(max_neighbor_hop, channels)
        self.time_encoder = NeighborTimeEncoder(channels)
        self.tfs_encoder = NeighborTfsEncoder(channels, node_type_map, col_names_dict,
                                             col_stats_dict)
        self.pe_encoder = GNNPEEncoder(channels)
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(5)])
        self.in_mixture = nn.Sequential(nn.Linear(5 * channels, 2 * channels),
                                        nn.ReLU(), nn.Linear(2 * channels, channels))
        self.local_module = LocalModule(channels, n_layers=local_num_layers,
                                        num_heads=heads, hidden_dim=channels,
                                        dropout_rate=ff_dropout,
                                        attention_dropout_rate=attn_dropout,
                                        max_time_days=max_time_days)
        self.layer_norm_local = nn.LayerNorm(channels)
        self.gnn_module = GNNStack(channels, ff_dropout, edge_dropout)
        self.layer_norm_gnn = nn.LayerNorm(channels)
        self.head = MLP(channels, hidden_channels=channels, out_channels=out_channels,
                        num_layers=2, norm_kwargs={'allow_single_element': True})
        self.weight = nn.Parameter(torch.tensor(0.2))

    def filter_neighbors(self, x, times, hops, edge_index, valid_mask):
        batch_size, old_length, channels = x.shape
        keep_count = min(self.sample_node_len, old_length)
        scores = (x * x[:, :1]).sum(-1).masked_fill(~valid_mask, -torch.inf)
        hop_order = torch.where(hops == 1, 2, torch.where(hops == 2, 1, 0))
        hop_order[:, 0] = 3
        by_score = scores.argsort(dim=1, descending=True, stable=True)
        ordered_hops = hop_order.gather(1, by_score)
        priority = ordered_hops.argsort(dim=1, descending=True, stable=True)
        selected = by_score.gather(1, priority)[:, :keep_count].sort(dim=1).values
        selected_mask = valid_mask.gather(1, selected)
        selected_x = x.gather(1, selected[..., None].expand(-1, -1, channels))
        selected_times = times.gather(1, selected)
        old_positions = selected + torch.arange(batch_size, device=x.device)[:, None] * old_length
        mapping = torch.full((batch_size * old_length,), -1, dtype=torch.long, device=x.device)
        mapping[old_positions.flatten()] = torch.arange(batch_size * keep_count, device=x.device)
        mapped = mapping[edge_index]
        mapped = mapped[:, (mapped >= 0).all(dim=0)]
        flat_valid = selected_mask.flatten()
        mapped = mapped[:, flat_valid[mapped].all(dim=0)]
        return selected_x, selected_times, mapped, selected_mask

    def forward(self, neighbor_types, neighbor_indices, neighbor_hops, neighbor_times,
                grouped_tf_dict, edge_index):
        valid = (neighbor_hops == 1) | (neighbor_hops == 2)
        valid[:, 0] = True
        types = neighbor_types.masked_fill(~valid, 0)
        indices = neighbor_indices.masked_fill(~valid, 0)
        hops = neighbor_hops.masked_fill(~valid, 0)
        times = neighbor_times.masked_fill(~valid, 0)
        node_keys = indices * self.type_count + types
        parts = (self.type_encoder(types), self.hop_encoder(hops),
                 self.time_encoder(times), self.tfs_encoder(grouped_tf_dict, types),
                 self.pe_encoder(edge_index, valid, node_keys))
        normalized = [norm(value).masked_fill(~valid[..., None], 0)
                      for norm, value in zip(self.norms, parts)]
        x = self.in_mixture(torch.cat(normalized, dim=-1)).masked_fill(~valid[..., None], 0)
        x, times, edges, valid = self.filter_neighbors(x, times, hops, edge_index, valid)
        local = self.layer_norm_local(self.local_module(x, times, valid))
        batch_size, length, channels = x.shape
        gnn = self.gnn_module(x.reshape(-1, channels), edges)
        gnn = self.layer_norm_gnn(gnn.view(batch_size, length, channels)[:, 0])
        weight = self.weight.sigmoid()
        return self.head(weight * gnn + (1 - weight) * local)
