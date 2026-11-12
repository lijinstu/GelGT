import math
import torch
from torch import nn
from torch.nn import functional as F
import torch_frame
from torch_frame.nn.models import ResNet
from torch_geometric.nn import GINConv, PositionalEncoding
from local_module import SafeBatchNorm1d


class NeighborNodeTypeEncoder(nn.Module):
    def __init__(self, node_type_map, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(max(node_type_map.values()) + 2, embedding_dim)

    def forward(self, types):
        return self.embedding(types)


class NeighborHopEncoder(nn.Module):
    def __init__(self, max_neighbor_hop, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(max_neighbor_hop + 2, embedding_dim)

    def forward(self, hops):
        return self.embedding(hops + 1)


class NeighborTimeEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.mask_vector = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, times):
        output = self.linear(self.pos_encoder(times.reshape(-1)))
        output = output.view(*times.shape, -1)
        return torch.where((times < 0)[..., None], self.mask_vector, output)


class NeighborTfsEncoder(nn.Module):
    def __init__(self, channels, node_type_map, col_names_dict, col_stats_dict):
        super().__init__()
        self.type_names = {index: name for name, index in node_type_map.items()}
        self.channels = channels
        classes = {
            torch_frame.categorical: torch_frame.nn.EmbeddingEncoder,
            torch_frame.numerical: torch_frame.nn.LinearEncoder,
            torch_frame.multicategorical: torch_frame.nn.MultiCategoricalEmbeddingEncoder,
            torch_frame.embedding: torch_frame.nn.LinearEmbeddingEncoder,
            torch_frame.timestamp: torch_frame.nn.TimestampEncoder,
        }
        self.encoders = nn.ModuleDict()
        for name, columns in col_names_dict.items():
            encoders = {stype: classes[stype]() for stype in columns if stype in classes}
            self.encoders[name] = ResNet(channels=128, num_layers=4, out_channels=channels,
                                         col_stats=col_stats_dict[name],
                                         col_names_dict=columns, stype_encoder_dict=encoders)

    def forward(self, batch_dict, neighbor_types):
        batch_size, length = neighbor_types.shape
        output = torch.zeros(batch_size * length, self.channels, device=neighbor_types.device)
        for type_index, frame in batch_dict['grouped_tfs'].items():
            frame = frame.to(neighbor_types.device)
            for stype, values in frame.feat_dict.items():
                if values.is_floating_point():
                    frame.feat_dict[stype] = torch.nan_to_num(values, nan=0.0,
                                                            posinf=1e6, neginf=-1e6)
            values = self.encoders[self.type_names[type_index]](frame)
            positions = torch.as_tensor(batch_dict['grouped_indices'][type_index],
                                         dtype=torch.long, device=neighbor_types.device)
            output[positions] = values.reshape(-1, self.channels)
        return output.view(batch_size, length, self.channels)


class GNNPEEncoder(nn.Module):
    def __init__(self, embedding_dim, num_layers=4):
        super().__init__()
        width = max(1, embedding_dim // 4)
        self.input_proj = nn.Linear(1, width)
        self.convs = nn.ModuleList([GINConv(nn.Sequential(
            nn.Linear(width, 2 * width), SafeBatchNorm1d(2 * width), nn.ReLU(),
            nn.Linear(2 * width, width)), train_eps=True) for _ in range(num_layers)])
        self.norms = nn.ModuleList([SafeBatchNorm1d(width) for _ in range(num_layers)])
        self.final_transform = nn.Linear(width, embedding_dim)

    def forward(self, edge_index, valid_mask, node_keys):
        valid_positions = valid_mask.flatten().nonzero(as_tuple=False).flatten()
        mapping = torch.full((valid_mask.numel(),), -1, device=edge_index.device,
                              dtype=torch.long)
        mapping[valid_positions] = torch.arange(valid_positions.numel(), device=edge_index.device)
        edges = mapping[edge_index]
        edges = edges[:, (edges >= 0).all(dim=0)]
        keys = (node_keys.flatten()[valid_positions] + 1).remainder(2147483647)
        keys = (keys * 48271 + 17).remainder(2147483647)
        keys = (keys * 48271 + 17).remainder(2147483647)
        uniform = (keys.double() + 0.5) / 2147483647
        initial = (math.sqrt(2) * torch.erfinv((2 * uniform - 1).clamp(-0.999999, 0.999999))).float()
        x = self.input_proj(initial[:, None])
        for conv, norm in zip(self.convs, self.norms):
            x = x + F.relu(norm(conv(x, edges)))
        x = self.final_transform(x)
        output = x.new_zeros(valid_mask.numel(), x.shape[-1])
        output[valid_positions] = x
        return output.view(*valid_mask.shape, -1)
