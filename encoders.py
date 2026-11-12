from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_frame
import torch_geometric.transforms as T
from torch_geometric.data import Data

class NeighborNodeTypeEncoder(nn.Module):
    pass

    def __init__(self, node_type_map, embedding_dim):
        pass
        super(NeighborNodeTypeEncoder, self).__init__()
        num_types = max(node_type_map.values()) + 1
        self.embedding = nn.Embedding(num_embeddings=num_types + 1, embedding_dim=embedding_dim)

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, type_indices):
        pass
        return self.embedding(type_indices)

class NeighborHopEncoder(nn.Module):
    pass

    def __init__(self, max_neighbor_hop, embedding_dim):
        pass
        super(NeighborHopEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=max_neighbor_hop + 2, embedding_dim=embedding_dim)

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, hop_distances):
        pass
        shifted = hop_distances + 1
        return self.embedding(shifted)
from torch_geometric.nn import PositionalEncoding

class NeighborTimeEncoder(nn.Module):
    pass

    def __init__(self, embedding_dim):
        pass
        super(NeighborTimeEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.mask_vector = nn.Parameter(torch.zeros(embedding_dim))

    def reset_parameters(self):
        self.linear.reset_parameters()
        nn.init.normal_(self.mask_vector, mean=0.0, std=0.02)

    def forward(self, rel_time):
        pass
        (B, K) = rel_time.shape
        flattened_time = rel_time.view(-1)
        pos_encoded = self.pos_encoder(flattened_time)
        linear_out = self.linear(pos_encoded)
        linear_out = linear_out.view(B, K, -1)
        mask = (rel_time < 0).unsqueeze(-1).float()
        mask_vector = self.mask_vector.unsqueeze(0).unsqueeze(0).expand(B, K, -1)
        out = (1 - mask) * linear_out + mask * mask_vector
        return out
from torch_frame.nn.models import ResNet
from typing import Dict, Any

class NeighborTfsEncoder(nn.Module):
    pass

    def __init__(self, channels: int, node_type_map, col_names_dict, col_stats_dict, torch_frame_model_cls=ResNet, torch_frame_model_kwargs: Dict[str, Any]={'channels': 128, 'num_layers': 4}, default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any]={torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}), torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}), torch_frame.multicategorical: (torch_frame.nn.MultiCategoricalEmbeddingEncoder, {}), torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}), torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {})}):
        pass
        super(NeighborTfsEncoder, self).__init__()
        self.node_type_map = node_type_map
        self.inv_node_type_map = {idx: nt for (nt, idx) in node_type_map.items()}
        self.encoders = nn.ModuleDict()
        self.channels = channels
        for (node_type, stype_dict) in col_names_dict.items():
            stype_encoder_dict = {stype: default_stype_encoder_cls_kwargs[stype][0](**default_stype_encoder_cls_kwargs[stype][1]) for stype in stype_dict.keys() if stype in default_stype_encoder_cls_kwargs}
            self.encoders[node_type] = torch_frame_model_cls(**torch_frame_model_kwargs, out_channels=channels, col_stats=col_stats_dict[node_type], col_names_dict=stype_dict, stype_encoder_dict=stype_encoder_dict)

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(self, batch_dict, neighbor_types):
        pass
        grouped_tfs = batch_dict['grouped_tfs']
        grouped_indices = batch_dict['grouped_indices']
        flat_batch_idx = batch_dict['flat_batch_idx']
        flat_nbr_idx = batch_dict['flat_nbr_idx']
        (B, K) = neighbor_types.shape
        N = len(flat_batch_idx)
        device = neighbor_types.device
        encoded_flat_tensor = torch.zeros((N, self.channels), device=device)
        for (t_int, big_tf) in grouped_tfs.items():
            node_type_str = self.inv_node_type_map[t_int]
            encoder = self.encoders[node_type_str]
            big_tf = big_tf.to(device=device)
            for (stype, tensor) in big_tf.feat_dict.items():
                if isinstance(tensor, torch.Tensor):
                    big_tf.feat_dict[stype] = torch.nan_to_num(tensor, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
            out_t = encoder(big_tf)
            if out_t.dim() == 3 and out_t.shape[1] == 1:
                out_t = out_t.squeeze(1)
            idx_list = grouped_indices[t_int]
            idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
            encoded_flat_tensor[idx_tensor] = out_t
        output = torch.zeros((B, K, self.channels), device=device)
        indices_i = torch.tensor(flat_batch_idx, dtype=torch.long, device=device)
        indices_j = torch.tensor(flat_nbr_idx, dtype=torch.long, device=device)
        output[indices_i, indices_j] = encoded_flat_tensor
        return output
from torch_geometric.nn import GINConv

class GNNPEEncoder(nn.Module):
    pass

    def __init__(self, embedding_dim: int, num_layers: int=4, pooling: str='none', pe_dim: int=0):
        super().__init__()
        self.pooling = pooling.lower()
        self.num_layers = num_layers
        self.layer_embedding_dim = embedding_dim // 4
        self.pe_dim = pe_dim
        if self.pe_dim > 0:
            self.input_proj = nn.Linear(self.pe_dim, self.layer_embedding_dim)
        else:
            self.input_proj = nn.Linear(1, self.layer_embedding_dim)
        self.conv = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(self.layer_embedding_dim, self.layer_embedding_dim * 2), nn.BatchNorm1d(self.layer_embedding_dim * 2), nn.ReLU(), nn.Linear(self.layer_embedding_dim * 2, self.layer_embedding_dim))
            self.conv.append(GINConv(mlp, train_eps=True))
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(self.layer_embedding_dim))
        if self.pooling == 'cat':
            final_input_dim = self.layer_embedding_dim * num_layers
        elif self.pooling in ['none', 'mean', 'max']:
            final_input_dim = self.layer_embedding_dim
        else:
            raise ValueError("Invalid pooling method. Choose from 'none', 'cat', 'mean', 'max'.")
        self.final_transform = nn.Linear(final_input_dim, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        for conv in self.conv:
            for layer in conv.nn:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        nn.init.xavier_uniform_(self.final_transform.weight)
        if self.final_transform.bias is not None:
            nn.init.zeros_(self.final_transform.bias)

    def forward(self, edge_index, batch):
        pass
        device = edge_index.device
        total_nodes = batch.size(0)
        if self.pe_dim > 0:
            data = Data(edge_index=edge_index, num_nodes=total_nodes)
            transform = T.AddLaplacianEigenvectorPE(k=self.pe_dim)
            data = transform(data)
            x_input = data.laplacian_eigenvector_pe.to(device)
        else:
            x_input = torch.randn(total_nodes, 1, device=device)
        x = self.input_proj(x_input)
        outputs = []
        for (i, conv) in enumerate(self.conv):
            x_res = x
            x_new = conv(x, edge_index)
            x_new = self.bns[i](x_new)
            x_new = F.relu(x_new)
            x = x_new + x_res
            outputs.append(x)
        if self.pooling == 'none':
            x_final = outputs[-1]
        elif self.pooling == 'cat':
            x_final = torch.cat(outputs, dim=-1)
        elif self.pooling == 'mean':
            outputs_tensor = torch.stack(outputs, dim=-1)
            x_final = torch.mean(outputs_tensor, dim=-1)
        elif self.pooling == 'max':
            outputs_tensor = torch.stack(outputs, dim=-1)
            x_final = torch.max(outputs_tensor, dim=-1)[0]
        x = self.final_transform(x_final)
        B = batch.max().item() + 1
        K = total_nodes // B
        out = x.view(B, K, -1)
        return out
