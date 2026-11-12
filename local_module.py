import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LocalModule(nn.Module):

    def __init__(self, seq_len, input_dim, node_only_readout=False, n_layers=1, num_heads=8, hidden_dim=64, dropout_rate=0.3, attention_dropout_rate=0, max_time_days=100, temporal_bias='gaussian'):
        super().__init__()
        self.seq_len = seq_len
        self.node_only_readout = node_only_readout
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim
        self.num_heads = num_heads
        self.temporal_bias = temporal_bias
        self.max_time_days = max(float(max_time_days), 1.0)
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)
        if self.temporal_bias == 'gaussian':
            self.num_kernels = num_heads
            self.kernel_mu = nn.Parameter(torch.linspace(0, max_time_days, self.num_kernels))
            self.kernel_sigma = nn.Parameter(torch.ones(self.num_kernels) * (max_time_days / self.num_kernels / 2))
            self.kernel_proj = nn.Linear(self.num_kernels, num_heads)
        elif self.temporal_bias == 'linear':
            self.linear_time_weight = nn.Parameter(torch.zeros(self.num_heads))
            self.linear_time_bias = nn.Parameter(torch.zeros(self.num_heads))
        elif self.temporal_bias == 'embedding':
            self.num_time_bins = 64
            self.time_embedding = nn.Embedding(self.num_time_bins, self.num_heads)
        else:
            raise ValueError(f'Unknown temporal_bias: {self.temporal_bias}')
        encoders = [EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads) for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.attn_layer = nn.Linear(2 * hidden_dim, 1)

    def reset_parameters(self):
        self.att_embeddings_nope.reset_parameters()
        self.attn_layer.reset_parameters()
        self.final_ln.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        with torch.no_grad():
            if self.temporal_bias == 'gaussian':
                nn.init.constant_(self.kernel_sigma, 10.0)
                self.kernel_proj.reset_parameters()
            elif self.temporal_bias == 'linear':
                nn.init.zeros_(self.linear_time_weight)
                nn.init.zeros_(self.linear_time_bias)
            else:
                self.time_embedding.reset_parameters()

    def forward(self, batched_data, neighbor_times=None, pretrain_token=False):
        tensor = self.att_embeddings_nope(batched_data)
        attn_bias = None
        if neighbor_times is not None:
            time_diff = neighbor_times.unsqueeze(2) - neighbor_times.unsqueeze(1)
            time_diff = torch.abs(time_diff).unsqueeze(-1)
            if self.temporal_bias == 'linear':
                bias = time_diff.squeeze(-1).unsqueeze(-1) * self.linear_time_weight.view(1, 1, 1, -1) + self.linear_time_bias.view(1, 1, 1, -1)
            elif self.temporal_bias == 'embedding':
                bucket = torch.clamp((time_diff.squeeze(-1) / self.max_time_days * (self.num_time_bins - 1)).long(), min=0, max=self.num_time_bins - 1)
                bias = self.time_embedding(bucket)
            else:
                mu = self.kernel_mu.view(1, 1, 1, -1)
                sigma = torch.abs(self.kernel_sigma.view(1, 1, 1, -1)) + 1e-05
                gaussian_feat = torch.exp(-(time_diff - mu) ** 2 / sigma ** 2)
                bias = self.kernel_proj(gaussian_feat)
            attn_bias = bias.permute(0, 3, 1, 2)
        for enc_layer in self.layers:
            tensor = enc_layer(tensor, attn_bias=attn_bias)
        output = self.final_ln(tensor)
        if pretrain_token:
            return output
        _target = output[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        split_tensor = torch.split(output, [1, self.seq_len - 1], dim=1)
        node_tensor = split_tensor[0]
        _neighbor_tensor = split_tensor[1]
        if self.node_only_readout:
            indices = torch.arange(1, self.seq_len, 1)
            neighbor_tensor = _neighbor_tensor[:, indices]
            target = _target[:, indices]
        else:
            target = _target
            neighbor_tensor = _neighbor_tensor
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)
        neighbor_tensor = neighbor_tensor * layer_atten
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        output = (node_tensor + neighbor_tensor).squeeze()
        return output

class FeedForwardNetwork(nn.Module):

    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.bn_in = nn.BatchNorm1d(hidden_size)
        self.bn_out = nn.BatchNorm1d(hidden_size)
        self.ffn_net = nn.Sequential(nn.Linear(hidden_size, ffn_size), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(ffn_size, hidden_size), nn.Dropout(dropout_rate))

    def reset_parameters(self):
        for layer in self.ffn_net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn_in(x)
        x = x.permute(0, 2, 1)
        x = self.ffn_net(x)
        x = x.permute(0, 2, 1)
        x = self.bn_out(x)
        x = x.permute(0, 2, 1)
        return x

class EncoderLayer(nn.Module):

    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout_rate = attention_dropout_rate
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)

    def reset_parameters(self):
        self.self_attention_norm.reset_parameters()
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x, attn_bias=None):
        residual = x
        x_norm = self.self_attention_norm(x)
        Q = self.q_proj(x_norm)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)
        (B, L, D) = Q.shape
        head_dim = D // self.num_heads
        Q = Q.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_bias, dropout_p=self.attention_dropout_rate, is_causal=False)
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        attn_output = self.out_proj(attn_output)
        attn_output = self.self_attention_dropout(attn_output)
        x = residual + attn_output
        residual = x
        x_norm = self.ffn_norm(x)
        ffn_output = self.ffn(x_norm)
        x = residual + ffn_output
        return x
