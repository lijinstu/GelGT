import torch
from torch import nn
from torch.nn import functional as F


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        if self.training and x.shape[0] < 2:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight,
                                self.bias, False, self.momentum, self.eps)
        return super().forward(x)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.bn_in = SafeBatchNorm1d(hidden_size)
        self.bn_out = SafeBatchNorm1d(hidden_size)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size), nn.GELU(),
            nn.Dropout(dropout_rate), nn.Linear(2 * hidden_size, hidden_size),
            nn.Dropout(dropout_rate))

    def reset_parameters(self):
        self.bn_in.reset_parameters()
        self.bn_out.reset_parameters()
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, valid_mask):
        selected = self.bn_out(self.layers(self.bn_in(x[valid_mask])))
        output = torch.zeros_like(x)
        output[valid_mask] = selected
        return output


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout_rate, attention_dropout_rate, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dropout_rate = attention_dropout_rate
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, dropout_rate)

    def reset_parameters(self):
        self.self_attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.ffn.reset_parameters()

    def forward(self, x, attn_bias, valid_mask):
        batch_size, length, width = x.shape
        normalized = self.self_attention_norm(x)
        head_width = width // self.num_heads
        q, k, v = [layer(normalized).view(batch_size, length, self.num_heads,
                                         head_width).transpose(1, 2)
                   for layer in (self.q_proj, self.k_proj, self.v_proj)]
        bias = attn_bias.masked_fill(~valid_mask[:, None, None, :], -torch.inf)
        attention = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.attention_dropout_rate if self.training else 0.0,
            is_causal=False)
        attention = attention.transpose(1, 2).reshape(batch_size, length, width)
        x = x + self.self_attention_dropout(self.out_proj(attention))
        x = x.masked_fill(~valid_mask[..., None], 0)
        return x + self.ffn(self.ffn_norm(x), valid_mask)


class LocalModule(nn.Module):
    def __init__(self, input_dim, n_layers=1, num_heads=4, hidden_dim=512,
                 dropout_rate=0.1, attention_dropout_rate=0.1, max_time_days=365):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError('hidden_dim must be divisible by num_heads')
        self.att_embeddings_nope = nn.Linear(input_dim, hidden_dim)
        self.kernel_mu = nn.Parameter(torch.linspace(0, max_time_days, num_heads))
        self.kernel_sigma = nn.Parameter(torch.full((num_heads,), 10.0))
        self.kernel_proj = nn.Linear(num_heads, num_heads)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, dropout_rate, attention_dropout_rate, num_heads)
            for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.attn_layer = nn.Linear(2 * hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.att_embeddings_nope.reset_parameters()
        self.kernel_proj.reset_parameters()
        with torch.no_grad():
            self.kernel_sigma.fill_(10.0)
        self.final_ln.reset_parameters()
        self.attn_layer.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, batched_data, neighbor_times, valid_mask):
        tensor = self.att_embeddings_nope(batched_data)
        tensor = tensor.masked_fill(~valid_mask[..., None], 0)
        times = neighbor_times.masked_fill(~valid_mask, 0)
        delta = (times[:, :, None] - times[:, None, :]).abs()[..., None]
        sigma = self.kernel_sigma.abs() + 1e-5
        features = torch.exp(-(delta - self.kernel_mu).square() / sigma.square())
        bias = self.kernel_proj(features).permute(0, 3, 1, 2)
        for layer in self.layers:
            tensor = layer(tensor, bias, valid_mask)
        output = self.final_ln(tensor).masked_fill(~valid_mask[..., None], 0)
        seed, neighbors = output[:, :1], output[:, 1:]
        target = seed.expand(-1, neighbors.shape[1], -1)
        scores = self.attn_layer(torch.cat((target, neighbors), dim=-1))
        mask = valid_mask[:, 1:, None]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=1) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return seed[:, 0] + (neighbors * weights).sum(dim=1)
