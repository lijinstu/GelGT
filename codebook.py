from re import X
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    pass

    def __init__(self, num_embeddings, embedding_dim, decay=0.99):
        super(VectorQuantizerEMA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self.register_buffer('_embedding', torch.randn(self._num_embeddings, self._embedding_dim))
        self.register_buffer('_embedding_output', torch.randn(self._num_embeddings, self._embedding_dim))
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.randn(self._num_embeddings, self._embedding_dim))
        self._decay = decay
        self.bn = torch.nn.BatchNorm1d(self._embedding_dim, affine=False)

    def reset_parameters(self):
        nn.init.normal_(self._embedding, mean=0.0, std=1.0)
        nn.init.normal_(self._embedding_output, mean=0.0, std=1.0)
        nn.init.zeros_(self._ema_cluster_size)
        nn.init.normal_(self._ema_w, mean=0.0, std=1.0)
        self.bn.reset_parameters()

    def get_k(self):
        pass
        return self._embedding_output

    def get_v(self):
        pass
        return self._embedding_output[:, :self._embedding_dim]

    def update(self, x):
        inputs_normalized = self.bn(x)
        embedding_normalized = self._embedding
        distances = torch.sum(inputs_normalized ** 2, dim=1, keepdim=True) + torch.sum(embedding_normalized ** 2, dim=1) - 2 * torch.matmul(inputs_normalized, embedding_normalized.t())
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        if self.training:
            self._ema_cluster_size.data = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size.data = (self._ema_cluster_size + 1e-05) / (n + self._num_embeddings * 1e-05) * n
            dw = torch.matmul(encodings.t(), inputs_normalized)
            self._ema_w.data = self._ema_w * self._decay + (1 - self._decay) * dw
            self._embedding.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)
            running_std = torch.sqrt(self.bn.running_var + 1e-05).unsqueeze(dim=0)
            running_mean = self.bn.running_mean.unsqueeze(dim=0)
            self._embedding_output.data = self._embedding * running_std + running_mean
        return encoding_indices
