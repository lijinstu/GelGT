import hashlib
import json
import os
import random
from collections import deque
from pathlib import Path
import h5py
import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import Dataset
from relbench.modeling.graph import get_node_train_table_input


class GloveTextEmbedding:
    def __init__(self, device):
        self.device = device
        self.model = None

    def __call__(self, sentences):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                'sentence-transformers/average_word_embeddings_glove.6B.300d',
                device=self.device)
        return torch.from_numpy(self.model.encode(sentences))


def graph_fingerprint(data):
    digest = hashlib.sha256()
    for node_type in data.node_types:
        digest.update(node_type.encode())
        digest.update(str(data[node_type].num_nodes).encode())
        if 'time' in data[node_type]:
            digest.update(memoryview(data[node_type].time.cpu().contiguous().numpy()))
    for edge_type in data.edge_types:
        digest.update(repr(edge_type).encode())
        digest.update(memoryview(data[edge_type].edge_index.cpu().contiguous().numpy()))
    return digest.hexdigest()


class TemporalSampler:
    def __init__(self, data, seed):
        self.data = data
        self.seed = seed
        self.adjacency = None

    def _build_adjacency(self):
        adjacency = {name: [set() for _ in range(self.data[name].num_nodes)]
                     for name in self.data.node_types}
        for source, relation, target in self.data.edge_types:
            edges = self.data[source, relation, target].edge_index
            for left, right in zip(edges[0].tolist(), edges[1].tolist()):
                adjacency[source][left].add((target, right))
                adjacency[target][right].add((source, left))
        self.adjacency = adjacency

    def _neighbors(self, key, query_time, limit):
        result = []
        for name, index in self.adjacency[key[0]][key[1]]:
            store = self.data[name]
            timestamp = float(store.time[index]) if 'time' in store else None
            if timestamp is None or timestamp < query_time:
                result.append((name, index, timestamp))
        result.sort(key=lambda item: (-(item[2] if item[2] is not None else -float('inf')),
                                      item[0], item[1]))
        return result[:limit]

    def sample(self, node_type, node_index, query_time, budget):
        if self.adjacency is None:
            self._build_adjacency()
        seed_key = (node_type, node_index)
        identity = json.dumps([node_type, node_index, query_time, budget, self.seed])
        rng = random.Random(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big'))
        first = [entry for entry in self._neighbors(seed_key, query_time, 5000)
                 if entry[:2] != seed_key]
        rng.shuffle(first)
        queue = deque((name, index, timestamp, 1) for name, index, timestamp in first)
        seen = {seed_key} | {(name, index) for name, index, _ in first}
        chosen = []
        while queue and len(chosen) < budget - 1:
            name, index, timestamp, hop = queue.popleft()
            chosen.append((name, index, hop,
                           (query_time - timestamp) / 86400 if timestamp is not None else 0.0))
            if hop == 1:
                for other_name, other_index, other_time in self._neighbors((name, index), query_time, 1000):
                    key = (other_name, other_index)
                    if key not in seen:
                        seen.add(key)
                        queue.append((other_name, other_index, other_time, 2))
        rng.shuffle(chosen)
        tokens = [(node_type, node_index, 0, 0.0)] + chosen
        local = {(name, index): position for position, (name, index, _, _) in enumerate(tokens)}
        edges = [(position, local[neighbor])
                 for position, (name, index, _, _) in enumerate(tokens)
                 for neighbor in sorted(self.adjacency[name][index]) if neighbor in local]
        edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2).T
        return tokens, edge_array


class RelGTTokens(Dataset):
    def __init__(self, data, task, budget, split, cache_dir, sampling_seed,
                 graph_hash, sampler=None):
        self.data = data
        self.budget = budget
        self.table = task.get_table(split)
        inputs = get_node_train_table_input(self.table, task)
        self.node_type, self.node_indices = inputs.nodes
        self.targets = inputs.target
        self.times = inputs.time
        self.node_types = list(data.node_types)
        self.node_type_to_index = {name: index for index, name in enumerate(self.node_types)}
        self.max_neighbor_hop = 3
        digest = hashlib.sha256()
        digest.update(json.dumps(['sampling-v2', graph_hash, self.node_type,
                                 budget, sampling_seed]).encode())
        digest.update(self.node_indices.cpu().contiguous().numpy().tobytes())
        digest.update(self.times.cpu().contiguous().numpy().tobytes())
        self.fingerprint = digest.hexdigest()
        self.path = Path(cache_dir) / 'sampling-v2' / f'{split}-{self.fingerprint}.h5'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.path) + '.lock'):
            if not self._cache_valid():
                if sampler is None:
                    raise RuntimeError('Sampling cache is missing or incompatible')
                self._build_cache(sampler)

    def _cache_valid(self):
        if not self.path.exists():
            return False
        try:
            with h5py.File(self.path, 'r') as file:
                return (file.attrs.get('complete', False)
                        and file.attrs.get('fingerprint', '') == self.fingerprint
                        and file['types'].shape == (len(self), self.budget)
                        and len(file['edge_offsets']) == len(self) + 1)
        except (OSError, KeyError):
            return False

    def _build_cache(self, sampler):
        temporary = self.path.with_suffix(f'.{os.getpid()}.tmp')
        try:
            with h5py.File(temporary, 'w') as file:
                count = len(self)
                columns = {}
                for name, dtype in [('types', 'i4'), ('indices', 'i8'), ('hops', 'i1'), ('times', 'f4')]:
                    columns[name] = file.create_dataset(name, shape=(count, self.budget), dtype=dtype)
                edges = file.create_dataset('edges', shape=(2, 0), maxshape=(2, None),
                                             dtype='i8', chunks=(2, 8192))
                offsets = file.create_dataset('edge_offsets', shape=(count + 1,), dtype='i8')
                offsets[0] = 0
                edge_count = 0
                for row in range(count):
                    tokens, edge_array = sampler.sample(self.node_type,
                        int(self.node_indices[row]), float(self.times[row]), self.budget)
                    values = {name: np.zeros(self.budget, dtype=columns[name].dtype) for name in columns}
                    for position, (name, index, hop, relative_time) in enumerate(tokens):
                        values['types'][position] = self.node_type_to_index[name]
                        values['indices'][position] = index
                        values['hops'][position] = hop
                        values['times'][position] = relative_time
                    for name, array in values.items():
                        columns[name][row] = array
                    next_count = edge_count + edge_array.shape[1]
                    if next_count > edge_count:
                        edges.resize((2, next_count))
                        edges[:, edge_count:next_count] = edge_array
                    offsets[row + 1] = next_count
                    edge_count = next_count
                file.attrs['fingerprint'] = self.fingerprint
                file.attrs['complete'] = True
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def __len__(self):
        return len(self.node_indices)

    def __getitem__(self, row):
        with h5py.File(self.path, 'r') as file:
            item = {name: torch.from_numpy(file[name][row]).long()
                    for name in ('types', 'indices', 'hops')}
            item['times'] = torch.from_numpy(file['times'][row])
            start, end = file['edge_offsets'][row:row + 2]
            item['edge_index'] = torch.from_numpy(file['edges'][:, start:end]).long()
        item['row'] = row
        item['label'] = self.targets[row] if self.targets is not None else None
        return item

    def collate(self, items):
        types, indices, hops, times = [torch.stack([item[name] for item in items])
                                      for name in ('types', 'indices', 'hops', 'times')]
        valid = (hops == 1) | (hops == 2)
        valid[:, 0] = True
        grouped_frames, grouped_indices = {}, {}
        for type_index, name in enumerate(self.node_types):
            positions = ((types == type_index) & valid).flatten().nonzero(as_tuple=False).flatten()
            if positions.numel():
                grouped_frames[type_index] = self.data[name].tf[indices.flatten()[positions]]
                grouped_indices[type_index] = positions
        edges = torch.cat([item['edge_index'] + row * self.budget
                           for row, item in enumerate(items)], dim=1)
        labels = torch.stack([item['label'] for item in items]) if self.targets is not None else None
        return dict(neighbor_types=types, neighbor_indices=indices, neighbor_hops=hops,
                    neighbor_times=times, edge_index=edges, labels=labels,
                    grouped_tfs=grouped_frames, grouped_indices=grouped_indices,
                    global_idx=torch.tensor([item['row'] for item in items]))

    def get_max_time(self):
        result = 0.0
        with h5py.File(self.path, 'r') as file:
            for start in range(0, len(self), 10000):
                block = file['times'][start:start + 10000]
                if block.size:
                    result = max(result, float(block.max()))
        return result
