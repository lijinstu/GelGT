import random
import os
_GELGT_CACHE_DIR = os.environ.setdefault('GELGT_CACHE_DIR', os.environ.get('GELGT_CACHE_DIR', os.path.expanduser('~/.cache/gelgt')))
os.environ.setdefault('XDG_CACHE_HOME', _GELGT_CACHE_DIR)
os.environ.setdefault('RELBENCH_CACHE', os.path.join(_GELGT_CACHE_DIR, 'relbench'))
os.environ.setdefault('HF_HOME', os.path.join(_GELGT_CACHE_DIR, 'huggingface'))
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(_GELGT_CACHE_DIR, 'huggingface', 'hub'))
os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME', os.path.join(_GELGT_CACHE_DIR, 'sentence_transformers'))
os.environ.setdefault('TORCH_HOME', os.path.join(_GELGT_CACHE_DIR, 'torch'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_GELGT_CACHE_DIR, 'matplotlib'))
os.environ.setdefault('WANDB_DIR', os.path.join(_GELGT_CACHE_DIR, 'wandb'))
for _p in [_GELGT_CACHE_DIR, os.environ['RELBENCH_CACHE'], os.environ['HF_HOME'], os.environ['HUGGINGFACE_HUB_CACHE'], os.environ['SENTENCE_TRANSFORMERS_HOME'], os.environ['TORCH_HOME'], os.environ['MPLCONFIGDIR'], os.environ['WANDB_DIR']]:
    os.makedirs(_p, exist_ok=True)
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import List, Optional, Tuple, Dict
import numpy as np
import h5py
import gc
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from relbench.base import Dataset, EntityTask
from relbench.modeling.graph import get_node_train_table_input
from collections import defaultdict, deque
GLOBAL_ADJ = None
GLOBAL_ALL_NODES = None

class GloveTextEmbedding:

    def __init__(self, device: torch.device):
        self.model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d', device=device)

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))

def build_adjacency_hetero(hetero_data: HeteroData, undirected: bool=True):
    adjacency = {node_type: [set() for _ in range(hetero_data[node_type].num_nodes)] for node_type in hetero_data.node_types}
    for edge_type in hetero_data.edge_types:
        (src_type, _, dst_type) = edge_type
        if 'edge_index' not in hetero_data[edge_type]:
            continue
        edge_index = hetero_data[edge_type].edge_index
        src_list = edge_index[0].tolist()
        dst_list = edge_index[1].tolist()
        for (s, d) in zip(src_list, dst_list):
            adjacency[src_type][s].add((dst_type, d))
            if undirected:
                adjacency[dst_type][d].add((src_type, s))
    return adjacency

def init_worker_globals(adj, all_nodes):
    global GLOBAL_ADJ, GLOBAL_ALL_NODES
    GLOBAL_ADJ = adj
    GLOBAL_ALL_NODES = all_nodes

def gather_1_and_2_hop_with_seed_time(adjacency: Dict[str, List[set]], data: dict, node_type: str, node_idx: int, seed_time: float, max_1hop_threshold: int=5000, max_2hop_threshold: int=1000) -> List[Tuple[str, int, int, float, Optional[set]]]:
    pass
    n1_full = list(adjacency[node_type][node_idx])
    valid_n1 = []
    for (nbr_t, nbr_i) in n1_full:
        if hasattr(data[nbr_t], 'time'):
            nbr_time = data[nbr_t].time[nbr_i].item()
            if nbr_time <= seed_time:
                valid_n1.append((nbr_t, nbr_i, nbr_time))
        else:
            valid_n1.append((nbr_t, nbr_i, 0.0))
    if len(valid_n1) > max_1hop_threshold:
        valid_n1.sort(key=lambda x: x[2], reverse=True)
        valid_n1 = valid_n1[:max_1hop_threshold]
    neighbors_with_time = []
    n1_ids = []
    for (nbr_t, nbr_i, nbr_time) in valid_n1:
        if hasattr(data[nbr_t], 'time'):
            relative_time_days = (seed_time - nbr_time) / (60 * 60 * 24)
        else:
            relative_time_days = 0.0
        neighbors_with_time.append((nbr_t, nbr_i, 1, relative_time_days, None))
        n1_ids.append((nbr_t, nbr_i))
    n2_candidates = defaultdict(set)
    for (nbr_t, nbr_i) in n1_ids:
        nbr2_full = list(adjacency[nbr_t][nbr_i])
        valid_nbr2_for_this_node = []
        for (nbr2_t, nbr2_i) in nbr2_full:
            if (nbr2_t, nbr2_i) == (node_type, node_idx):
                continue
            if hasattr(data[nbr2_t], 'time'):
                nbr2_time = data[nbr2_t].time[nbr2_i].item()
                if nbr2_time <= seed_time:
                    valid_nbr2_for_this_node.append((nbr2_t, nbr2_i, nbr2_time))
            else:
                valid_nbr2_for_this_node.append((nbr2_t, nbr2_i, 0.0))
        if len(valid_nbr2_for_this_node) > max_2hop_threshold:
            valid_nbr2_for_this_node.sort(key=lambda x: x[2], reverse=True)
            valid_nbr2_for_this_node = valid_nbr2_for_this_node[:max_2hop_threshold]
        for nbr2_id in valid_nbr2_for_this_node:
            (nbr2_t, nbr2_i) = (nbr2_id[0], nbr2_id[1])
            n2_candidates[nbr2_t, nbr2_i].add((nbr_t, nbr_i))
    n1_set = set(((x[0], x[1]) for x in neighbors_with_time))
    for ((nbr2_t, nbr2_i), connecting_1hops) in n2_candidates.items():
        if (nbr2_t, nbr2_i) not in n1_set:
            if hasattr(data[nbr2_t], 'time'):
                nbr2_time = data[nbr2_t].time[nbr2_i].item()
                relative_time_days = (seed_time - nbr2_time) / (60 * 60 * 24)
            else:
                relative_time_days = 0.0
            neighbors_with_time.append((nbr2_t, nbr2_i, 2, relative_time_days, connecting_1hops))
    return neighbors_with_time

def _process_one_seed(args):
    pass
    global GLOBAL_ADJ
    (data, K, seed_node_type, seed_node_idx, seed_time, seed_val) = args
    random.seed(seed_val)
    T_hat = gather_1_and_2_hop_with_seed_time(GLOBAL_ADJ, data, seed_node_type, seed_node_idx, seed_time)
    K_minus_1 = K - 1
    candidate_nodes = {(t, i): (t, i, hop, t_val, c1hops) for (t, i, hop, t_val, c1hops) in T_hat}
    local_adj = defaultdict(list)
    for (t, i, hop, _, c1hops) in T_hat:
        if hop == 2 and c1hops is not None:
            for c in c1hops:
                if c in candidate_nodes:
                    local_adj[c].append((t, i))
    queue = deque()
    visited = set()
    chosen_neighbors = []
    one_hop_nodes = [(t, i) for (t, i, hop, _, _) in T_hat if hop == 1]
    random.shuffle(one_hop_nodes)
    for u in one_hop_nodes:
        queue.append(u)
        visited.add(u)
    while queue and len(chosen_neighbors) < K_minus_1:
        u = queue.popleft()
        chosen_neighbors.append(candidate_nodes[u])
        for v in local_adj.get(u, []):
            if v not in visited:
                visited.add(v)
                queue.append(v)
    final_tokens = []
    seed_token = (seed_node_type, seed_node_idx, 0, 0.0, 0)
    final_tokens.append(seed_token)
    final_tokens.extend(chosen_neighbors)
    if len(final_tokens) > 1:
        first = final_tokens[0]
        rest = final_tokens[1:]
        random.shuffle(rest)
        final_tokens = [first] + rest
    local_map = {}
    for (j, (t_str, i, hop, t_val, c1hops)) in enumerate(final_tokens):
        local_map[t_str, i] = j
    edges = []
    for (j_src, (t_str, i, hop, t_val, c1hops)) in enumerate(final_tokens):
        if t_str in GLOBAL_ADJ and i < len(GLOBAL_ADJ[t_str]):
            for (nbr_t, nbr_i) in GLOBAL_ADJ[t_str][i]:
                if (nbr_t, nbr_i) in local_map:
                    j_dst = local_map[nbr_t, nbr_i]
                    edges.append((j_src, j_dst))
    if len(edges) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int32)
    else:
        arr = np.array(edges, dtype=np.int32)
        edge_index = arr.T
    return (seed_node_type, seed_node_idx, final_tokens, edge_index)

def local_nodes_hetero(data: HeteroData, K: int, table_input_nodes: tuple, table_input_time: torch.Tensor, undirected: bool=True, num_workers: int=None):
    pass
    global GLOBAL_ADJ, GLOBAL_ALL_NODES
    if GLOBAL_ADJ is None:
        adjacency = build_adjacency_hetero(data, undirected=undirected)
        GLOBAL_ADJ = adjacency
    else:
        adjacency = GLOBAL_ADJ
    if GLOBAL_ALL_NODES is None:
        all_nodes_all_types = []
        for nt in data.node_types:
            for i in range(data[nt].num_nodes):
                all_nodes_all_types.append((nt, i))
        GLOBAL_ALL_NODES = all_nodes_all_types
    else:
        all_nodes_all_types = GLOBAL_ALL_NODES
    (seed_node_type, seed_node_idxs) = table_input_nodes
    assert len(seed_node_idxs) == len(table_input_time), 'Mismatch in seed_node_idxs vs table_input_time'
    tasks = []
    for (i, node_idx_t) in enumerate(seed_node_idxs):
        node_idx = node_idx_t.item()
        seed_t = table_input_time[i].item()
        seed_val = hash((seed_node_type, node_idx, seed_t, K)) & 4294967295
        tasks.append((data, K, seed_node_type, node_idx, seed_t, seed_val))
    if num_workers is None:
        from multiprocessing import cpu_count
        num_workers = min(cpu_count() - 20, len(tasks))
    with Pool(processes=num_workers, initializer=init_worker_globals, initargs=(adjacency, all_nodes_all_types)) as pool:
        results = pool.map(_process_one_seed, tasks)
    S = {seed_node_type: {}}
    for (nt, idx, final_list, edge_index) in results:
        S[nt][idx] = (final_list, edge_index)
    return S

class RelGTTokens(Dataset):

    def __init__(self, data, task, K: int, split: str='train', undirected: bool=True, num_workers: int=None, precompute: bool=True, precomputed_dir: str=None, train_stage: str='finetune'):
        super().__init__()
        self.data = data
        self.task = task
        self.split = split
        self.K = K
        self.undirected = undirected
        self.num_workers = num_workers
        self.precompute = precompute
        self.precomputed_dir = precomputed_dir
        self.table = self.task.get_table(split=self.split)
        self.table_input = get_node_train_table_input(self.table, self.task)
        (self.node_type, self.node_idxs) = self.table_input.nodes
        self.target = self.table_input.target if self.table_input.target is not None else None
        self.time = getattr(self.table_input, 'time', None)
        self.transform = getattr(self.table_input, 'transform', None)
        self.node_types = self.data.node_types
        self.node_type_to_index = {nt: idx for (idx, nt) in enumerate(self.node_types)}
        self.index_to_node_type = {idx: nt for (idx, nt) in enumerate(self.node_types)}
        self.max_neighbor_hop = 2 + 1
        self._create_global_mappings()
        self.precomputed_path = self._construct_precomputed_path()
        self.train_stage = train_stage
        if self.precompute:
            if os.path.exists(self.precomputed_path):
                print(f'[{self.split}] Found existing HDF5 at {self.precomputed_path}')
            else:
                print(f'[{self.split}] Precomputing neighbor sampling (K={self.K})...')
                self._precompute_sampling()

    def _create_global_mappings(self):
        pass
        self.type_local_to_global = {}
        self.global_to_type_local = {}
        global_index = 0
        for (type_idx, node_type) in self.index_to_node_type.items():
            if 'x' in self.data[node_type]:
                num_nodes = self.data[node_type]['x'].size(0)
            else:
                num_nodes = self.data[node_type].num_nodes
            for local_idx in range(num_nodes):
                key = (type_idx, local_idx)
                self.type_local_to_global[key] = global_index
                self.global_to_type_local[global_index] = key
                global_index += 1

    def get_global_index(self, type_idxs: List[int], local_idxs: List[int]) -> List[int]:
        pass
        out = []
        for (t_i, l_i) in zip(type_idxs, local_idxs):
            out.append(self.type_local_to_global[t_i, l_i])
        return out

    def _construct_precomputed_path(self) -> str:
        if not self.precomputed_dir:
            raise ValueError("must provide a 'precomputed_dir' to store expansions.")
        path = os.path.join(self.precomputed_dir, str(self.K) + 'BFS3407', f'{self.split}.h5')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def __len__(self):
        return len(self.node_idxs)

    def _create_datasets(self, h5file: h5py.File, total_samples: int) -> dict:
        _chunk_size = min(total_samples, 10000)
        return {'types': h5file.create_dataset('types', shape=(total_samples, self.K), dtype='int16', chunks=(_chunk_size, self.K)), 'indices': h5file.create_dataset('indices', shape=(total_samples, self.K), dtype='int32', chunks=(_chunk_size, self.K)), 'hops': h5file.create_dataset('hops', shape=(total_samples, self.K), dtype='int8', chunks=(_chunk_size, self.K)), 'times': h5file.create_dataset('times', shape=(total_samples, self.K), dtype='float32', chunks=(_chunk_size, self.K))}

    def _precompute_sampling(self):
        pass
        total = len(self.node_idxs)
        chunk_size = 10000
        with h5py.File(self.precomputed_path, 'w') as hf:
            datasets = self._create_datasets(hf, total)
            adjacency_all = [None] * total
            with tqdm(total=total, desc=f"Precomputing '{self.split}'") as pbar:
                for start_idx in range(0, total, chunk_size):
                    end_idx = min(start_idx + chunk_size, total)
                    size_chunk = end_idx - start_idx
                    chunk_node_idxs = self.node_idxs[start_idx:end_idx]
                    chunk_times = None
                    if self.time is not None:
                        chunk_times = self.time[start_idx:end_idx]
                    S_chunk = local_nodes_hetero(data=self.data.to('cpu'), K=self.K, table_input_nodes=(self.node_type, chunk_node_idxs), table_input_time=chunk_times, undirected=self.undirected, num_workers=self.num_workers)
                    c_types = np.zeros((size_chunk, self.K), dtype=np.int16)
                    c_indices = np.zeros((size_chunk, self.K), dtype=np.int32)
                    c_hops = np.zeros((size_chunk, self.K), dtype=np.int8)
                    c_times = np.zeros((size_chunk, self.K), dtype=np.float32)
                    for (i, node_id) in enumerate(chunk_node_idxs):
                        (final_nodes, edge_index) = S_chunk[self.node_type][int(node_id)]
                        for (j, (t_str, nbr_loc_idx, hop, t_val, c1hops)) in enumerate(final_nodes):
                            c_types[i, j] = self.node_type_to_index[t_str]
                            c_indices[i, j] = nbr_loc_idx
                            c_hops[i, j] = hop
                            c_times[i, j] = t_val
                        adjacency_all[start_idx + i] = edge_index
                    datasets['types'][start_idx:end_idx] = c_types
                    datasets['indices'][start_idx:end_idx] = c_indices
                    datasets['hops'][start_idx:end_idx] = c_hops
                    datasets['times'][start_idx:end_idx] = c_times
                    pbar.update(size_chunk)
                    del S_chunk
                    gc.collect()
            offsets = np.zeros(total + 1, dtype=np.uint64)
            for i in range(total):
                if adjacency_all[i] is not None:
                    E_i = adjacency_all[i].shape[1]
                else:
                    E_i = 0
                offsets[i + 1] = offsets[i] + E_i
            total_edges = offsets[-1]
            edges_dset = hf.create_dataset('edges', shape=(2, total_edges), dtype='int16')
            for i in range(total):
                e_arr = adjacency_all[i]
                start = offsets[i]
                end_ = offsets[i + 1]
                if e_arr is not None and e_arr.size > 0:
                    edges_dset[:, start:end_] = e_arr
            hf.create_dataset('edges_offsets', data=offsets)

    def __getitem__(self, idx: int):
        pass
        with h5py.File(self.precomputed_path, 'r') as hf:
            sample = {'types': torch.from_numpy(hf['types'][idx]).long(), 'indices': torch.from_numpy(hf['indices'][idx]).long(), 'hops': torch.from_numpy(hf['hops'][idx]).long(), 'times': torch.from_numpy(hf['times'][idx])}
            offsets = hf['edges_offsets']
            edges_dset = hf['edges']
            start = offsets[idx]
            end_ = offsets[idx + 1]
            if start == end_:
                eidx = torch.zeros((2, 0), dtype=torch.long)
            else:
                edge_np = edges_dset[:, start:end_]
                eidx = torch.from_numpy(edge_np).long()
            sample['edge_index'] = eidx
        label = self.target[idx] if self.target is not None else None
        sample['first_type'] = sample['types'][0].item()
        sample['first_index'] = sample['indices'][0].item()
        sample['tfs'] = [self.data[self.index_to_node_type[t.item()]].tf[i.item()] for (t, i) in zip(sample['types'], sample['indices'])]
        sample['global_idx'] = idx
        return (sample, label)

    def collate(self, batch: List[Tuple[dict, Optional[torch.Tensor]]]):
        (samples, labels) = zip(*batch)
        neighbor_types = torch.stack([s['types'] for s in samples], dim=0)
        neighbor_indices = torch.stack([s['indices'] for s in samples], dim=0)
        neighbor_hops = torch.stack([s['hops'] for s in samples], dim=0)
        neighbor_times = torch.stack([s['times'] for s in samples], dim=0)
        out = {'neighbor_types': neighbor_types, 'neighbor_indices': neighbor_indices, 'neighbor_hops': neighbor_hops, 'neighbor_times': neighbor_times}
        if self.target is not None:
            out['labels'] = torch.stack(labels, dim=0)
        else:
            out['labels'] = None
        first_types = [s['first_type'] for s in samples]
        first_indices = [s['first_index'] for s in samples]
        out['node_indices'] = torch.tensor(self.get_global_index(first_types, first_indices), dtype=torch.long)
        (B, K) = neighbor_types.shape
        grouped_tfs = {}
        grouped_positions = {}
        for t_id in range(len(self.node_types)):
            mask = neighbor_types == t_id
            if not mask.any():
                continue
            local_idxs = neighbor_indices[mask]
            type_str = self.index_to_node_type[t_id]
            offsets_list = []
            positions_2d = torch.nonzero(mask, as_tuple=False)
            for (b, k) in positions_2d.tolist():
                offset = b * K + k
                offsets_list.append(offset)
            grouped_tfs[t_id] = self.data[type_str].tf[local_idxs]
            grouped_positions[t_id] = offsets_list
        flat_batch_idx = torch.arange(B).unsqueeze(1).expand(B, K).reshape(-1).tolist()
        flat_nbr_idx = torch.arange(K).repeat(B).tolist()
        global_idxs = [s['global_idx'] for s in samples]
        global_idxs = torch.tensor(global_idxs, dtype=torch.long)
        out.update({'grouped_tfs': grouped_tfs, 'grouped_indices': grouped_positions, 'flat_batch_idx': flat_batch_idx, 'flat_nbr_idx': flat_nbr_idx, 'global_idx': global_idxs})
        batched_edges = []
        batch_vec = []
        node_offset = 0
        for (i, sample) in enumerate(samples):
            eidx = sample['edge_index']
            K_i = sample['types'].size(0)
            shifted = eidx + node_offset
            batched_edges.append(shifted)
            sub_batch = torch.full((K_i,), i, dtype=torch.long)
            batch_vec.append(sub_batch)
            node_offset += K_i
        edge_index = torch.cat(batched_edges, dim=1) if batched_edges else torch.zeros((2, 0), dtype=torch.long)
        batch_out = torch.cat(batch_vec, dim=0) if batch_vec else torch.zeros((0,), dtype=torch.long)
        out.update({'edge_index': edge_index, 'batch': batch_out})
        return out

    def get_max_time(self) -> float:
        if not os.path.exists(self.precomputed_path):
            return 100.0
        with h5py.File(self.precomputed_path, 'r') as hf:
            if 'times' not in hf:
                return 0.0
            all_times = hf['times'][:]
            max_t = np.max(all_times)
            return float(max_t)
