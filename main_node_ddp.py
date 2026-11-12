import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch_geometric.seed import seed_everything
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.base import TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from model import GelGT
from utils import GloveTextEmbedding, RelGTTokens, TemporalSampler, graph_fingerprint


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='rel-f1')
    parser.add_argument('--task', default='driver-top3')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--channels', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_neighbors', type=int, default=800)
    parser.add_argument('--sample_node_len', type=int, default=300)
    parser.add_argument('--edge_dropout', type=float, default=0.2)
    parser.add_argument('--ff_dropout', type=float, default=0.1)
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_steps_per_epoch', type=int, default=3000)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=Path, default=Path('results'))
    parser.add_argument('--cache_dir', type=Path,
                        default=Path(os.environ.get('GELGT_CACHE_DIR', '~/.cache/gelgt')).expanduser())
    args = parser.parse_args(argv)
    for name in ('epochs', 'batch_size', 'channels', 'num_layers', 'num_heads',
                 'num_neighbors', 'sample_node_len', 'max_steps_per_epoch'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if args.sample_node_len > args.num_neighbors:
        parser.error('sample_node_len must not exceed num_neighbors')
    if args.channels % args.num_heads or args.num_workers < 0:
        parser.error('channels must be divisible by num_heads; num_workers must be nonnegative')
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error('lr must be positive; weight_decay must be nonnegative')
    if any(not 0 <= getattr(args, name) < 1 for name in ('edge_dropout', 'ff_dropout', 'attn_dropout')):
        parser.error('dropout probabilities must be in [0, 1)')
    return args


def configure_cache(path):
    path = path.expanduser().resolve()
    os.environ['GELGT_CACHE_DIR'] = str(path)
    os.environ['XDG_CACHE_HOME'] = str(path)
    os.environ['RELBENCH_CACHE'] = str(path / 'relbench')
    for key, suffix in [('HF_HOME', 'huggingface'), ('TORCH_HOME', 'torch'),
                        ('SENTENCE_TRANSFORMERS_HOME', 'sentence_transformers')]:
        os.environ.setdefault(key, str(path / suffix))
    path.mkdir(parents=True, exist_ok=True)
    return path


def forward(model, batch, device):
    grouped = {name: batch[name] for name in ('grouped_tfs', 'grouped_indices')}
    return model(*(batch[name].to(device) for name in (
        'neighbor_types', 'neighbor_indices', 'neighbor_hops', 'neighbor_times')),
        grouped, edge_index=batch['edge_index'].to(device)).reshape(-1)


def train_epoch(model, loader, optimizer, loss_fn, device, max_steps):
    model.train()
    total_loss, total_count = 0.0, 0
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        prediction = forward(model, batch, device)
        labels = batch['labels'].to(device).float().reshape(-1)
        loss = loss_fn(prediction, labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.detach().item() * labels.numel()
        total_count += labels.numel()
    totals = torch.tensor([total_loss, total_count], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(totals)
    return float(totals[0] / totals[1].clamp_min(1))


@torch.no_grad()
def predict(model, loader, device, classification, rank, world_size):
    model.eval()
    rows, values = [], []
    for batch in loader:
        output = forward(model, batch, device)
        if classification:
            output = output.sigmoid()
        rows.append(batch['global_idx'].numpy())
        values.append(output.cpu().numpy())
    local = (np.concatenate(rows) if rows else np.empty(0, dtype=np.int64),
             np.concatenate(values) if values else np.empty(0, dtype=np.float32))
    if dist.is_initialized():
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local, object_gather_list=gathered, dst=0)
    else:
        gathered = [local]
    if rank != 0:
        return None
    result = np.empty(len(loader.dataset), dtype=np.float64)
    seen = np.zeros(len(result), dtype=bool)
    for indices, predictions in gathered:
        if len(np.unique(indices)) != len(indices) or seen[indices].any():
            raise RuntimeError('Duplicate evaluation rows')
        result[indices] = predictions
        seen[indices] = True
    if not seen.all() or not np.isfinite(result).all():
        raise RuntimeError('Evaluation predictions are incomplete or non-finite')
    return result


def make_loader(tokens, args, rank, world_size, training=False):
    if training:
        sampler = DistributedSampler(tokens, num_replicas=world_size, rank=rank,
                                      shuffle=True, seed=args.seed)
    else:
        sampler = range(rank, len(tokens), world_size)
    return DataLoader(tokens, batch_size=args.batch_size, sampler=sampler,
                      collate_fn=tokens.collate, num_workers=args.num_workers,
                      persistent_workers=args.num_workers > 0, pin_memory=True)


def main(argv=None):
    args = parse_args(argv)
    cache_dir = configure_cache(args.cache_dir)
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world_size > 1:
        dist.init_process_group('nccl')
    try:
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 1) // world_size)))
        seed_everything(args.seed)
        dataset = get_dataset(args.dataset, download=True)
        task = get_task(args.dataset, args.task, download=True)
        if task.task_type not in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
            raise ValueError('Only binary classification and regression are supported')
        classification = task.task_type == TaskType.BINARY_CLASSIFICATION
        database = dataset.get_db(upto_test_timestamp=True)
        graph_cache = cache_dir / args.dataset / 'main-v2'
        columns_path = graph_cache / 'stypes.json'
        if rank == 0:
            graph_cache.mkdir(parents=True, exist_ok=True)
            if not columns_path.exists():
                columns = get_stype_proposal(database)
                columns_path.write_text(json.dumps(columns, default=str))
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])
        columns = json.loads(columns_path.read_text())
        columns = {name: {column: stype(value) for column, value in mapping.items()}
                   for name, mapping in columns.items()}

        def materialize():
            return make_pkey_fkey_graph(database, col_to_stype_dict=columns,
                text_embedder_cfg=TextEmbedderConfig(text_embedder=GloveTextEmbedding(str(device)),
                                                     batch_size=256),
                cache_dir=str(graph_cache / 'materialized'))

        if rank == 0:
            graph, statistics = materialize()
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])
        if rank != 0:
            graph, statistics = materialize()
        graph_hash = graph_fingerprint(graph)
        sampler = TemporalSampler(graph, args.seed) if rank == 0 else None
        token_args = dict(data=graph, task=task, budget=args.num_neighbors,
                           cache_dir=graph_cache, sampling_seed=args.seed,
                           graph_hash=graph_hash)
        if rank == 0:
            train_tokens = RelGTTokens(**token_args, split='train', sampler=sampler)
            val_tokens = RelGTTokens(**token_args, split='val', sampler=sampler)
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])
        if rank != 0:
            train_tokens = RelGTTokens(**token_args, split='train')
            val_tokens = RelGTTokens(**token_args, split='val')
        if not len(train_tokens) or not len(val_tokens):
            raise ValueError('Training and validation tables must be nonempty')
        train_loader = make_loader(train_tokens, args, rank, world_size, training=True)
        val_loader = make_loader(val_tokens, args, rank, world_size)
        model = GelGT(max_neighbor_hop=train_tokens.max_neighbor_hop,
            node_type_map=train_tokens.node_type_to_index,
            col_names_dict={name: graph[name].tf.col_names_dict for name in graph.node_types},
            col_stats_dict=statistics, local_num_layers=args.num_layers,
            channels=args.channels, heads=args.num_heads, ff_dropout=args.ff_dropout,
            attn_dropout=args.attn_dropout, sample_node_len=args.sample_node_len,
            max_time_days=max(1.0, train_tokens.get_max_time()), edge_dropout=args.edge_dropout).to(device)
        if world_size > 1:
            model = DistributedDataParallel(model, device_ids=[local_rank],
                                             find_unused_parameters=True)
        module = model.module if world_size > 1 else model
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss() if classification else nn.L1Loss()
        metric_name = 'roc_auc' if classification else 'mae'
        best_score = -math.inf if classification else math.inf
        best_epoch, best_state = 0, None
        output = args.out_dir / args.dataset / args.task / f'seed-{args.seed}'
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, args.epochs + 1):
            train_loader.sampler.set_epoch(epoch)
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device,
                                     args.max_steps_per_epoch)
            if world_size > 1:
                for value in module.buffers():
                    dist.broadcast(value, src=0)
            predictions = predict(module, val_loader, device, classification, rank, world_size)
            if rank == 0:
                metrics = task.evaluate(predictions, val_tokens.table)
                score = float(metrics[metric_name])
                if not math.isfinite(score):
                    raise RuntimeError('Validation metric is non-finite')
                if (score > best_score) if classification else (score < best_score):
                    best_score, best_epoch = score, epoch
                    best_state = {name: value.detach().cpu().clone()
                                  for name, value in module.state_dict().items()}
                print(json.dumps(dict(epoch=epoch, train_loss=train_loss, val_metrics=metrics)), flush=True)
            if world_size > 1:
                dist.barrier(device_ids=[local_rank])
        if rank == 0:
            if best_state is None:
                raise RuntimeError('No validation-selected checkpoint is available')
            torch.save(best_state, output / 'model.pt')
            module.load_state_dict(best_state)
        if world_size > 1:
            for value in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(value.data, src=0)
        final_val = predict(module, val_loader, device, classification, rank, world_size)
        if rank == 0:
            test_tokens = RelGTTokens(**token_args, split='test', sampler=sampler)
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])
        if rank != 0:
            test_tokens = RelGTTokens(**token_args, split='test')
        test_loader = make_loader(test_tokens, args, rank, world_size)
        final_test = predict(module, test_loader, device, classification, rank, world_size)
        if rank == 0:
            result = dict(dataset=args.dataset, task=args.task, seed=args.seed,
                best_epoch=best_epoch, selection_split='val',
                val_metrics=task.evaluate(final_val, val_tokens.table),
                test_metrics=task.evaluate(final_test),
                parameters=sum(parameter.numel() for parameter in module.parameters()),
                configuration={name: str(value) if isinstance(value, Path) else value
                               for name, value in vars(args).items()},
                world_size=world_size, graph_fingerprint=graph_hash,
                train_cache_fingerprint=train_tokens.fingerprint,
                val_cache_fingerprint=val_tokens.fingerprint,
                test_cache_fingerprint=test_tokens.fingerprint)
            (output / 'metrics.json').write_text(json.dumps(result, indent=2))
            np.save(output / 'val_predictions.npy', final_val)
            np.save(output / 'test_predictions.npy', final_test)
            print(json.dumps(result), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
