import argparse
import copy
import json
import math
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
from pathlib import Path
from typing import Dict
import wandb
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pynvml
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.seed import seed_everything
from tqdm import tqdm
from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from model import RelGT
from utils import GloveTextEmbedding, RelGTTokens
from regression_utils import RegressionTargetTransform, apply_regression_calibration
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='rel-f1')
parser.add_argument('--task', type=str, default='driver-top3')
parser.add_argument('--precompute', action='store_true', default=True)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--warmup_steps', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--channels', type=int, default=512)
parser.add_argument('--aggr', type=str, default='sum')
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--gt_conv_type', type=str, default='full')
parser.add_argument('--ablate', type=str, default='none')
parser.add_argument('--gnn_pe_dim', type=int, default=0)
parser.add_argument('--num_neighbors', type=int, default=800)
parser.add_argument('--sample_node_len', type=int, default=300)
parser.add_argument('--edge_dropout', type=float, default=0.2)
parser.add_argument('--num_centroids', type=int, default=4096)
parser.add_argument('--ff_dropout', type=float, default=0.1)
parser.add_argument('--attn_dropout', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=1e-05)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'cosine'])
parser.add_argument('--min_lr_ratio', type=float, default=0.1)
parser.add_argument('--label_smoothing', type=float, default=0.0)
parser.add_argument('--warmup_ratio', type=float, default=0.0, help='Fraction of optimization steps used for linear learning-rate warmup; 0 disables it. Example: 0.1 warms up over the first 10%% of steps.')
parser.add_argument('--ema_decay', type=float, default=0.0, help='Exponential moving-average decay; 0 disables EMA. Evaluation and checkpoints use EMA weights when enabled.')
parser.add_argument('--select_on', type=str, default='val', choices=['val', 'test'], help='Checkpoint selection split: val (default) or test. Test-based selection is only for explicitly authorized tasks; its scores are not comparable with validation-selected scores. Recorded as select_on in JSON.')
parser.add_argument('--select_tol', type=float, default=0.0, help='Checkpoint selection tolerance: 0 selects the best validation epoch; a positive value selects the last epoch within this tolerance of the best validation score. The driver-top3 verification uses 0.002.')
parser.add_argument('--pos_weight', type=float, default=0.0, help='BCE pos_weight; <=0 disables')
parser.add_argument('--regression_transform', type=str, default='none', choices=['none', 'standardize', 'log1p', 'asinh', 'logit'])
parser.add_argument('--regression_loss', type=str, default='l1', choices=['l1', 'smooth_l1', 'mse', 'bce'])
parser.add_argument('--regression_calibrate', type=str, default='none', choices=['none', 'linear', 'median_blend', 'mae', 'shrink', 'logit_temp', 'power'])
parser.add_argument('--regression_jac_weight', type=float, default=0.0, help='Exponent for target-space Jacobian loss weighting; 0 disables it (default). Aligns the transformed log1p/logit training objective with original-space MAE.')
parser.add_argument('--regression_clamp', type=str, default='percentile', choices=['none', 'percentile', 'minmax'])
parser.add_argument('--regression_clamp_low', type=float, default=2.0)
parser.add_argument('--regression_clamp_high', type=float, default=98.0)
parser.add_argument('--smooth_l1_beta', type=float, default=1.0)
parser.add_argument('--save_raw_preds', action='store_true', help='Also save uncalibrated validation/test predictions for offline calibration comparisons (disabled by default).')
parser.add_argument('--repeat_prior_att28_coef', type=float, default=0.0, help='Post-hoc user-repeat prior: coefficient for z-scored past-28d yes attendance count')
parser.add_argument('--repeat_prior_birthyear_coef', type=float, default=0.0, help='Post-hoc user-repeat prior: coefficient for z-scored user birthyear')
parser.add_argument('--temporal_strategy', type=str, default='uniform')
parser.add_argument('--temporal_bias', type=str, default='gaussian', choices=['gaussian', 'linear', 'embedding'])
parser.add_argument('--pos_enc', type=str, default='none')
parser.add_argument('--max_degree', type=int, default=10000)
parser.add_argument('--pos_enc_dim', type=int, default=128)
parser.add_argument('--max_steps_per_epoch', type=int, default=3000)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out_dir', type=str, default='results/debug')
parser.add_argument('--run_name', type=str, default='debug')
parser.add_argument('--model_parameters', type=int, default=0, help='Number of model parameters')
parser.add_argument('--cache_dir', type=str, default=os.path.expanduser(os.environ.get('GELGT_CACHE_DIR', os.path.expanduser('~/.cache/gelgt'))))
parser.add_argument('--train_stage', type=str, default='finetune', choices=['finetune'])
parser.add_argument('--eval_ckpt', type=str, default=None, help='Evaluation only: load the specified finetuned.pt, skip training, and write final validation/test predictions and metrics as NPY/JSON. None runs normal training.')
parser.add_argument('--deterministic', type=str, default='off', choices=['off', 'strict', 'warn'], help='opt-in bit-reproducible training; default off = unchanged behaviour')
args = parser.parse_args()
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
device = torch.device('cuda', local_rank)
torch.cuda.set_device(device)
if local_rank == 0:
    args.run_name = f'{args.dataset}-{args.task}-{args.run_name}'

def init_gpu_utilization(device_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    return handle

def get_gpu_stats(handle, device):
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_util = util.gpu
    mem_allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
    mem_reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
    return (gpu_util, mem_allocated, mem_reserved)

def _user_repeat_prior_features(table_df, dataset_obj):
    pass
    db = dataset_obj.get_db()
    attendees = db.table_dict['event_attendees'].df[['user_id', 'start_time', 'status']].dropna(subset=['user_id']).copy()
    attendees['user_id'] = attendees['user_id'].astype(int)
    attendees = attendees[attendees['status'].eq('yes')].sort_values(['user_id', 'start_time'])
    groups = {int(u): g['start_time'].values.astype('datetime64[ns]').astype('int64') for (u, g) in attendees.groupby('user_id', sort=False)}
    ns_day = 24 * 3600 * 10 ** 9
    att28 = []
    for (ts, u) in zip(pd.to_datetime(table_df['timestamp']), table_df['user'].astype(int)):
        t = np.datetime64(ts).astype('datetime64[ns]').astype('int64')
        times = groups.get(int(u))
        if times is None:
            att28.append(0.0)
            continue
        idx = np.searchsorted(times, t, side='right')
        att28.append(float((times[:idx] > t - 28 * ns_day).sum()))
    users = db.table_dict['users'].df.set_index('user_id')
    birth = table_df['user'].map(users['birthyear']).fillna(users['birthyear'].median()).to_numpy(dtype=float)
    return (np.asarray(att28, dtype=float), birth)

def apply_user_repeat_prior(val_preds, test_preds, task_obj, dataset_obj, args_obj):
    if not (args_obj.dataset == 'rel-event' and args_obj.task == 'user-repeat'):
        return (val_preds, test_preds)
    if args_obj.repeat_prior_att28_coef == 0.0 and args_obj.repeat_prior_birthyear_coef == 0.0:
        return (val_preds, test_preds)
    val_df = task_obj.get_table('val').df
    test_df = task_obj.get_table('test').df
    (v_att28, v_birth) = _user_repeat_prior_features(val_df, dataset_obj)
    (t_att28, t_birth) = _user_repeat_prior_features(test_df, dataset_obj)

    def zscore(v, t):
        mu = float(np.nanmean(v))
        sd = float(np.nanstd(v)) + 1e-09
        return ((v - mu) / sd, (t - mu) / sd)
    (v_att28, t_att28) = zscore(v_att28, t_att28)
    (v_birth, t_birth) = zscore(v_birth, t_birth)
    val_blend = val_preds + args_obj.repeat_prior_att28_coef * v_att28 + args_obj.repeat_prior_birthyear_coef * v_birth
    test_blend = test_preds + args_obj.repeat_prior_att28_coef * t_att28 + args_obj.repeat_prior_birthyear_coef * t_birth
    print(f'Applied user-repeat prior: att28_coef={args_obj.repeat_prior_att28_coef}, birthyear_coef={args_obj.repeat_prior_birthyear_coef}')
    return (val_blend, test_blend)
print(f'Using device: {device}')
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)
if args.deterministic != 'off':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=args.deterministic == 'warn')
    print(f"[deterministic] mode={args.deterministic} cudnn.deterministic=True benchmark=False CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
gpu_handle = init_gpu_utilization(local_rank)
dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)
stypes_cache_path = Path(f'{args.cache_dir}/{args.dataset}/stypes.json')
try:
    with open(stypes_cache_path, 'r') as f:
        col_to_stype_dict = json.load(f)
    for (table, col_to_stype) in col_to_stype_dict.items():
        for (col, stype_str) in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, 'w') as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)
(data, col_stats_dict) = make_pkey_fkey_graph(dataset.get_db(), col_to_stype_dict=col_to_stype_dict, text_embedder_cfg=TextEmbedderConfig(text_embedder=GloveTextEmbedding(device=f'cuda:{local_rank}'), batch_size=256), cache_dir=f'{args.cache_dir}/{args.dataset}/materialized')
data = {split: RelGTTokens(data=data, task=task, K=args.num_neighbors, split=split, undirected=True, precompute=args.precompute, precomputed_dir=f'{args.cache_dir}/precomputed/{args.dataset}/{args.task}', num_workers=args.num_workers, train_stage=args.train_stage) for split in ['train', 'val', 'test']}
print('Scanning precomputed files for max relative time...')
max_time_list = []
for (split, dataset_obj) in data.items():
    t = dataset_obj.get_max_time()
    max_time_list.append(t)
GLOBAL_MAX_TIME = max(max_time_list)
if GLOBAL_MAX_TIME <= 0:
    GLOBAL_MAX_TIME = 365.0
if local_rank == 0:
    print(f'Detected Max Relative Time: {GLOBAL_MAX_TIME:.2f} days')
train_sampler = DistributedSampler(data['train'], shuffle=True, seed=args.seed)
loader_train = DataLoader(data['train'], batch_size=args.batch_size, sampler=train_sampler, collate_fn=data['train'].collate, num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=True)
val_sampler = DistributedSampler(data['val'], shuffle=False, seed=args.seed, drop_last=False)
loader_val = DataLoader(data['val'], batch_size=args.batch_size, sampler=val_sampler, collate_fn=data['val'].collate, num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=True)
test_sampler = DistributedSampler(data['test'], shuffle=False, seed=args.seed, drop_last=False)
loader_test = DataLoader(data['test'], batch_size=args.batch_size, sampler=test_sampler, collate_fn=data['test'].collate, num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=True)
loader_dict: Dict[str, DataLoader] = {'train': loader_train, 'val': loader_val, 'test': loader_test}
(clamp_min, clamp_max) = (None, None)
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    if args.pos_weight and args.pos_weight > 0:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    tune_metric = 'roc_auc'
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    if args.regression_loss == 'smooth_l1':
        loss_fn = nn.SmoothL1Loss(beta=args.smooth_l1_beta)
    elif args.regression_loss == 'mse':
        loss_fn = MSELoss()
    elif args.regression_loss == 'bce':
        if args.regression_transform != 'logit':
            raise ValueError('--regression_loss bce requires --regression_transform logit')
        if args.pos_weight and args.pos_weight > 0:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        if local_rank == 0:
            print('[bce] Regression uses BCEWithLogits loss (pos_weight=%s)' % args.pos_weight)
    else:
        loss_fn = L1Loss()
    if args.regression_jac_weight and args.regression_jac_weight > 0:
        if args.regression_loss == 'smooth_l1':
            loss_fn_none = nn.SmoothL1Loss(beta=args.smooth_l1_beta, reduction='none')
        elif args.regression_loss == 'mse':
            loss_fn_none = MSELoss(reduction='none')
        else:
            loss_fn_none = L1Loss(reduction='none')
        if local_rank == 0:
            print(f'[jac] Target-space Jacobian weighting enabled; exponent={args.regression_jac_weight}')
    tune_metric = 'mae'
    higher_is_better = False
    train_table = task.get_table('train')
    regression_train_labels = train_table.df[task.target_col].to_numpy()
    regression_transform = RegressionTargetTransform.from_labels(regression_train_labels, mode=args.regression_transform, clamp_mode=args.regression_clamp, clamp_percentiles=(args.regression_clamp_low, args.regression_clamp_high))
    (clamp_min, clamp_max) = (regression_transform.clamp_min, regression_transform.clamp_max)
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    loss_fn = BCEWithLogitsLoss()
    tune_metric = 'multilabel_auprc_macro'
    higher_is_better = True
else:
    raise ValueError(f'Task type {task.task_type} is unsupported')
model = RelGT(num_nodes=data['train'].data.num_nodes, max_neighbor_hop=data['train'].max_neighbor_hop, node_type_map=data['train'].node_type_to_index, col_names_dict={node_type: data['train'].data[node_type].tf.col_names_dict for node_type in data['train'].data.node_types}, col_stats_dict=col_stats_dict, local_num_layers=args.num_layers, channels=args.channels, out_channels=out_channels, global_dim=args.channels // 2, heads=args.num_heads, ff_dropout=args.ff_dropout, attn_dropout=args.attn_dropout, conv_type=args.gt_conv_type, ablate=args.ablate, gnn_pe_dim=args.gnn_pe_dim, num_centroids=args.num_centroids, sample_node_len=args.sample_node_len, max_time_days=GLOBAL_MAX_TIME, edge_dropout=args.edge_dropout, temporal_bias=args.temporal_bias, args=args).to(device)
for (name, param) in model.named_parameters():
    if param.dtype == torch.int16:
        param.data = param.data.to(torch.int64)
for (name, buf) in model.named_buffers():
    if buf.dtype == torch.int16:
        buf.data = buf.data.to(torch.int64)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
if local_rank == 0:
    print(model)
total_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
if local_rank == 0:
    print(f'Total model parameters: {total_params}')
args.model_parameters = total_params
if local_rank == 0:
    wandb.init(project='rel-gt-expts', name=args.run_name, config=vars(args))
output_path = os.path.join(args.out_dir, args.dataset, args.task)
os.makedirs(output_path, exist_ok=True)
world_size = dist.get_world_size()
base_lr = args.lr * world_size
if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

class ModelEMA:
    pass

    def __init__(self, module, decay):
        self.decay = decay
        self.module = copy.deepcopy(module).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, module):
        msd = module.state_dict()
        for (k, v) in self.module.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                v.copy_(src)
steps_per_epoch = min(len(loader_dict['train']), args.max_steps_per_epoch)
total_opt_steps = max(1, steps_per_epoch * args.epochs)
scheduler = None
step_scheduler_per_step = False
if args.warmup_ratio and args.warmup_ratio > 0:
    warmup_steps = max(1, int(args.warmup_ratio * total_opt_steps))
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / warmup_steps, end_factor=1.0, total_iters=warmup_steps), torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_opt_steps - warmup_steps), eta_min=base_lr * args.min_lr_ratio)], milestones=[warmup_steps])
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / warmup_steps, end_factor=1.0, total_iters=warmup_steps)
    step_scheduler_per_step = True
    if local_rank == 0:
        print(f"[warmup] Total optimization steps {total_opt_steps}, warmup {warmup_steps} steps ({args.warmup_ratio:.1%}), then {('cosine annealing' if args.scheduler == 'cosine' else 'constant LR')}")
elif args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=base_lr * args.min_lr_ratio)
ema = None
if args.ema_decay and args.ema_decay > 0:
    ema = ModelEMA(model.module, args.ema_decay)
    if local_rank == 0:
        print(f'[EMA] decay={args.ema_decay}, evaluation and checkpoints use EMA weights')
global_step = 0
train_time = 0

def train_supervised(epoch) -> float:
    global global_step
    global train_time
    model.train()
    loss_accum = count_accum = 0
    total_steps = min(len(loader_dict['train']), args.max_steps_per_epoch)
    train_sampler.set_epoch(epoch)
    for (step, batch) in enumerate(tqdm(loader_dict['train'], total=total_steps, desc='Train'), start=1):
        neighbor_types = batch['neighbor_types'].to(device)
        node_indices = batch['node_indices'].to(device)
        neighbor_hops = batch['neighbor_hops'].to(device)
        neighbor_times = batch['neighbor_times'].to(device)
        edge_index = batch['edge_index'].to(device)
        batch_vec = batch['batch'].to(device)
        grouped_tf_dict = {'grouped_tfs': batch['grouped_tfs'], 'grouped_indices': batch['grouped_indices'], 'flat_batch_idx': batch['flat_batch_idx'], 'flat_nbr_idx': batch['flat_nbr_idx']}
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        start_time = time.time()
        pred = model(neighbor_types, node_indices, neighbor_hops, neighbor_times, grouped_tf_dict, edge_index=edge_index, batch=batch_vec)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        labels_for_loss = labels
        if task.task_type == TaskType.REGRESSION and args.regression_loss != 'bce':
            labels_for_loss = regression_transform.forward(labels_for_loss)
        if args.label_smoothing and args.label_smoothing > 0 and (task.task_type == TaskType.BINARY_CLASSIFICATION):
            labels_for_loss = labels * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
        if task.task_type == TaskType.REGRESSION and args.regression_jac_weight > 0 and (args.regression_loss != 'bce'):
            _p = pred.float().reshape(-1)
            _t = labels_for_loss.float().reshape(-1)
            _w = regression_transform.jacobian_weight(labels.float().reshape(-1))
            _w = _w.pow(args.regression_jac_weight)
            loss = (loss_fn_none(_p, _t) * _w).sum() / torch.clamp(_w.sum(), min=1e-06)
        else:
            loss = loss_fn(pred.float(), labels_for_loss.float())
        loss.backward()
        train_time = train_time + time.time() - start_time
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ema is not None:
            ema.update(model.module)
        if scheduler is not None and step_scheduler_per_step:
            scheduler.step()
        loss_value = loss.detach().item()
        (gpu_util, mem_allocated, mem_reserved) = get_gpu_stats(gpu_handle, device)
        if local_rank == 0:
            wandb.log({'train_loss': loss_value, 'global_step': global_step, 'lr': optimizer.param_groups[0]['lr'], 'gpu_util_percent': gpu_util, 'gpu_mem_allocated_MB': mem_allocated, 'gpu_mem_reserved_MB': mem_reserved})
        loss_accum += loss_value * pred.size(0)
        count_accum += pred.size(0)
        global_step += 1
        if step >= args.max_steps_per_epoch:
            break
    return loss_accum / count_accum if count_accum > 0 else float('inf')
inference_time = 0

@torch.no_grad()
def test(loader: DataLoader, eval_model, epoch, desc) -> np.ndarray:
    global inference_time
    if loader.sampler is not None and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    eval_model.eval()
    pred_list = []
    idx_list = []
    for batch in tqdm(loader, desc=desc, disable=local_rank != 0):
        neighbor_types = batch['neighbor_types'].to(device)
        node_indices = batch['node_indices'].to(device)
        neighbor_hops = batch['neighbor_hops'].to(device)
        neighbor_times = batch['neighbor_times'].to(device)
        edge_index = batch['edge_index'].to(device)
        batch_vec = batch['batch'].to(device)
        grouped_tf_dict = {'grouped_tfs': batch['grouped_tfs'], 'grouped_indices': batch['grouped_indices'], 'flat_batch_idx': batch['flat_batch_idx'], 'flat_nbr_idx': batch['flat_nbr_idx']}
        start_time = time.time()
        pred = eval_model(neighbor_types, node_indices, neighbor_hops, neighbor_times, grouped_tf_dict, edge_index=edge_index, batch=batch_vec)
        inference_time = time.time() - start_time
        if task.task_type == TaskType.REGRESSION:
            pred = regression_transform.inverse(pred)
        if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
            pred = torch.sigmoid(pred)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu().numpy())
        idx_list.append(batch['global_idx'].cpu().numpy())
    local_preds = np.concatenate(pred_list, axis=0) if pred_list else np.array([])
    local_idxs = np.concatenate(idx_list, axis=0) if idx_list else np.array([])
    gathered = [None for _ in range(world_size)] if local_rank == 0 else None
    dist.gather_object((local_idxs, local_preds), object_gather_list=gathered, dst=0)
    if local_rank == 0:
        all_preds = np.full((len(loader.dataset),), -100.0)
        for i in range(world_size):
            (g_idx, g_pred) = gathered[i]
            for (idx, pred) in zip(g_idx, g_pred):
                all_preds[idx] = pred
        return all_preds
    else:
        return None
if args.train_stage == 'finetune':
    best_val_metric = -math.inf if higher_is_better else math.inf
    best_epoch = 0
    state_dict = None
    if args.eval_ckpt:
        try:
            state_dict = torch.load(args.eval_ckpt, map_location='cpu', weights_only=True)
        except TypeError:
            state_dict = torch.load(args.eval_ckpt, map_location='cpu')
        print(f'[eval_ckpt] loaded {args.eval_ckpt}, skipping training')
    for epoch in () if args.eval_ckpt else range(1, args.epochs + 1):
        train_loss = train_supervised(epoch)
        print(f'epoch:{epoch} Training Time is {train_time / epoch}')
        dist.barrier()
        eval_model = ema.module if ema is not None else model.module
        val_pred = test(loader_dict['val'], eval_model=eval_model, epoch=epoch, desc='Val')
        test_pred = test(loader_dict['test'], eval_model=eval_model, epoch=epoch, desc='Test')
        if local_rank == 0:
            if task.task_type == TaskType.REGRESSION and args.regression_calibrate != 'none':
                val_targets = task.get_table('val').df[task.target_col].to_numpy()
                (val_pred, test_pred) = apply_regression_calibration(val_pred, test_pred, val_targets, method=args.regression_calibrate, center=float(np.median(regression_train_labels)), clamp_min=clamp_min, clamp_max=clamp_max)
            val_metrics = task.evaluate(val_pred, task.get_table('val'))
            test_metrics = task.evaluate(test_pred)
            print(f'Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}, Test metrics:{test_metrics}')
            wandb.log({'epoch': epoch, 'epoch_train_loss': train_loss, **{f'val_{k}': v for (k, v) in val_metrics.items()}})
            cur_val = val_metrics[tune_metric]
            if args.select_on == 'test':
                cur_val = test_metrics[tune_metric]
            if higher_is_better:
                is_better = cur_val > best_val_metric
                within_tol = cur_val >= best_val_metric - args.select_tol
            else:
                is_better = cur_val < best_val_metric
                within_tol = cur_val <= best_val_metric + args.select_tol
            if is_better:
                best_val_metric = cur_val
            if within_tol:
                best_epoch = epoch
                state_dict = copy.deepcopy(eval_model.state_dict())
                torch.save(state_dict, os.path.join(output_path, 'finetuned.pt'))
                if local_rank == 0:
                    print(f'[select] e{epoch:02d} {tune_metric}={cur_val:.4f} peak={best_val_metric:.4f} tol={args.select_tol} -> save checkpoint')
        if scheduler is not None and (not step_scheduler_per_step):
            scheduler.step()
        dist.barrier()
    if local_rank == 0 and state_dict is not None:
        model.module.load_state_dict(state_dict)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for buf in model.buffers():
        dist.broadcast(buf.data, src=0)
    dist.barrier()
    final_val_preds = test(loader_dict['val'], eval_model=model.module, epoch=best_epoch, desc='Val')
    final_test_preds = test(loader_dict['test'], eval_model=model.module, epoch=best_epoch, desc='Test')
    if local_rank == 0:
        (final_val_preds, final_test_preds) = apply_user_repeat_prior(final_val_preds, final_test_preds, task, dataset, args)
        if args.save_raw_preds:
            np.save(os.path.join(output_path, f'{args.seed}_val_pred_raw.npy'), final_val_preds)
            np.save(os.path.join(output_path, f'{args.seed}_test_pred_raw.npy'), final_test_preds)
        if task.task_type == TaskType.REGRESSION and args.regression_calibrate != 'none':
            val_targets = task.get_table('val').df[task.target_col].to_numpy()
            (final_val_preds, final_test_preds) = apply_regression_calibration(final_val_preds, final_test_preds, val_targets, method=args.regression_calibrate, center=float(np.median(regression_train_labels)), clamp_min=clamp_min, clamp_max=clamp_max)
        np.save(os.path.join(output_path, f'{args.seed}_val_pred.npy'), final_val_preds)
        np.save(os.path.join(output_path, f'{args.seed}_test_pred.npy'), final_test_preds)
        val_metrics = task.evaluate(final_val_preds, task.get_table('val'))
        print(f'Best Val metrics: {val_metrics}')
        test_metrics = task.evaluate(final_test_preds)
        print(f'Best Test metrics: {test_metrics}')
        best_metrics_dict = {'val_metrics': val_metrics, 'test_metrics': test_metrics, 'select_on': args.select_on}
        file_path = os.path.join(output_path, str(args.seed) + '.json')
        with open(file_path, 'w') as f:
            json.dump(best_metrics_dict, f, indent=4)
    if local_rank == 0:
        print(f'[{args.train_stage.capitalize()} Stage] Training complete. No supervised evaluation performed.')
    print('train_time:', train_time)
    print('inference_time:', inference_time)
dist.destroy_process_group()
