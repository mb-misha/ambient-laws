# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch

import dataset_gbm
import dnnlib
from torch_utils import distributed as dist
from training import training_loop
import ambient_utils
import warnings
import wandb
import string
import random

warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       type=click.Choice(['vp', 've', 'edm']), default='edm', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--weight_decay',  help='Weight decay', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=False), default=0.0, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option("--expr_id", help="Experiment ID", type=str, default="test")
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)


# Scaling laws related
@click.option("--corruption_probability", help="Controls what percentage of images should be corrupted.", type=float, default=0.0)
@click.option("--sigma", help="How much noise to add to the corrupted images.", type=float, default=0.0)
@click.option('--dataset_keep_percentage', help='Limit training samples.', type=float, default=1.0, show_default=True)
@click.option("--noise_type", help="Type of noise to add to corrupt images.", type=str)

# Consistency params
@click.option("--consistency_batch_size", help="Batch size for the consistency loss.", type=int, default=32)
@click.option("--with_weight", help="Whether to use weight in the consistency loss.", type=bool, default=False)
@click.option("--with_grad", help="Whether to use gradient in the consistency loss.", type=bool, default=True)
@click.option("--num_consistency_steps", help="Number of steps for the consistency loss.", type=int, default=6)
@click.option("--num_primes", help="Number of primes for the consistency loss.", type=int, default=6)
@click.option("--consistency_coeff", help="Coefficient for the consistency loss.", type=float, default=0.0)


# GBM params
@click.option("--stochastic_model", help="Stochastic model to use.", type=str, default="GBM")
@click.option("--n_paths", help="Number of paths for the simulation.", type=int, default=10000)
@click.option("--n_steps", help="Number of steps in the simulation.", type=int, default=200)
@click.option("--n_ts_features", help="Number of time-series features.", type=int, default=1)
@click.option("--s_price", help="Initial value of the asset price.", type=float, default=100.0)
@click.option("--mu", help="Expected return rate of the asset.", type=float, default=0.05)
@click.option("--sigma_gbm", help="Volatility of the asset.", type=float, default=0.0)
@click.option("--return_log_returns", help="Whether to return log returns instead of price paths.", type=bool, default=False)
@click.option("--normalize", help="Normalization method for the paths.", type=str, default=None)

# Heston params
@click.option("--kappa", help="Mean reversion rate.", type=float, default=1.0)
@click.option("--theta", help="Long-term variance.", type=float, default=0.04)
@click.option("--sigma_v", help="Volatility of volatility.", type=float, default=0.1)
@click.option("--rho", help="Correlation between asset and volatility.", type=float, default=0.5)
@click.option("--v0", help="Initial variance.", type=float, default=0.01)

# MJD params
@click.option("--mu_j", help="Expected jump size.", type=float, default=0.01)
@click.option("--sigma_j", help="Volatility of the jump size.", type=float, default=0.2)
@click.option("--lamb", help="Jump intensity.", type=float, default=5.0)

# Market data params
@click.option('--symbol', help='Symbol of the stock to use.', type=str, default='AAPL')
@click.option('--sliding_window', help='Sliding window type', type=str, default='non_overlapping')

def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    if dist.get_rank() == 0:
        wandb.init(project="ambient_laws",
                   config=opts, name=opts.expr_id,
                   dir=opts.outdir)

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache, sigma=opts.sigma,
                                       corruption_probability_per_image=opts.corruption_probability, corruption_probability_per_pixel=1.0,
                                       only_positive=False)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8, weight_decay=opts.weight_decay)
    c.gbm_kwargs = dnnlib.EasyDict(
        n_paths=opts.n_paths,
        n_steps=opts.n_steps,
        n_ts_features=opts.n_ts_features,
        s_price=opts.s_price,
        mu=opts.mu,
        sigma_gbm=opts.sigma_gbm,
        return_log_returns=opts.return_log_returns,
        normalize=opts.normalize,
        noise_type=opts.noise_type,
        sigma=opts.sigma,
        corruption_probability=opts.corruption_probability,
    )
    if opts.stochastic_model == "Heston":
        c.gbm_kwargs.update(
            kappa=opts.kappa,
            theta=opts.theta,
            sigma_v=opts.sigma_v,
            rho=opts.rho,
            v0=opts.v0
        )
        c.gbm_kwargs.pop("sigma_gbm")
    elif opts.stochastic_model == "MJD":
        c.gbm_kwargs.update(
            mu_j=opts.mu_j,
            sigma_j=opts.sigma_j,
            lamb=opts.lamb
        )
    elif opts.stochastic_model == "MarketData":
        c.gbm_kwargs.update(
            symbol=opts.symbol,
            sliding_window=opts.sliding_window
        )
        c.gbm_kwargs.pop('s_price')
        c.gbm_kwargs.pop('mu')
        c.gbm_kwargs.pop('sigma_gbm')
    elif opts.stochastic_model == "HistoricalData":
        c.gbm_kwargs.pop('s_price')
        c.gbm_kwargs.pop('mu')
        c.gbm_kwargs.pop('sigma_gbm')
    elif opts.stochastic_model == 'CorrelatedGBMGenerativeDataset':
        c.gbm_kwargs.update(
            corr_matrix=[
                [1.0, 0.8, 0.4],
                [0.8, 1.0, 0.2],
                [0.4, 0.2, 1.0],
            ]
        ),
        c.gbm_kwargs.pop('mu')
        c.gbm_kwargs.pop('sigma_gbm')
        c.gbm_kwargs.update(
            mu=[0.05, 0.03, 0.07],
            sigma_gbm=[0.2, 0.15, 0.25],
        )
    elif opts.stochastic_model == 'Gaussian':
        c.gbm_kwargs.clear()

    c.stochastic_model = opts.stochastic_model
    opts.dump = None

    # Validate dataset options.
    try:
        if opts.stochastic_model == "GBM":
            dataset_obj = dataset_gbm.GBMGenerativeDataset(**c.gbm_kwargs)
        elif opts.stochastic_model == "Heston":
            dataset_obj = dataset_gbm.HestonGenerativeDataset(**c.gbm_kwargs)
        elif opts.stochastic_model == "MJD":
            dataset_obj = dataset_gbm.MJDGenerativeDataset(**c.gbm_kwargs)
        elif opts.stochastic_model == "MarketData":
            dataset_obj = dataset_gbm.RealMarketDataset(**c.gbm_kwargs)
        elif opts.stochastic_model == "HistoricalData":
            dataset_obj = dataset_gbm.HistoricalMarketDataset(**c.gbm_kwargs)
        elif opts.stochastic_model == "CorrelatedGBMGenerativeDataset":
            dataset_obj = dataset_gbm.CorrelatedGBMGenerativeDataset(**c.gbm_kwargs)
        elif opts.stochastic_model == "Gaussian":
            dataset_obj = dataset_gbm.STDGaussianGenerativeDataset()
        dataset_name = dataset_obj.name
        c.dataset_kwargs.dataset_keep_percentage = opts.dataset_keep_percentage
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = int(len(dataset_obj) * opts.dataset_keep_percentage)
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet1D', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])


    assert opts.precond == 'edm'
    c.network_kwargs.class_name = 'training.networks.EDMPrecond'
    c.loss_kwargs.class_name = 'training.loss.EDMLoss'
    c.loss_kwargs.update(consistency_batch_size_per_gpu=opts.consistency_batch_size // dist.get_world_size())
    # whether to use weight for the consistency terms
    c.loss_kwargs.update(with_weight=opts.with_weight)
    # whether to use gradient for the consistency terms
    c.loss_kwargs.update(with_grad=opts.with_grad)
    c.loss_kwargs.update(num_consistency_steps=opts.num_consistency_steps)
    c.loss_kwargs.update(num_primes=opts.num_primes)
    c.loss_kwargs.update(consistency_coeff=opts.consistency_coeff)
    c.loss_kwargs.update(sigma_data=1.0)

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        # add a random string of length 5 to run_dir
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}-{random_string}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    del c.dataset_kwargs.dataset_keep_percentage
    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------