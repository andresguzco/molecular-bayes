import argparse
import datetime
import itertools
import pickle
import subprocess
import wandb
import math
import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import os
from logger import FileLogger
from pathlib import Path

from laplace import Laplace
from laplace.curvature import CurvlinopsEF
from datasets.QM9 import QM9

# AMP
from contextlib import suppress
from timm.utils import NativeScaler

import nets
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import save_checkpoint

from timm.utils import ModelEmaV2
from timm.scheduler import create_scheduler
from optim_factory import create_optimizer

from engine import train_one_epoch, evaluate, compute_stats

# distributed training
import utils

ModelEma = ModelEmaV2


def get_args_parser():
    parser = argparse.ArgumentParser('Training equivariant networks', add_help=False)
    parser.add_argument('--output-dir', type=str, default=None)
    # network architecture
    parser.add_argument('--model-name', type=str, default='equiformer_v2')
    parser.add_argument('--laplace', type=bool, default=False)
    parser.add_argument('--variational', type=bool, default=False)
    parser.add_argument('--input-irreps', type=str, default=None)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--num-basis', type=int, default=32)
    parser.add_argument('--output-channels', type=int, default=1)
    # training hyperparameters
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--epochs-bayes", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.9999, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    # regularization
    parser.add_argument('--drop-path_rate', type=float, default=0.0)
    # optimizer (timm)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    # learning rate schedule parameters (timm)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    # logging
    parser.add_argument("--print-freq", type=int, default=100)
    # task
    parser.add_argument("--target", type=int, default=7)
    parser.add_argument("--data-path", type=str, default='datasets/QM9')
    parser.add_argument('--feature-type', type=str, default='one_hot')
    parser.add_argument('--compute-stats', action='store_true', dest='compute_stats')
    parser.set_defaults(compute_stats=False)
    parser.add_argument('--no-standardize', action='store_false', dest='standardize')
    parser.set_defaults(standardize=True)
    parser.add_argument('--loss', type=str, default='l1')
    # random
    parser.add_argument("--seed", type=int, default=0)
    # data loader config
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    # AMP
    parser.add_argument('--no-amp', action='store_false', dest='amp', 
                        help='Disable FP16 training.')
    parser.set_defaults(amp=False)
    # distributed training parameters
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def filename(bayes=None): 
    if bayes:
        path = f'results/models/{bayes}/Checkpoint.pth.tar'
    else:
        path = f'results/models/Checkpoint{bayes}.pth.tar'
    return path


def main(args):
    if args.distributed:
        utils.init_distributed_mode(args)
    else:
        args.rank = 0
    is_main_process = (args.rank == 0)

    _log = FileLogger(is_master=is_main_process, is_rank0=is_main_process, output_dir=args.output_dir)
    _log.info(args)
    wandb.login()
    run = wandb.init(
    project="equiformer_v2",
    # Track hyperparameters and run metadata
    config=args,
    )
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ''' Dataset '''
    train_dataset = QM9(args.data_path, 'train', feature_type=args.feature_type)
    val_dataset   = QM9(args.data_path, 'valid', feature_type=args.feature_type)
    test_dataset  = QM9(args.data_path, 'test', feature_type=args.feature_type)
    _log.info('Training set mean: {}, std:{}'.format(
        train_dataset.mean(args.target), train_dataset.std(args.target)))
    # calculate dataset stats
    task_mean, task_std = 0, 1
    if args.standardize:
        task_mean, task_std = train_dataset.mean(args.target), train_dataset.std(args.target)
    norm_factor = [task_mean, task_std]
    
    # since dataset needs random 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ''' Network '''
    create_model = registry.get_model_class(args.model_name)
    model = create_model(0, 0, 0)
    _log.info(model)
    model = model.to(device)
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)

    # distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info('Number of params: {}'.format(n_parameters))
    
    ''' Optimizer and LR Scheduler '''
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = None #torch.nn.MSELoss() #torch.nn.L1Loss() # torch.nn.MSELoss() 
    if args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'l2':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError

    ''' AMP (from timm) '''
    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
    
    ''' Data Loader '''
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
                train_dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True
            )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
            sampler=sampler_train, num_workers=args.workers, pin_memory=args.pin_mem, 
            drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
            shuffle=True, num_workers=args.workers, pin_memory=args.pin_mem, 
            drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    ''' Compute stats '''
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return
    
    best_epoch, best_train_err, best_val_err, best_test_err = 0, float('inf'), float('inf'), float('inf')
    best_ema_epoch, best_ema_val_err, best_ema_test_err = 0, float('inf'), float('inf')
    
    for epoch in range(args.epochs):
        
        epoch_start_time = time.perf_counter()
        
        lr_scheduler.step(epoch)

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        train_err = train_one_epoch(model=model, criterion=criterion, norm_factor=norm_factor,
            target=args.target, data_loader=train_loader, optimizer=optimizer,
            device=device, epoch=epoch, model_ema=model_ema, 
            amp_autocast=amp_autocast, loss_scaler=loss_scaler,
            print_freq=args.print_freq, logger=_log, wandb=run, variational=args.variational)
        
        val_err, val_loss = evaluate(model, norm_factor, args.target, val_loader, device, 
            amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log)
        
        test_err, test_loss = evaluate(model, norm_factor, args.target, test_loader, device, 
            amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log)
        wandb.log({
            "Validation Error": val_err,
            "Test Error": test_err,
        })
        
        # record the best results
        if val_err < best_val_err:
            best_val_err = val_err
            best_test_err = test_err
            best_train_err = train_err
            best_epoch = epoch

        info_str = 'Epoch: [{epoch}] Target: [{target}] train MAE: {train_mae:.5f}, '.format(
            epoch=epoch + 1, target=args.target, train_mae=train_err)
        info_str += 'val MAE: {:.5f}, '.format(val_err)
        info_str += 'test MAE: {:.5f}, '.format(test_err)
        info_str += 'Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)
        _log.info(info_str)
        
        info_str = 'Best -- epoch={}, train MAE: {:.5f}, val MAE: {:.5f}, test MAE: {:.5f}\n'.format(
            best_epoch + 1, best_train_err, best_val_err, best_test_err)
        _log.info(info_str)
        
        # evaluation with EMA
        if model_ema is not None:
            ema_val_err, _ = evaluate(model_ema.module, norm_factor, args.target, val_loader, device, 
                amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log)
            
            ema_test_err, _ = evaluate(model_ema.module, norm_factor, args.target, test_loader, device, 
                amp_autocast=amp_autocast, print_freq=args.print_freq, logger=_log)
            
            # record the best results
            if (ema_val_err) < best_ema_val_err:
                best_ema_val_err = ema_val_err
                best_ema_test_err = ema_test_err
                best_ema_epoch = epoch
    
            info_str = 'Epoch: [{epoch}] Target: [{target}] '.format(
                epoch=epoch + 1, target=args.target)
            info_str += 'EMA val MAE: {:.5f}, '.format(ema_val_err)
            info_str += 'EMA test MAE: {:.5f}, '.format(ema_test_err)
            info_str += 'Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)
            _log.info(info_str)
            
            info_str = 'Best EMA -- epoch={}, val MAE: {:.5f}, test MAE: {:.5f}\n'.format(
                best_ema_epoch + 1, best_ema_val_err, best_ema_test_err)
            _log.info(info_str)

        if epoch % 10 == 0:
            torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'loss': train_err,
                    }, os.path.join(os.getcwd(), filename(bayes='laplace' if args.laplace else 'VI' if args.variational else None)))
    
    if args.laplace is True:
        # We isolate the last layer for the Laplace Approximation
        last_layer_params = sum(p.numel() for p in model.energy_block.so3_linear_2.parameters())
        print(f'SO3 Linear Layer Parameters: {last_layer_params}')

        for parameter in model.sphere_embedding.parameters():
            parameter.requires_grad = False
            
        for block in model.blocks:
            for parameter in block.parameters():
                parameter.requires_grad = False

        for parameter in model.norm.parameters():
            parameter.requires_grad = False

        for parameter in model.edge_degree_embedding.parameters():
            parameter.requires_grad = False
        
        for parameter in model.energy_block.so3_linear_1.parameters():
            parameter.requires_grad = False

        for parameter in model.energy_block.gating_linear.parameters():
            parameter.requires_grad = False
        
        # for parameter in model.energy_block.parameters():
        #     parameter.requires_grad = False

        la = Laplace(
            model,
            'regression', 
            hessian_structure='full', 
            subset_of_weights='all', 
            backend=CurvlinopsEF
            )
        
        _log.info("Fitting Laplace.")
        bayes_start_time = time.perf_counter()
        la.fit(
            train_loader, 
            is_equiformer=True, 
            target=args.target
            )
        _log.info(f"Laplace fitted. Time taken: {time.perf_counter() - bayes_start_time:.2f}")

        _log.info(f"Optimising Prior Precision")
        la.optimize_prior_precision_base(
            method='marglik',
            pred_type='nn',
            val_loader=val_loader,
            loss=criterion,
            verbose=True
            )
        _log.info(f"Prior precision optimised. Time taken: {time.perf_counter() - bayes_start_time:.2f}")

        log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)

        for epoch in range(args.epochs_bayes):
            epoch_start_time = time.perf_counter()

            hyper_optimizer.zero_grad()

            neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
            neg_marglik.backward()
            hyper_optimizer.step()

            info_str = 'Epoch: [{epoch}] Target: [{target}] '.format(
                epoch=epoch + 1, target=args.target)
            info_str += f'Negative Marginal Likelihood: {neg_marglik:.5f}, '
            info_str += 'Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)
            _log.info(info_str)
            wandb.log({"Negative Marginal Log-likelihood": neg_marglik})

            if epoch % 10 == 0:
                torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': la.state_dict(),
                        'optimizer_state_dict': hyper_optimizer.state_dict(),
                        'loss': neg_marglik,
                        }, os.path.join(os.getcwd(), filename(bayes='laplace')))
                
        _log.info(f"Evaluating model.")


        model.eval()
        y = torch.zeros(len(train_dataset.data.x))
        pred = torch.zeros(len(train_dataset.data.x))

        with torch.no_grad():
            for i, data in enumerate(train_loader):
                batch_start_time = time.perf_counter()

                y[i * args.batch_size: (i + 1) * args.batch_size] = data.y[:, args.target]
                data = data.to(device)

                pred_i, _ = la(data, pred_type='nn', link_approx='mc')
                pred[i * args.batch_size: (i + 1) * args.batch_size] = pred_i.cpu()

                if i % 100 == 0:
                    _log.info(f"Batch {i} done. Time taken: {time.perf_counter() - batch_start_time}:.2f")

        _log.info(f"Preparing calibration plot.")
        utils.calibration_plot(predictions=pred, observations=y, num_bins=20, wandb=run)


    else:
        _log.info(f"Evaluating model.")

        model.eval()
        y = torch.zeros(len(train_dataset.data.y[:, args.target]))
        pred = torch.zeros(len(train_dataset.data.y[:, args.target]))

        with torch.no_grad():
            for i, data in enumerate(train_loader):
                batch_start_time = time.perf_counter()
                y[i * args.batch_size: (i + 1) * args.batch_size] = data.y[:, args.target].flatten()
                data = data.to(device)

                pred_i = model(data)
                pred[i * args.batch_size: (i + 1) * args.batch_size] = pred_i.cpu()

                if i % 100 == 0:
                    _log.info(f"Batch {i} done. Time taken: {time.perf_counter() - batch_start_time:.2f}")

        _log.info(f"Preparing calibration plot.")
        utils.calibration_plot(predictions=pred, observations=y, num_bins=20, wandb=run, logger=_log)

        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Training equivariant networks', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
        