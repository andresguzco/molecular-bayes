import os
import time
import wandb
import torch
import argparse
import numpy as np

from pathlib import Path
from contextlib import suppress
from torch_geometric.loader import DataLoader

from laplace import Laplace
from laplace.curvature import CurvlinopsEF

from timm.utils import ModelEmaV2, NativeScaler
from timm.scheduler import create_scheduler
from ocpmodels.common.registry import registry

import nets
from .parser import get_args_parser
from Pipeline import (
    train_one_epoch, 
    evaluate, 
    compute_stats, 
    create_optimizer,
    FileLogger, 
    calibration_plot
    )
from datasets.QM9 import QM9


ModelEma = ModelEmaV2

def main(args):
    # Set up log file
    _log = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    _log.info(args)
    wandb.login()

    # Set up weights and biases connection
    run = wandb.init(
    project="equiformer_v2",
    config=args,
    )
    
    # Initialise seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ''' Dataset '''
    train_dataset = QM9(args.data_path, 'train', feature_type=args.feature_type)
    val_dataset   = QM9(args.data_path, 'valid', feature_type=args.feature_type)
    test_dataset  = QM9(args.data_path, 'test', feature_type=args.feature_type)


    ############################
    # Subsampling 
    ############################
    if args.prototyping:
        train_dataset = train_dataset[torch.randint(0, len(train_dataset), (100,))]
        val_dataset = val_dataset[torch.randint(0, len(val_dataset), (50,))]
        test_dataset = test_dataset[torch.randint(0, len(test_dataset), (50,))]
    ############################

    _log.info('Training set mean: {}, std:{}'.format(
        train_dataset.mean(args.target), train_dataset.std(args.target)))
    # calculate dataset stats
    task_mean, task_std = 0, 1
    if args.standardize:
        task_mean, task_std = train_dataset.mean(args.target), train_dataset.std(args.target)
    norm_factor = [task_mean, task_std]
    
    # Initialise seeds for the dataset
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Specify the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ''' Network '''
    create_model = registry.get_model_class(args.model_name)
    model = create_model(0, 0, 0)
    _log.info(model)
    model = model.to(device)
    
    # EMA model
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)

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
    # Setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
    
    ''' Data Loader '''
    batch_size = 5 if args.prototyping else args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=args.workers, pin_memory=args.pin_mem, 
        drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    ''' Compute stats '''
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return
    
    best_epoch, best_train_err, best_val_err, best_test_err = 0, float('inf'), float('inf'), float('inf')
    best_ema_epoch, best_ema_val_err, best_ema_test_err = 0, float('inf'), float('inf')
    
    for epoch in range(args.epochs):
        
        epoch_start_time = time.perf_counter()
        
        lr_scheduler.step(epoch)
        
        train_err = train_one_epoch(model=model, criterion=criterion, norm_factor=norm_factor,
            target=args.target, data_loader=train_loader, optimizer=optimizer,
            device=device, epoch=epoch, model_ema=model_ema, 
            amp_autocast=amp_autocast, loss_scaler=loss_scaler,
            print_freq=args.print_freq, logger=_log, wandb=run, model_type=args.model_type)
        
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
                    }, 
                    os.path.join(os.getcwd(), f'checkpoints/Checkpoint_{args.model_type}.pth.tar'))
    
    if args.model_type == "Laplace":

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

        # Initialise the model for the laplace approximation
        la = Laplace(
            model,
            'regression', 
            hessian_structure='full', 
            subset_of_weights='all', 
            backend=CurvlinopsEF
            )
        
        # Fit the Laplace Approximation
        _log.info("Fitting Laplace.")
        bayes_start_time = time.perf_counter()
        la.fit(
            train_loader, 
            is_equiformer=True, 
            target=args.target
            )
        _log.info(f"Laplace fitted. Time taken: {time.perf_counter() - bayes_start_time:.2f}")

        # Optimise the prior precision
        _log.info(f"Optimising Prior Precision")
        la.optimize_prior_precision_base(
            method='marglik',
            pred_type='nn',
            val_loader=val_loader,
            loss=criterion,
            verbose=True
            )
        _log.info(f"Prior precision optimised. Time taken: {time.perf_counter() - bayes_start_time:.2f}")

        # Train Laplace Approximation
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
                        }, 
                        os.path.join(os.getcwd(), f'checkpoints/Checkpoint_{args.model_type}.pth.tar'))
                
        _log.info(f"Evaluating model.")

    # Evaluate the model
    _log.info(f"Evaluating model.")

    model.eval()
    y = torch.zeros(len(train_dataset.data.y[:, args.target]))
    pred = torch.zeros(len(train_dataset.data.y[:, args.target]))

    with torch.no_grad():
        for i, data in enumerate(train_loader):
            batch_start_time = time.perf_counter()

            y[i * args.batch_size: (i + 1) * args.batch_size] = data.y[:, args.target].flatten()
            data = data.to(device)

            if args.model_type == "Laplace":
                pred_i, _ = la(data, pred_type='nn', link_approx='mc')
            else:
                pred_i = model(data)

            pred[i * args.batch_size: (i + 1) * args.batch_size] = pred_i.cpu()

            if i % 100 == 0:
                _log.info(f"Batch {i} done. Time taken: {time.perf_counter() - batch_start_time:.2f}")

    _log.info(f"Preparing calibration plot.")
    calibration_plot(predictions=pred, observations=y, num_bins=20, wandb=run, logger=_log)

        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Training equivariant networks', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
        
