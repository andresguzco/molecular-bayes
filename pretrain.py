############################
# Imports
############################
import os
import gc
import time
import wandb
import torch
import argparse
import numpy as np

from pathlib import Path
from contextlib import suppress
from torch_geometric.loader import DataLoader

from timm.utils import NativeScaler
from timm.scheduler import create_scheduler
from ocpmodels.common.registry import registry

from Interface.pars_args import get_args_parser
from Pipeline import (
    compute_stats, 
    FileLogger,
    train_one_epoch, 
    evaluate,
    create_optimizer,
    save_checkpoint
    )
from my_datasets.QM9 import QM9
import nets


############################
############################
############################

def main(args):
    ############################
    # Logger
    ############################
    _log = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    _log.info(args)
    wandb.login()
    ############################

    ############################
    # Weights and Biases
    ############################
    args.model_type = "Equiformer" if args.model_name == "equiformer_v2" else "MPNN"
    run = wandb.init(
    project="3D_Bayes",
    config=args,
        name=f"Pre - {args.model_type} - {args.dataset_size // 1000}k - Target {args.target}",
    )
    ############################    

    ############################    
    # Set random state 
    ############################    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ############################    
    
    ############################
    # Dataset 
    ############################
    # general_dataset = QM9(args.data_path, 'train', feature_type=args.feature_type)
    # val_dataset     = QM9(args.data_path, 'valid', feature_type=args.feature_type)
    # test_dataset    = QM9(args.data_path, 'test', feature_type=args.feature_type)
    general_dataset = QM7b(args.data_path, 'train')
    val_dataset     = QM7b(args.data_path, 'valid')
    test_dataset    = QM7b(args.data_path, 'test')
    ############################

    ############################
    # Subsampling 
    ############################
    if len(args.target) > 1:
        _, topk_indices = torch.topk(general_dataset.data.y[:, args.target[0]], args.exclusion)

        mask = torch.ones(general_dataset.data.y.size(0), dtype=bool)
        mask[topk_indices] = False

        remaining_indices = torch.nonzero(mask, as_tuple=False).squeeze()

        indices = remaining_indices[torch.randperm(remaining_indices.size(0))[:args.dataset_size]].tolist()

    else:

        indices = torch.randperm(general_dataset.data.y.size(0))[:args.dataset_size].tolist()

    train_dataset = general_dataset[:args.dataset_size].copy()
    train_dataset.data, train_dataset.slices = QM9.collate([general_dataset[i] for i in indices])

   ############################

    ############################
    # Calculate stats
    ############################
    _log.info(f'Training set mean: {train_dataset.mean(args.target):.4f}, std:{train_dataset.std(args.target):.4f}')

    task_mean, task_std = [0 for _ in range(len(args.target))], [1 for _ in range(len(args.target))]
    if args.standardize:
        task_mean = [train_dataset.mean(i) for i in args.target]
        task_std = [train_dataset.std(i) for i in args.target]
    norm_factor = [task_mean, task_std]
    ############################

    ############################
    # Specify the device
    ############################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ############################

    ############################
    # Network creator and info
    ############################
    encoder_creator = registry.get_model_class(args.model_name)
    encoder = encoder_creator(0, 0, 0) if args.model_name == "equiformer_v2" else encoder_creator()
    
    decoder_type = "predictive_layer_2" if args.model_name == "equiformer_v2" else "MLP"
    decoder = torch.nn.ModuleList([registry.get_model_class(decoder_type)() for _ in range(len(args.target))])

    setattr(encoder, 'name', args.model_name)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    if args.resume_training:
        state_dict = torch.load(
                os.getcwd()+f"/checkpoints/Checkpoint_{args.model_type}_{args.dataset_size // 1000}_{args.encoder_target}.pth.tar"
            )

        init_epoc = state_dict['epoch']
        encoder.load_state_dict(state_dict['model_state_dict'])
        decoder.load_state_dict(state_dict['decoder_state_dict'])
    else:
        init_epoc = 0
    ############################
    _log.info(encoder)
    _log.info(decoder)
    ############################

    ############################
    # Number of parameters
    ############################
    _log.info(f"Encoder of params: {sum(p.numel() for p in encoder.parameters())}")
    _log.info(f"Decoder of params: {sum(p.numel() for head in decoder for p in head.parameters())}")
    ############################
    
    ############################
    # Criterion
    ############################
    optimizer = create_optimizer(args, encoder, decoder)
    scheduler, _ = create_scheduler(args, optimizer)
    
    if args.resume_training:
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        scheduler.load_state_dict(state_dict['scheduler_state_dict'])

    criterion = None

    if args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'l2':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError
    ############################

    ############################
    # AMP (from timm)
    ############################
    amp_autocast = suppress
    loss_scaler = None
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
    ############################
     
    ############################
    # Data Loader 
    ############################
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
            shuffle=True, num_workers=args.workers, pin_memory=args.pin_mem, 
            drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    ############################

    counter, best_epoch, best_train_loss, best_val_loss, best_test_loss = 0, 0, float('inf'), float('inf'), float('inf')

    for epoch in range(init_epoc, args.epochs): # init_epoc, args.epochs):

        epoch_start_time = time.perf_counter()
        scheduler.step(epoch)

        train_err, train_loss = train_one_epoch(
            encoder=encoder,
            decoder=decoder,
            criterion=criterion, 
            norm_factor=norm_factor,
            target=args.target,
            data_loader=train_loader, 
            optimizer=optimizer,
            device=device, 
            epoch=epoch, 
            amp_autocast=amp_autocast, 
            loss_scaler=loss_scaler,
            print_freq=args.print_freq, 
            logger=_log
        )

        val_err, val_loss = evaluate(encoder, decoder, norm_factor, args.target, val_loader, device, amp_autocast)
        test_err, test_loss = evaluate(encoder, decoder, norm_factor, args.target, test_loader, device, amp_autocast)

        # record the best results
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            best_train_loss = train_loss
            best_epoch = epoch
            counter = 0
            save_checkpoint(encoder, decoder, optimizer, scheduler, epoch, train_loss, args)
        else:
            counter += 1

        info_str = f'Epoch: [{epoch}] Target: [{args.target}] train loss: {train_loss:.5f}, '
        info_str += f'val MAE: {val_err:.5f}, test MAE: {test_err:.5f}, '
        info_str += f'Time: {time.perf_counter() - epoch_start_time:.2f}s.\n'
        _log.info(info_str)

        info_str = f'Best -- epoch={best_epoch}, train MAE: {best_train_loss:.5f}, '
        info_str += f'val MAE: {best_val_loss:.5f}, test MAE: {best_test_loss:.5f}.\n'
        _log.info(info_str)

        wandb.log({
            # "MAE Train": train_err,
            "Loss Train": train_loss,
            # "MAE Test": test_err,
            "Loss Test": test_loss,
            # "MAE Val": val_err,
            "Loss Val": val_loss
            },
            step=epoch
        )

        if counter >= args.patience:
            _log.info(f"Early stopping at epoch {epoch}.")
            save_checkpoint(encoder, decoder, optimizer, scheduler, epoch, train_loss, args)
            break
    
    _log.info(f"Pretraining complete.")
    run.finish()

############################
############################
############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training equivariant networks', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
