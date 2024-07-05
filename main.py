############################
# Imports
############################
import gc
import wandb
import torch
import argparse
import numpy as np

from pathlib import Path
from contextlib import suppress
from torch_geometric.loader import DataLoader

from timm.utils import NativeScaler
from ocpmodels.common.registry import registry

from Interface.pars_args import get_args_parser
from Pipeline import (
    compute_stats, 
    FileLogger, 
    model_trainer,
    BayesianOptimiser
    )
from datasets.QM9 import QM9
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
    run = wandb.init(
    project="3D_Bayes",
    config=args,
        name=f"{args.model_type}, Seed={args.seed}" if args.model_type != "Laplace" else f"{args.model_type}, Seed={args.seed}, L=M={args.lmax}",
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
    general_dataset = QM9(args.data_path, 'train', feature_type=args.feature_type)
    test_dataset    = QM9(args.data_path, 'test', feature_type=args.feature_type)
    ############################

    ############################
    # Subsampling 
    ############################
    train_idx = 100
    stop_idx = 1000
    indices = torch.randperm(len(general_dataset)).tolist()

    train_indices =  indices[:train_idx]
    virtual_indices =  indices[train_idx:train_idx + stop_idx]

    if args.prototyping:
        virtual_library = general_dataset[:len(train_indices)].copy()
        virtual_library.data, virtual_library.slices = QM9.collate([general_dataset[i] for i in virtual_indices])

        train_dataset = general_dataset[:len(virtual_indices)].copy()
        train_dataset.data, train_dataset.slices = QM9.collate([general_dataset[i] for i in train_indices])

   ############################

    ############################
    # Calculate stats
    ############################
    _log.info(f'Training set mean: {train_dataset.mean(args.target)}, std:{train_dataset.std(args.target)}')

    task_mean, task_std = 0, 1
    if args.standardize:
        task_mean, task_std = train_dataset.mean(args.target), train_dataset.std(args.target)
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
    create_model = registry.get_model_class(args.model_name)
    model = create_model(
        num_atoms=0, bond_feat_dim=0, num_targets=0, lmax_list=[args.lmax], mmax_list=[args.lmax]
    ) if args.model_type == "Laplace" else create_model()
    _log.info(model)
    ############################

    ############################
    # EMA model
    ############################
    #model_ema = None
    #if args.model_ema:
    #    model_ema = ModelEma(
    #        model,
    #        decay=args.model_ema_decay,
    #        device='cpu' if args.model_ema_force_cpu else None)
    ############################

    ############################
    # Number of parameters
    ############################
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info(f'Number of params: {n_parameters}')
    ############################
    
    ############################
    # Criterion
    ############################
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
    batch_size = 10 if args.prototyping else args.batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    ############################

    ############################
    # Bayesian Optimisation Loop
    ############################   
    bayesOpt = BayesianOptimiser(model_type=args.model_type, alpha='EI')
    best_val = -np.inf

    for iter in range(args.iterations):
        ############################
        # Step 1: Train surrogate model
        ############################
        trained_model = model_trainer(
            model=model, criterion=criterion, 
            norm_factor=norm_factor, target=args.target, train_data=train_dataset,
            test_loader=test_loader, 
            device=device, epochs=args.epochs, 
            amp_autocast=amp_autocast, loss_scaler=loss_scaler, 
            print_freq=args.print_freq, logger=_log, wandb=run, 
            model_type=args.model_type, patience=args.patience, batch_size=batch_size, 
            args=args, model_creator=create_model
        )

        ###############################
        # Step 2: Acquisition function 
        ###############################
        idx_query = bayesOpt.query(
            model=trained_model if args.model_type == "Laplace" or args.model_type == "BGNN" else model,
            virtual_loader= DataLoader(virtual_library, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False),
            incumbent_y=torch.max(train_dataset.data.y[:, args.target]),
            device=device,
            norm_factor=norm_factor,
            logger=_log
            )

        _log.info(f"Iter: {iter}. Next query: {idx_query}. Selection: {virtual_library.y[idx_query, args.target]:.2f}. ")
        _log.info(f"Virtual Library Size: {len(virtual_indices)}. Train Library Size: {len(train_indices)}.\n")
        ############################
        # Step 3: Update and log 
        ############################
        if virtual_library.data.y[idx_query, args.target] > best_val:
            best_val = virtual_library.data.y[idx_query, args.target]
    
        run.log(data={"Selection": virtual_library.data.y[idx_query, args.target]})
        run.log(data={"Best": best_val})

        run.log(data={
            "Virtual Max": virtual_library.data.y[:, args.target].max(),
            "Train Max": train_dataset.data.y[:, args.target].max()
        })

        del trained_model, train_dataset, virtual_library
        torch.cuda.empty_cache()
        gc.collect()
        
        train_indices.append(virtual_indices[idx_query])
        train_dataset = general_dataset[:len(train_indices)].copy()
        train_dataset.data, train_dataset.slices = QM9.collate([general_dataset[i] for i in train_indices])

        del virtual_indices[idx_query]
        virtual_library = general_dataset[:len(virtual_indices)].copy()
        virtual_library.data, virtual_library.slices = QM9.collate([general_dataset[i] for i in virtual_indices])
        
    ############################
    # End of loop
    ############################

############################
############################
############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training equivariant networks', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
