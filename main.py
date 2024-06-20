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

from timm.utils import ModelEmaV2, NativeScaler
from timm.scheduler import create_scheduler
from ocpmodels.common.registry import registry

from Interface.pars_args import get_args_parser
from Pipeline import (
    compute_stats, 
    create_optimizer,
    FileLogger, 
    model_trainer,
    BayesianOptimiser,
    reset_model,
    plotter
    )
from datasets.QM9 import QM9
import nets


ModelEma = ModelEmaV2

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
    name=args.model_type
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
    val_dataset   = QM9(args.data_path, 'valid', feature_type=args.feature_type)
    test_dataset  = QM9(args.data_path, 'test', feature_type=args.feature_type)

    ############################
    # Subsampling 
    ############################
    train_idx = 49
    stop_idx = train_idx + 100000
    indices = torch.randperm(len(general_dataset)).tolist()

    proto_indices =  indices[:train_idx]
    virtual_indices =  indices[train_idx:stop_idx]
    if args.prototyping:
        virtual_library = general_dataset[virtual_indices]
        train_dataset = general_dataset[proto_indices]
        val_dataset = val_dataset[:99]
        test_dataset = test_dataset[:99]
    ############################

    _log.info(f"Observation [0]: {general_dataset[0]}")
    _log.info(f"Observation [1]: {general_dataset[1]}")
    _log.info(f"Nodes [0]: {general_dataset[0].x}")
    _log.info(f"Nodes [1]: {general_dataset[1].x}")
    _log.info(f"Edges [0]: {general_dataset[0].edge_attr}")
    _log.info(f"Edges [1]: {general_dataset[1].edge_attr}")

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
    # Initialise seeds for the dataset
    ############################
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ############################
    
    ############################
    # Specify the device
    ############################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ############################

    ############################
    # Network 
    ############################
    create_model = registry.get_model_class(args.model_name)
    model = create_model(0, 0, 0) if args.model_type == "Laplace" else create_model()
    _log.info(model)
    model = model.to(device)
    ############################

    ############################
    # EMA model
    ############################
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)
    ############################

    ############################
    # Number of parameters
    ############################
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info(f'Number of params: {n_parameters}')
    ############################
    
    ############################
    # Optimizer and LR Scheduler
    ############################
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
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
    batch_size = 5 if args.prototyping else args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    ############################
    # Compute stats 
    ############################
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius,
                       logger=_log, print_freq=args.print_freq)
        return


    ############################
    # Bayesian Optimisation Loop
    ############################    
    virtual_max = virtual_library.data.y[:, args.target].max()
    general_max = general_dataset.data.y[:, args.target].max()
    train_max = train_dataset.data.y[:, args.target].max()

    vSelection = torch.zeros(args.iterations)
    vIncumbent = torch.zeros(args.iterations)
    vIncumbent[0] = train_max

    for iter in range(args.iterations):

        virtual_loader = DataLoader(virtual_library, batch_size=batch_size, 
        shuffle=True, num_workers=0, pin_memory=False)

        ############################
        # Step 1: Train surrogate model
        ############################
        trained_model = model_trainer(
            scheduler=lr_scheduler, model=model, criterion=criterion, 
            norm_factor=norm_factor, target=args.target, train_loader=train_loader,
            val_loader= val_loader, test_loader=test_loader, optimizer=optimizer, 
            device=device, epochs=args.epochs, model_ema=model_ema, 
            amp_autocast=amp_autocast, loss_scaler=loss_scaler, 
            print_freq=args.print_freq, logger=_log, wandb=run, 
            model_type=args.model_type)

        ###############################
        # Step 2: Acquisition function 
        ###############################
        bayesOpt = BayesianOptimiser(model_type=args.model_type)

        ############################
        # Step 3: Query 
        ############################
        idx_query = bayesOpt.query(
            model=trained_model if args.model_type == "Laplace" else model,
            virtual_loader=virtual_loader,
            incumbent_y=vIncumbent[iter],
            device=device,
            logger=_log
            )

        _log.info(f"Iter: {iter}. Next query: {idx_query}. Selection: {virtual_library[idx_query, args.target]:.2f}.\n")

        ############################
        # Step 4: Update 
        ############################
        vSelection[iter] = virtual_library.data.y[idx_query, args.target]

        if iter < args.iterations - 1:
            vIncumbent[iter + 1] = vSelection[iter] if vSelection[iter] > vIncumbent[iter] else vIncumbent[iter]

        del train_loader, virtual_loader, trained_model, train_dataset, virtual_library
        torch.cuda.empty_cache()
        gc.collect() 

        train_dataset = general_dataset[sorted(proto_indices + [idx_query])]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=0, pin_memory=False)

        del virtual_indices[idx_query]
        virtual_library = general_dataset[virtual_indices]

        ############################
        # Reset the model
        ############################
        reset_model(model)

    plotter(vIncumbent, vSelection, train_max, virtual_max, general_max, wandb=run)

    ############################
    # End of loop
    ############################

############################
############################
############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training equivariant networks',
                                      parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)