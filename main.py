############################
# Imports
############################
import os
import gc
import re
import wandb
import torch
import random
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
    FileLogger, 
    BayesianOptimiser,
    train_laplace,
    train_gp
    )

from my_datasets.QM9 import QM9
from my_datasets.QM7b import QM7b
from my_datasets.GEOM import GEOM
from my_datasets.Molecule_Net import Molecule_Net
import nets


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_x = self.X[idx, :]
        sample_y = self.Y[idx]
        return sample_x, sample_y

def custom_collate_fn(batch):
    batch_x = [item[0] for item in batch]
    batch_y = [item[1] for item in batch]

    batch_x = torch.stack(batch_x, dim=0)
    batch_y = torch.stack(batch_y, dim=0)
    return batch_x, batch_y

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
        name=f"BO - {args.model_name} - {args.model_type} - {args.dataset_size // 1000}k - Target {args.encoder_target}"
    )
    ############################

    ############################
    # Specify the device
    ############################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ############################  

    ############################
    # Set random state
    ############################
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    ############################
    
    ############################
    # Dataset 
    ############################
    for i, target in enumerate(args.target):
        args.target[i] = int(target)

    general_dataset = QM9(f"my_{args.data_path}", split='train')
    # general_dataset.data.y = general_dataset.data.y.unsqueeze(1)
    _log.info(general_dataset)
    ############################

    ############################
    # Subsampling
    ############################    
    _, topk_indices = torch.topk(general_dataset.data.y[:, args.target[0]], args.exclusion)
    # _, topk_indices = torch.topk(general_dataset.data.y[args.target[0]], args.exclusion)
    
    mask = torch.ones(general_dataset.data.y.size(0), dtype=bool)
    mask[topk_indices] = False
    
    remaining_indices = torch.nonzero(mask, as_tuple=False).squeeze()
    
    virtual_indices = remaining_indices[torch.randperm(
        remaining_indices.size(0), generator=torch.Generator().manual_seed(0)
    )].tolist()
    virtual_indices += topk_indices.tolist()
    ############################

    ############################
    # Set random state
    ############################
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    ############################

    ############################
    if args.subsampling:
        virtual_indices = random.sample(virtual_indices, k=args.subsampling_size + args.initial_sample)
    
    general_library = general_dataset[:len(virtual_indices)].copy()
    general_library.data, general_library.slices = QM9.collate([general_dataset[i] for i in virtual_indices])
    _log.info(general_library)
    ############################

    ############################

    ############################
    # Network creator and info
    ############################
    name = "Equiformer" if args.model_name == "equiformer_v2" else "MPNN"
    path = os.getcwd()+f"/checkpoints/Checkpoint_{name}_{args.dataset_size // 1000}_{args.encoder_target}.pth.tar"
    
    encoder_creator = registry.get_model_class(args.model_name)
    encoder = encoder_creator(0, 0, 0) if args.model_name == "equiformer_v2" else encoder_creator()
    encoder.load_state_dict(torch.load(path)["model_state_dict"])
    
    if args.model_type == 'Laplace':
        decoder_name = "predictive_layer_2" if args.model_name == "equiformer_v2" else "MLP"
        decoder = registry.get_model_class(decoder_name)()
        setattr(decoder, 'name', args.model_name)
        setattr(decoder, 'model_type', args.model_type)
        setattr(decoder, 'size', args.dataset_size // 1000)
        setattr(decoder, 'target', args.encoder_target)
        
        if args.encoder_target == "Single":
            state_dict = torch.load(path)["decoder_state_dict"]
            corrected_state_dict = {}
            for key, value in state_dict.items():
                new_key = re.sub(r'^0\.', '', key)
                corrected_state_dict[new_key] = value

            decoder.load_state_dict(corrected_state_dict)

        decoder = decoder.to(device)
        _log.info(decoder)


    setattr(encoder, 'name', args.model_name)
    _log.info(encoder)
    encoder = encoder.to(device)
    ############################ 

    ############################
    # Number of parameters
    ############################
    for param in encoder.parameters():
        param.requires_grad = False

    if args.model_type == 'Laplace':
        n_parameters = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        _log.info(f'Number of params: {n_parameters}')
        ############################

        ############################
        # Criterion
        ############################
        optimizer = torch.optim.AdamW(
            decoder.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        # scheduler = create_scheduler(args, optimizer)

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
    # loss_scaler = None
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        # loss_scaler = NativeScaler()
    ############################

    ############################
    # Bayesian Optimisation Loop
    ############################
    bayesOpt = BayesianOptimiser(model_type=args.model_type, alpha=args.acquisition_function)
    # best_val = -np.inf
    ############################    

    ############################
    # Caching features
    ############################
    _log.info("Caching features.")
    encoder.eval()
    loader = DataLoader(
        general_library,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers, 
        pin_memory=args.pin_mem
    )

    embedded_molecules = list()

    for i, data in enumerate(loader):
        data = data.to(device)

        with amp_autocast() and torch.no_grad():
            if args.model_name == "equiformer_v2":
                embedding, _ =  encoder(data)
            elif args.model_name == "MPNN_Benchmark":
                embedding = encoder(data)

        embedded_molecules.append(embedding.detach().cpu())
    
    _log.info(f"Features cached. List: {len(embedded_molecules)}")

    X = torch.cat(embedded_molecules, dim=0)
    Y = general_dataset.data.y[:, args.target[0]]
    Y = Y.squeeze()

    total_indices = list(range(X.shape[0]))
    train_indices = random.sample(total_indices, args.initial_sample)
    virtual_indices = list(set(total_indices) - set(train_indices))

    train_X = torch.stack([X[idx, :] for idx in train_indices])
    train_Y = torch.stack([Y[idx].unsqueeze(0) for idx in train_indices])
    train_dataset = CustomDataset(train_X, train_Y)

    virtual_X = torch.stack([X[idx, :] for idx in virtual_indices])
    virtual_Y = torch.stack([Y[idx].unsqueeze(0) for idx in virtual_indices])
    virtual_library = CustomDataset(virtual_X, virtual_Y)
    ############################

    ############################
    # Calculate stats
    ############################
    task_mean, task_std = 0, 1
    if args.standardize:
        task_mean = train_dataset.Y.mean().item()
        task_std = train_dataset.Y.std().item()
    norm_factor = [task_mean, task_std]

    _log.info(f'Training set mean: {task_mean:.4f}, std:{task_std:.4f}')
    ############################


    ############################
    # Initial incumbent
    ############################
    ############################
    incumbent_y = train_dataset.Y.max().item()
    ############################
    
    ############################
    for iter in range(args.iterations):
        ############################
        # Step 1: Train surrogate model
        ############################
        if args.model_type == 'Laplace':            
            decoder.train()
            batch_size = args.batch_size if args.batch_size < len(train_dataset) else 5

            train_dataset.Y = (train_dataset.Y - task_mean) / task_std

            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=args.workers, 
                pin_memory=args.pin_mem, 
                collate_fn=custom_collate_fn
            )

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                output = decoder(x)
                loss = criterion(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            bayes_decoder = train_laplace(
                model=decoder, 
                loader=train_loader, 
                criterion=criterion, 
                logger=_log
            )

        elif args.model_type == 'GP':
            bayes_decoder = train_gp(
                data=train_dataset,
                logger=_log, 
                lr=args.lr,
                device=device,
                norm_factor=norm_factor
            )
        elif args.model_type == 'Variational':
            NotImplementedError(f"Variational model is work in progress.")
            bayes_decoder = ...
        else: 
            raise NotImplementedError(f"Model type {args.model_type} not implemented.")
        ###############################

        ###############################
        # Step 2: Acquisition function
        ###############################
        virtual_loader = torch.utils.data.DataLoader(
            virtual_library, 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=args.workers, 
            pin_memory=args.pin_mem,
            collate_fn=custom_collate_fn
        )

        idx_query = bayesOpt.query(
            decoder=bayes_decoder,
            virtual_loader=virtual_loader,
            incumbent_y=incumbent_y,
            device=device,
            logger=_log,
            norm_factor=norm_factor
            )

        selection = virtual_library.Y[idx_query].item()

        _log.info(f"Iter: {iter}. Next query: {idx_query}. Selection: {selection:.2f}.")
        ############################

        ############################
        # Step 3: Update and log 
        ############################
        if incumbent_y < selection:
            incumbent_y = selection

        run.log(data={
            "Selection": selection,
            "Best": incumbent_y,
            "Virtual Max": virtual_library.Y.max().item(),
            "Train Max": train_dataset.Y.max().item()
        })

        train_X = torch.cat([train_X, virtual_X[idx_query, :].unsqueeze(0)])
        train_Y = torch.cat([train_Y, virtual_Y[idx_query].unsqueeze(0)])
        train_dataset = CustomDataset(train_X, train_Y)

        virtual_X = torch.cat([virtual_X[:idx_query, :], virtual_X[idx_query + 1:, :]])

        virtual_Y = torch.cat([virtual_Y[:idx_query], virtual_Y[idx_query + 1:]])
        virtual_library = CustomDataset(virtual_X, virtual_Y)

        if selection > virtual_library.Y.max().item():
            for _ in range(iter, args.iterations):
                run.log(data={
                    "Selection": selection,
                    "Best": incumbent_y,
                    "Virtual Max": virtual_library.Y.max().item(),
                    "Train Max": train_dataset.Y.max().item()
                })
            break


        del virtual_loader
        gc.collect()
        ############################

    ############################
    # End of loop
    ############################
    _log.info(f"BO Completed. Best: {incumbent_y}.")
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
