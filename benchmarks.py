############################
# Imports
############################
import os
import gc
import wandb
import torch
import random
import argparse
import numpy as np

from rdkit.Chem import MolFromSmiles, AllChem
from pathlib import Path
from contextlib import suppress

from ocpmodels.common.registry import registry

from Interface.pars_args import get_args_parser
from Pipeline import ( 
    FileLogger, 
    BayesianOptimiser,
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
        name=f"BO - {args.model_name} - {args.model_type}"
    )
    ############################

    ############################
    # Set random state
    ############################
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    ############################

    ############################
    # Specify the device
    ############################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ############################
    
    ############################
    # Dataset 
    ############################
    for i, target in enumerate(args.target):
        args.target[i] = int(target)

    # general_dataset = QM9(args.data_path, 'train', feature_type=args.feature_type)
    # general_dataset = QM7b(args.data_path)
    general_dataset = GEOM(args.data_path)
    # general_dataset = GEOM(args.data_path)
    ############################
    
    ############################
    # Subsampling
    ############################    
    virtual_indices = [i for i in range(len(general_dataset))]
    if args.subsampling:
        virtual_indices = random.sample(virtual_indices, k=args.subsampling_size + args.initial_sample)
        training_indices = random.sample(virtual_indices, k=args.initial_sample)
        virtual_indices = [i for i in virtual_indices if i not in training_indices]
    else:
        virtual_indices = [i for i in range(len(general_dataset))] 
        training_indices = random.sample(virtual_indices, k=args.initial_sample)
        virtual_indices = [i for i in virtual_indices if i not in training_indices]

    if args.model_type == 'GP':
        fpgen = AllChem.GetMorganGenerator(radius=int(args.radius), fpSize=2048)

        virtual_library = general_dataset[:len(virtual_indices)].copy()
        virtual_library.data, virtual_library.slices = GEOM.collate([general_dataset[i] for i in virtual_indices])
        virtual_X = torch.tensor([fpgen.GetFingerprint(MolFromSmiles(smiles)) for smiles in virtual_library.data.smiles])
        # virtual_Y = virtual_library.data.y[:, args.target[0]]
        virtual_Y = virtual_library.data.y[:]
        virtual_dataset = CustomDataset(virtual_X.float(), virtual_Y)

        train_library = general_dataset[:len(training_indices)].copy()
        train_library.data, train_library.slices = GEOM.collate([general_dataset[i] for i in training_indices])
        train_X = torch.tensor([fpgen.GetFingerprint(MolFromSmiles(smiles)) for smiles in train_library.data.smiles])
        # train_Y = train_library.data.y[:, args.target[0]]
        train_Y = train_library.data.y[:]
        train_dataset = CustomDataset(train_X.float(), train_Y)

    else:        
        # virtual_Y = general_dataset.data.y[virtual_indices, args.target[0]]
        # train_Y = general_dataset.data.y[training_indices, args.target[0]]
        virtual_Y = general_dataset.data.y[virtual_indices]
        train_Y = general_dataset.data.y[training_indices]

    ############################

    ############################

    ############################
    # Retrieve model if necessary
    ############################
    if args.model_type == 'GP':
        model_creator = registry.get_model_class("GP_Tanimoto")
        bayesOpt = BayesianOptimiser(model_type=args.model_type, alpha=args.acquisition_function)
    ############################    

    ############################
    # Initial incumbent
    ############################
    ############################
    incumbent_y = train_Y.max().item()
    ############################
    
    ############################
    for iter in range(args.iterations):
        ############################
        # Step 1: Train surrogate model
        ############################
        if args.model_type == 'GP':

            bayes_decoder = train_gp(
                data=train_dataset,
                lr=args.lr,
                device=device,
                logger=_log,
                epochs=args.epochs
            )
            
            ###############################

            ###############################
            # Step 2: Acquisition function
            ###############################
            virtual_loader = torch.utils.data.DataLoader(
                virtual_dataset, 
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
                logger=_log
                )

            selection = virtual_Y[idx_query].item()

        else:
            idx_query = torch.randint(0, len(virtual_Y), (1,)).item()
            selection = virtual_Y[idx_query].item()

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
            "Virtual Max": virtual_Y.max().item(),
            "Train Max": train_Y.max().item()
        })

        train_Y = torch.cat([train_Y, virtual_Y[idx_query].unsqueeze(0)])
        virtual_Y = torch.cat([virtual_Y[:idx_query], virtual_Y[idx_query + 1:]])

        if args.model_type == 'GP':
            train_X = torch.cat([train_X, virtual_X[idx_query, :].unsqueeze(0)])
            virtual_X = torch.cat([virtual_X[:idx_query, :], virtual_X[idx_query + 1:, :]])

            train_dataset = CustomDataset(train_X.float(), train_Y)
            virtual_dataset = CustomDataset(virtual_X.float(), virtual_Y)

        if selection > virtual_Y.max().item():
            for _ in range(iter, args.iterations):
                run.log(data={
                    "Selection": selection,
                    "Best": incumbent_y,
                    "Virtual Max": virtual_Y.max().item(),
                    "Train Max": train_Y.max().item()
                })
            break
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
