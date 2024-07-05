############################
# Imports
############################
import gc
import nets
import wandb
import torch
import botorch
import gpytorch
import argparse
import numpy as np

from pathlib import Path
from botorch import fit_gpytorch_mll
from timm.scheduler import create_scheduler
from Interface.pars_args import get_args_parser
from Pipeline import (
    FileLogger, 
    BayesianOptimiser,
    calibration_plot,
    create_optimizer
)

from rdkit.Chem import AllChem
from datasets.QM9 import QM9
from ocpmodels.common.registry import registry





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
        name=f"{args.model_type}, Seed={args.seed}"
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

    if args.standardize:
        task_mean, task_std = train_dataset.mean(args.target), train_dataset.std(args.target)
    norm_factor = [task_mean, task_std]
    ############################

    ############################
    # Data Prep
    ############################
    fpgen = AllChem.GetMorganGenerator(radius=int(args.radius), fpSize=2048)

    X_val = torch.tensor([fpgen.GetFingerprint(data.mol) for data in test_dataset])
    Y_val = test_dataset.y[:, args.target]
    X_train = torch.tensor([fpgen.GetFingerprint(data.mol) for data in train_dataset])
    Y_train = train_dataset.y[:, args.target]
    X_virtual = torch.tensor([fpgen.GetFingerprint(data.mol) for data in virtual_library])
    Y_virtual = virtual_library.y[:, args.target]
    ############################

    ############################
    # Calculate stats
    ############################
    _log.info(f'Training set mean: {train_dataset.mean(args.target)}, std:{train_dataset.std(args.target)}')
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
    ############################

    ############################
    # Bayesian Optimisation Loop
    ############################

    bayesOpt = BayesianOptimiser(model_type="GP", alpha="ES")
    best_val = -np.inf

    for iter in range(args.iterations):

        ############################
        # Step 1: Train surrogate model
        ############################
        GP = GP_trainer(model_gen=create_model, X_train=X_train.float(), Y_train=Y_train.float(), 
                        X_test=X_val.float(), Y_test=Y_val.float(), device=device, logger=_log, 
                        wandb=run, args=args, norm_factor=norm_factor)

        if iter == 0: 
            _log.info(GP)
        ###############################
        # Step 2: Query 
        ############################
        idx_query = bayesOpt.query(
            model=GP,
            virtual_loader=X_virtual.float(),
            incumbent_y=torch.max(Y_train.float()),
            device=device,
            logger=_log
            )

        _log.info(f"Iter: {iter}. Next query: {idx_query}. Selection: {Y_virtual[idx_query]:.2f}.\n")
        _log.info(f"Virtual Library Size: {len(Y_virtual)}. Train Library Size: {len(Y_train)}.\n")
        ############################
        # Step 3: Update 
        ############################
        if Y_virtual[idx_query] > best_val:
            best_val = Y_virtual[idx_query]

        run.log(data={"Selection": virtual_library.data.y[idx_query, args.target]})
        run.log(data={"Best": best_val})

        run.log(data={
            "Virtual Max": Y_virtual.max(),
            "Train Max": Y_train.max()
        })

        X_train = torch.cat([X_train, X_virtual[idx_query].unsqueeze(0)])
        Y_train = torch.cat([Y_train, Y_virtual[idx_query].unsqueeze(0)])
        X_virtual = torch.cat([X_virtual[:idx_query], X_virtual[idx_query + 1:]])
        Y_virtual = torch.cat([Y_virtual[:idx_query], Y_virtual[idx_query + 1:]])

        del GP
        torch.cuda.empty_cache()
        gc.collect()

    ############################
    # End of loop
    ############################


def GP_trainer(model_gen, X_train, Y_train, X_test, Y_test, device, logger, wandb, args, norm_factor):

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = model_gen(X_train, Y_train, likelihood)
    model = model.to(device)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
    mll.train()
    model.train()
    likelihood.train()
    
    botorch.fit_gpytorch_mll(mll)
    
    mll.eval()
    model.eval()
    likelihood.eval()
                                                             
    logger.info(f"\nEvaluating model.\n")                                                                                                                                
                        
    with torch.no_grad(): 
        pred = model(X_test.to(device))  
        pred_mu = pred.mean.detach()
        pred_sigma = torch.sqrt(pred.variance.detach())
 
    logger.info(f"Preparing calibration plot.\n") 

    calibration_plot(
        pred_mu=pred_mu.cpu(), 
        pred_sigma=pred_sigma.cpu(),
        obs=Y_test, 
        wandb=wandb,
        logger=logger
    )

    logger.info(f"Mean: {pred_mu}")
    logger.info(f"Sigma: {pred_sigma}")

    del pred_mu, pred_sigma, mll, likelihood
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Training complete.\n")   
    return model

############################
############################
############################

if __name__ == "__main__":
    with botorch.settings.debug(True):
        parser = argparse.ArgumentParser('Training equivariant networks', parents=[get_args_parser()])
        args = parser.parse_args()  
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        main(args)
