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
from Pipeline import FileLogger, BayesianOptimiser
from rdkit.Chem import AllChem
from datasets.QM9 import QM9
from Pipeline.utils import calibration_plot, plotter
from Interface.pars_args import get_args_parser
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
    val_dataset     = QM9(args.data_path, 'valid', feature_type=args.feature_type)
    ############################

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
    ############################

    ############################
    # Data Prep
    ############################
    fpgen = AllChem.GetMorganGenerator(radius=int(args.radius), fpSize=2048)

    X_val = torch.tensor([fpgen.GetFingerprint(data.mol) for data in val_dataset])
    Y_val = val_dataset.y[:, args.target]
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
    create_model = registry.get_model_class("GP_Exact")
    ############################

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

        ############################
        # Step 1: Train surrogate model
        ############################
        GP = GP_trainer(model_gen=create_model, X_train=X_train.float(), Y_train=Y_train.float(), 
                        X_test=X_val.float(), Y_test=Y_val.float(), device=device, logger=_log, 
                        wandb=run)

        if iter == 0: 
            _log.info(GP)

        ###############################
        # Step 2: Acquisition function 
        ###############################
        bayesOpt = BayesianOptimiser(model_type="GP")

        ############################
        # Step 3: Query 
        ############################
        idx_query = bayesOpt.query(
            model=GP,
            virtual_loader=X_virtual.float(),
            incumbent_y=torch.max(Y_train.float()),
            device=device,
            logger=_log
            )

        _log.info(f"Iter: {iter}. Next query: {idx_query}. Selection: {Y_virtual[idx_query]:.2f}.\n")

        ############################
        # Step 4: Update 
        ############################
        vSelection[iter] = virtual_library.data.y[idx_query, args.target]
        if iter < args.iterations - 1:
            vIncumbent[iter + 1] = vSelection[iter] if vSelection[iter] > vIncumbent[iter] else vIncumbent[iter]


        X_train = torch.cat([X_train, X_virtual[idx_query].unsqueeze(0)])
        Y_train = torch.cat([Y_train, Y_virtual[idx_query].unsqueeze(0)])
        X_virtual = torch.cat([X_virtual[:idx_query], X_virtual[idx_query + 1:]])
        Y_virtual = torch.cat([Y_virtual[:idx_query], Y_virtual[idx_query + 1:]])

        del GP
        torch.cuda.empty_cache()
        gc.collect()
    
    plotter(vIncumbent, vSelection, train_max, virtual_max, general_max, wandb=run)

    ############################
    # End of loop
    ############################


def GP_trainer(model_gen, X_train, Y_train, X_test, Y_test, device, logger, wandb):

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = model_gen(X_train, Y_train, likelihood)
    model = model.to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll)

    model.eval()
    likelihood.eval()
                                                             
    logger.info(f"\nEvaluating model.\n")                                                                                                                                
                                                                                                    
    with torch.no_grad():                                                                                                                                    

        data = X_test.to(device)          
        pred = model(data)  
        pred_mu = pred.mean                                               
    
    logger.info(f"Preparing calibration plot.\n")  
    calibration_plot(
        pred_mu=pred_mu.cpu(), 
        obs=Y_test, 
        wandb=wandb,
        logger=logger
        )

    logger.info(f"Training complete.\n")   
    return model

############################
############################
############################

if __name__ == "__main__":
    with botorch.settings.debug(state=True):
        parser = argparse.ArgumentParser('Training equivariant networks',
                                        parents=[get_args_parser()])
        args = parser.parse_args()  
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        main(args)