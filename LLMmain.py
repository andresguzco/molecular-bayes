from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import *
import torch.utils.data as data_utils
import tqdm
from sklearn.utils import shuffle as skshuffle
import os
from pathlib import Path
import argparse
import math
import matplotlib.pyplot as plt
from Interface.pars_args import get_args_parser
from datetime import datetime
from torch import nn
from transformers import AutoModel, AutoTokenizer
from lapeft_bayesopt.problems.data_processor import DataProcessor
from lapeft_bayesopt.utils.acqf import thompson_sampling
from lapeft_bayesopt.utils import helpers
from lapeft_bayesopt.problems.prompting import PromptBuilder

from Pipeline import FileLogger
import wandb

from torch import nn, optim
from transformers import AutoModel, AutoTokenizer

import warnings

warnings.filterwarnings("ignore")
from torch import distributions as dists
import botorch.models.model as botorch_model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from laplace import Laplace
from laplace.curvature import BackPackGGN
from laplace.marglik_training import marglik_training


import argparse



class LaplaceBoTorch(botorch_model.Model):
    """
    BoTorch surrogate model with a Laplace-approximated Bayesian
    neural network. The Laplace class is defined in the library
    laplace-torch; install via:
    `pip install https://github.com/aleximmer/laplace.git`.

    Args:
    -----
    get_net: function None -> nn.Module
        Function that doesn't take any args and return a PyTorch model.
        Prefer torch.nn.Sequential model due to BackPACK dependency.
        Example usage: `get_net=lambda: nn.Sequential(...)`.

    train_X : torch.Tensor
        Training inputs of size (n_data, ...).

    train_Y : torch.Tensor
        Training targets of size (n_data, n_tasks).

    bnn : Laplace, optional, default=None
        When creating a new model from scratch, leave this at None.
        Use this only to update this model with a new observation during BO.

    likelihood : {'regression', 'classification'}
        Indicates whether the problem is regression or classification.

    noise_var : float | None, default=None.
        Output noise variance. If float, must be >= 0. If None,
        it is learned by marginal likelihood automatically.

    last_layer : bool, default False
        Whether to do last-layer Laplace. If True, then the model used is the
        so-called "neural linear" model.

    hess_factorization : {'full', 'diag', 'kron'}, default='kron'
        Which Hessian factorization to use to do Laplace. 'kron' provides the best
        tradeoff between speed and approximation error.

    marglik_mode : {'posthoc', 'online'}, default='posthoc'
        Whether to do online marginal-likelihood training or do standard NN training
        and use marglik to optimize hyperparams post-hoc.

    posthoc_marglik_iters: int > 0, default=100
        Number of iterations of post-hoc marglik tuning.

    online_marglik_freq: int > 0 default=50
        How often (in terms of training epoch) to do online marglik tuning.

    batch_size : int, default=64
        Batch size to use for the NN training and doing Laplace.

    n_epochs : int, default=500
        Number of epochs for training the NN.

    lr : float, default=1e-1
        Learning rate to use for training the NN.

    wd : float, default=1e-3
        Weight decay for training the NN.

    device : {'cpu', 'cuda'}, default='cpu'
        Which device to run the experiment on.
    """

    def __init__(
        self,
        get_net: Callable[[], nn.Module],
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        bnn: Laplace = None,
        likelihood: str = "regression",
        noise_var: float | None = None,
        last_layer: bool = False,
        hess_factorization: str = "kron",
        marglik_mode: str = "posthoc",
        posthoc_marglik_iters: int = 100,
        online_marglik_freq: int = 50,
        batch_size: int = 20,
        n_epochs: int = 500,
        lr: float = 1e-3,
        wd: float = 5e-4,
        device: str = "cpu",
    ):
        super().__init__()

        self.train_X = train_X
        self.train_Y = train_Y
        assert likelihood in ["regression"]  # For now
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.last_layer = last_layer
        self.subset_of_weights = "last_layer" if last_layer else "all"
        self.hess_factorization = hess_factorization
        self.posthoc_marglik_iters = posthoc_marglik_iters
        self.online_marglik_freq = online_marglik_freq
        assert device in ["cpu", "cuda"]
        self.device = device
        assert marglik_mode in ["posthoc", "online"]
        self.marglik_mode = marglik_mode
        self.n_epochs = n_epochs
        self.lr = lr
        self.wd = wd
        self.get_net = get_net
        self.bnn = bnn

        if type(noise_var) != float and noise_var is not None:
            raise ValueError("Noise variance must be float >= 0. or None")
        if type(noise_var) == float and noise_var < 0:
            raise ValueError("Noise variance must be >= 0.")
        self.noise_var = noise_var

        # Initialize Laplace
        if self.bnn is None:
            self._train_model(self._get_train_loader())

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        mean_y, var_y = self.get_prediction(X, use_test_loader=False, joint=False)
        if len(var_y.shape) != 3:  # Single objective
            return dists.Normal(mean_y, var_y.squeeze(-1))
        else:  # Multi objective
            return dists.MultivariateNormal(mean_y, var_y)

    def condition_on_observations(
        self, X: torch.Tensor, Y: torch.Tensor, **kwargs: Any
    ) -> LaplaceBoTorch:
        # Append new observation to the current data
        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_Y = torch.cat([self.train_Y, Y], dim=0)

        # Update Laplace with the updated data
        train_loader = self._get_train_loader()
        self._train_model(train_loader)

        return LaplaceBoTorch(
            # Replace the dataset & retrained BNN
            get_net=self.get_net,
            train_X=self.train_X,  # Important!
            train_Y=self.train_Y,  # Important!
            bnn=self.bnn,  # Important!
            likelihood=self.likelihood,
            noise_var=self.noise_var,
            last_layer=self.last_layer,
            hess_factorization=self.hess_factorization,
            marglik_mode=self.marglik_mode,
            posthoc_marglik_iters=self.posthoc_marglik_iters,
            online_marglik_freq=self.online_marglik_freq,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            lr=self.lr,
            wd=self.wd,
            device=self.device,
        )

    def get_prediction(self, test_X: torch.Tensor, joint=True, use_test_loader=False):
        """
        Batched Laplace prediction.

        Args:
        -----
        test_X: torch.Tensor
            Array of size `(batch_shape, feature_dim)`.

        joint: bool, default=True
            Whether to do joint predictions (like in GP).

        use_test_loader: bool, default=False
            Set to True if your test_X is large.


        Returns:
        --------
        mean_y: torch.Tensor
            Tensor of size `(batch_shape, num_tasks)`.

        cov_y: torch.Tensor
            Tensor of size `(batch_shape*num_tasks, batch_shape*num_tasks)`
            if joint is True. Otherwise, `(batch_shape, num_tasks, num_tasks)`.
        """
        if self.bnn is None:
            raise Exception("Train your model first before making prediction!")

        if not use_test_loader:
            mean_y, cov_y = self.bnn(test_X.to(self.device), joint=joint)
        else:
            test_loader = data_utils.DataLoader(
                data_utils.TensorDataset(test_X, torch.zeros_like(test_X)),
                batch_size=256,
            )

            mean_y, cov_y = [], []

            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                _mean_y, _cov_y = self.bnn(X_batch, joint=joint)
                mean_y.append(_mean_y)
                cov_y.append(_cov_y)

            mean_y = torch.cat(mean_y, dim=0).squeeze()
            cov_y = torch.cat(cov_y, dim=0).squeeze()

        return mean_y, cov_y

    @property
    def num_outputs(self) -> int:
        """The number of outputs of the model."""
        return self.train_Y.shape[-1]

    def _train_model(self, train_loader):
        del self.bnn

        if self.marglik_mode == "posthoc":
            self._posthoc_laplace(train_loader)
        else:
            # Online
            la, model, _, _ = marglik_training(
                # Ensure that the base net is re-initialized
                self.get_net(),
                train_loader,
                likelihood=self.likelihood,
                hessian_structure=self.hess_factorization,
                n_epochs=self.n_epochs,
                backend=BackPackGGN,
                optimizer_kwargs={"lr": self.lr},
                scheduler_cls=optim.lr_scheduler.CosineAnnealingLR,
                scheduler_kwargs={"T_max": self.n_epochs * len(train_loader)},
                marglik_frequency=self.online_marglik_freq,
                # enable_backprop=True  # Important!
            )
            self.bnn = la

        # Override sigma_noise if self.noise_var is not None
        if self.noise_var is not None:
            self.bnn.sigma_noise = math.sqrt(self.noise_var)

    def _posthoc_laplace(self, train_loader):
        net = self.get_net()  # Ensure that the base net is re-initialized
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.n_epochs * len(train_loader)
        )
        loss_func = (
            nn.MSELoss() if self.likelihood == "regression" else nn.CrossEntropyLoss()
        )

        for _ in range(self.n_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = net(x)
                loss = loss_func(output, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

        net.eval()
        self.bnn = Laplace(
            net,
            self.likelihood,
            subset_of_weights=self.subset_of_weights,
            hessian_structure=self.hess_factorization,
            backend=BackPackGGN,
        )
        self.bnn.fit(train_loader)

        if self.likelihood == "classification":
            self.bnn.optimize_prior_precision(n_steps=self.posthoc_marglik_iters)
        else:
            # For regression, tune prior precision and observation noise
            log_prior, log_sigma = (
                torch.ones(1, requires_grad=True),
                torch.ones(1, requires_grad=True),
            )
            hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
            for _ in range(self.posthoc_marglik_iters):
                hyper_optimizer.zero_grad()
                neg_marglik = -self.bnn.log_marginal_likelihood(
                    log_prior.exp(), log_sigma.exp()
                )
                neg_marglik.backward()
                hyper_optimizer.step()

    def _get_train_loader(self):
        return data_utils.DataLoader(
            data_utils.TensorDataset(self.train_X, self.train_Y),
            batch_size=self.batch_size,
            shuffle=True,
        )


def get_molformer_tokenizer():
    return AutoTokenizer.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True, use_fast=False
    )


class MolFormerRegressor(nn.Module):
    def __init__(self, tokenizer, n_last_hidden_units=100):
        super().__init__()
        self.tokenizer = tokenizer
        self.feature_extractor = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            deterministic_eval=True,
            trust_remote_code=True,
        )
        self.feature_dim = self.feature_extractor.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, n_last_hidden_units),
            nn.ReLU(),
            nn.Linear(n_last_hidden_units, 1),
        )

    def forward(self, data):
        feat = self.forward_features(data)
        return self.head(feat)

    def forward_features(self, data):
        input_ids, attn_mask = data["input_ids"], data["attention_mask"]
        device = next(self.parameters()).device
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        feat = self.feature_extractor(input_ids, attn_mask).pooler_output
        return feat

    def freeze_params(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_params(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = True


class MyDataProcessor(DataProcessor):
    """
    Data columns (total 2 columns):
    #   Column                  Dtype
    --  ------                  -----
    0   SMILES                  object
    1   energy                  float64

    Objective: Maximize energy 
    """

    def __init__(self, prompt_builder, tokenizer):
        super().__init__(
            prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer
        )
        self.x_col = "SMILES"
        self.target_col = "energy"
        self.obj_str = "total tnegry in Hartree"
        self.maximization = True
    
    def _get_columns_to_remove(self) -> List[str]:
        return [
            "SMILES",
            "energy"
        ]


class MyPromptBuilder(PromptBuilder):
    def __init__(self, kind: str):
        self.kind = kind

    def get_prompt(self, x: str, obj_str: str) -> str:
        if self.kind == "completion":
            return f"The estimated {obj_str} of the molecule {x} is: "
        elif self.kind == "just-smiles":
            return x
        else:
            return NotImplementedError


def main(args):

    _log = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    _log.info(args)
    wandb.login()

    ############################
    # Weights and Biases
    ############################
    run = wandb.init(
    project="3D_Bayes",
    config=args,
    name=f"BO - LLM"
    )
    ############################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ############################
    # Dataset 
    ############################
    pd_dataset = pd.read_csv(f"{args.data_path}/smiles_energy.csv")

    if pd_dataset.shape[0] > 10010:
        pd_dataset = pd_dataset.sample(n=10010, random_state=args.seed)
        pd_dataset.reset_index(drop=True, inplace=True)
        pd_dataset["energy"] = pd_dataset["energy"].astype(float)
        pd_dataset["SMILES"] = pd_dataset["SMILES"].astype(str) 

    _log.info(f"Loaded dataset with {pd_dataset.shape[0]} molecules.")

    dataset = {
        "pd_dataset": pd_dataset,
        "smiles_col": "SMILES",
        "obj_col": "energy",
        "maximization": True,
        "cache_path": f"{args.data_path}/",
        "opt_val": pd_dataset["energy"].min(),
    }

    results = run_bayesopt(dataset, n_init_data=10, T=1000, device=device, randseed=args.seed, wandb=run, logger=_log)

    # Plot
    t = np.arange(len(results))
    plt.axhline(dataset["opt_val"], color="black", linestyle="dashed")
    plt.plot(t, results)
    plt.xlabel(r"$t$")
    plt.ylabel(r"Objective ($\downarrow$)")
    plt.show()

    ############################
    # End of loop
    ############################
    _log.info(f"BO Completed. Best: {max(results)}.")
    wandb.finish()
    return None


def load_features(dataset, device):
    """
    Load cached features (e.g. the fingerprints of each molecules in the dataset) if exist.
    Otherwise, compute the said features and cache them.
    Also, transform the problem into maximization.
    """
    pd_dataset = dataset["pd_dataset"]
    CACHE_PATH = dataset["cache_path"]
    SMILES_COL = dataset["smiles_col"]
    OBJ_COL = dataset["obj_col"]
    MAXIMIZATION = dataset["maximization"]

    # If cache exists then just load it, otherwise compute the features
    if os.path.exists(f"{CACHE_PATH}/cached_feats.bin"):
        features = torch.load(f"{CACHE_PATH}/cached_feats.bin")
        targets = torch.load(f"{CACHE_PATH}/cached_targets.bin")
    else:
        tokenizer = get_molformer_tokenizer()
        llm_feat_extractor = MolFormerRegressor(tokenizer=tokenizer)

        # Need CUDA otherwise will be so slow!
        llm_feat_extractor = llm_feat_extractor.to(device)
        llm_feat_extractor.eval()
        llm_feat_extractor.freeze_params()

        # Here, we use the raw SMILES string as the input to the LLM
        prompt_builder = MyPromptBuilder(kind="completion")
        data_processor = MyDataProcessor(prompt_builder, tokenizer)
        dataloader = data_processor.get_dataloader(pd_dataset, shuffle=False, append_eos=False)

        # Forward pass through the LLM, take the aggregate (over sequence dimension)
        # of the last transformer embeddings/features
        features, targets = [], []
        for data in tqdm.tqdm(dataloader):
            with torch.no_grad():
                feat = llm_feat_extractor.forward_features(data)

            features += list(feat.cpu())

            # Here we transform the target so that the optimization problem
            # always corresponds to maximization
            targets += list(data["labels"])

        # Cache to files
        torch.save(features, f"{CACHE_PATH}/cached_feats.bin")
        torch.save(targets, f"{CACHE_PATH}/cached_targets.bin")

    return features, targets


def run_bayesopt(dataset, n_init_data=10, T=1000, device="cpu", randseed=1, wandb=None, logger=None):
    np.random.seed(randseed)
    torch.manual_seed(randseed)

    features, targets = load_features(dataset, device)

    # Shuffle since some .csv datasets are ordered by the objective values
    features, targets = skshuffle(features, targets, random_state=randseed)
    feature_dim = features[0].shape[-1]
    ground_truth_max = torch.tensor(targets).flatten().max()

    # Obtain a small initial dataset for BO.
    # There are different strategy for this. Here, we use a simple random sampling.
    train_x, train_y = [], []
    while len(train_x) < n_init_data:
        idx = np.random.randint(len(features))
        # Make sure that the optimum is not included
        if targets[idx].item() >= ground_truth_max:
            continue
        train_x.append(features.pop(idx))
        train_y.append(targets.pop(idx))
    train_x, train_y = torch.stack(train_x), torch.stack(train_y)

    # Surrogate
    def get_net():
        activation = torch.nn.Tanh
        return torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 50),
            activation(),
            torch.nn.Linear(50, 50),
            activation(),
            # For multiobjective problems, change 1 -> num_of_objectives
            torch.nn.Linear(50, 1),
        )
    device = "cpu"
    model = LaplaceBoTorch(
        get_net, train_x, train_y, noise_var=0.001, hess_factorization="kron"
    )
    model = model.to(device) if model is not None else model

    # Prepare for the BO loop
    MAXIMIZATION = dataset["maximization"]
    best_y = train_y.max().item()  # Current best f(x) from the initial dataset
    pbar = tqdm.trange(T)
    pbar.set_description(
        f"[Best f(x) = {helpers.y_transform(best_y, MAXIMIZATION):.3f}]"
    )

    # To store the logged best f(x) over time
    trace_best_y = [helpers.y_transform(best_y, MAXIMIZATION)]

    features_tensor, targets_tensor = torch.stack(features), torch.stack(targets).flatten()

    # The BayesOpt loop --- or just use BoTorch since LaplaceBoTorch is compatible
    for _ in pbar:
        features_tensor, targets_tensor = torch.stack(features), torch.stack(targets).flatten()

        # Don't shuffle so that the ordering is the same as the pandas dataset
        dataloader = data_utils.DataLoader(
            data_utils.TensorDataset(features_tensor, targets_tensor),
            batch_size=256,
            shuffle=False,
        )

        # Make prediction over all unknown molecules, then use the predictive mean
        # and variance to compute the acquisition function values
        acq_vals = []
        # The same random seed for Thompson sampling across the batches
        rstate = int(datetime.now().timestamp())
        for x, y in dataloader:
            # Obtain the posterior predictive distribution p(g_t(x) | D_t)
            posterior = model.posterior(x)

            # For multiobjective problems take the covariance matrix
            # i.e., f_cov = posterior.covariance_matrix
            f_mean, f_var = posterior.mean, posterior.variance

            # Feel free to use different acquisition function
            acq_vals.append(thompson_sampling(f_mean, f_var, random_state=rstate))

        # Pick a molecule (a row in the current dataset) that maximizes the acquisition.
        # Also remove it from the pool (hence we use .pop)
        acq_vals = torch.cat(acq_vals, dim=0).cpu().squeeze()
        idx_best = torch.argmax(acq_vals).item()
        new_x, new_y = features.pop(idx_best), targets.pop(idx_best)

        # Update the current best y
        if new_y.item() > best_y:
            best_y = new_y.item()

        # Remember that the cached features are always in maximization format.
        # So here, we transform it back if necessary.
        pbar.set_description(
            f"[Best f(x) = {helpers.y_transform(best_y, MAXIMIZATION):.3f}, "
            + f"curr f(x) = {helpers.y_transform(new_y.item(), MAXIMIZATION):.3f}]"
        )

        # Concatenate the newly acquired (x, y) and then update the surrogate
        model = model.condition_on_observations(new_x.unsqueeze(0), new_y.unsqueeze(0))

        # Log the current best f(x) value
        trace_best_y.append(best_y)

        wandb.log(data={
            "Selection": idx_best,
            "Best": best_y,
            # "Virtual Max": virtual_library.Y.max().item(),
            # "Train Max": train_dataset.Y.max().item()
        })

    return trace_best_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training LLM molecular feature extractor.')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default="results/models/LLM")

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    # Compute GAP from Roman Garnett