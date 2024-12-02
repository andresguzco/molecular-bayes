from typing import Iterable, Optional

from timm.utils import ModelEmaV2, dispatch_clip_grad
import gc
import os
import time
import torch
import random
import torch_geometric
from torch_cluster import radius_graph

from .utils import calibration_plot
from laplace import Laplace
from laplace.curvature import (
    AsdlEF,
    AsdlGGN,
    AsdlHessian,
    BackPackGGN,
    CurvlinopsEF,
    CurvlinopsGGN,
    CurvlinopsHessian,
)
from torch_geometric.loader import DataLoader
from nets import ExactGPModel
import gpytorch


ModelEma = ModelEmaV2


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def denoising_loss(denoised_output, original_features):
    return torch.nn.functional.mse_loss(denoised_output, original_features)


def add_noise_to_features(features, corrupt_ratio=0.1, noise_level=0.1):
    num_features_to_corrupt = int(corrupt_ratio * features.size(0))
    noise_indices = torch.randperm(features.size(0))[:num_features_to_corrupt]
    noise = torch.randn_like(features[noise_indices]) * noise_level
    noisy_features = features.clone()
    noisy_features[noise_indices] += noise
    return noisy_features


def train_one_epoch(encoder, decoder, criterion, norm_factor, target, data_loader, optimizer,
                    device, epoch, amp_autocast=None, loss_scaler=None, clip_grad=None,
                    print_freq=100, logger=None):

    encoder.train()
    decoder.train()
    criterion.train()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    KL_metric = AverageMeter()
    
    start_time = time.perf_counter()
    
    task_mean = norm_factor[0] 
    task_std  = norm_factor[1] 
    
    for step, data in enumerate(data_loader):
        
        # Apply denoising based on the denoising probability
        if encoder.name == "equiformer_v2":
            if random.random() < 0.5:

                true_pos = data.pos
                true_pos = true_pos.to(device)

                data.pos = add_noise_to_features(data.pos)
                data = data.to(device)

                with amp_autocast():
                    embedding, denoised_output = encoder(data)
                    preds = [head(embedding) for head in decoder]

                    loss = 0
                    denoise_loss = denoising_loss(denoised_output, true_pos)
                    loss += 0.1 * denoise_loss
                    for i, pred in enumerate(preds):
                        loss += criterion(pred.squeeze(), (data.y[:, target[i]] - task_mean[i]) / task_std[i])
            else:

                data = data.to(device)

                with amp_autocast():
                    embedding, _ = encoder(data)
                    preds = [head(embedding) for head in decoder]

                    loss = 0
                    denoise_loss = 0
                    loss += 0.1 * denoise_loss
                    for i, pred in enumerate(preds):
                        loss += criterion(pred.squeeze(), (data.y[:, target[i]] - task_mean[i]) / task_std[i])
        else:
            data = data.to(device)

            with amp_autocast():
                embedding = encoder(data)
                preds = [head(embedding) for head in decoder]

                loss = 0
                for i, pred in enumerate(preds):
                    loss += criterion(pred.squeeze(), (data.y[:, target[i]] - task_mean[i]) / task_std[i])

        N = len(preds[0]) if preds[0].dim() != 0 else 1

        params = [param for param in encoder.parameters()] + [param for head in decoder for param in head.parameters()]
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=params)
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(params, value=clip_grad, mode='norm')
        optimizer.step()
        
        loss_metric.update(loss.item(), n=N)

        errors = [(pred.detach() * task_std[i] + task_mean[i] - data.y[:, target[i]]) / data.y[:, target[i]] for i, pred in enumerate(preds)]
        err = torch.stack(errors, dim=0)
        mae_metric.update(torch.mean(torch.abs(err)).item(), n=N)
        
        # logging
        if step % print_freq == 0 or step == len(data_loader) - 1: 
            w = time.perf_counter() - start_time
            e = (step + 1) / len(data_loader)
            info_str = f'Epoch: [{epoch}][{step}/{len(data_loader)}] \t loss: {loss_metric.avg:.5f}, '
            info_str += f'MAE: {mae_metric.avg:.5f}, time/step={(1e3 * w / e / len(data_loader)):.0f}ms, '
            info_str += f'lr={optimizer.param_groups[0]["lr"]:.2e}'
            logger.info(info_str)
                 
    return mae_metric.avg, loss_metric.avg




def evaluate(encoder, decoder, norm_factor, target, data_loader, device, amp_autocast=None):
    encoder.eval()
    decoder.eval()

    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    criterion = torch.nn.L1Loss()
    criterion.eval()

    task_mean = norm_factor[0] 
    task_std  = norm_factor[1]

    with torch.no_grad():

        for data in data_loader:
            data = data.to(device)

            with amp_autocast():
                if encoder.name == "equiformer_v2":
                    embedding, _ = encoder(data)
                    preds = [head(embedding) for head in decoder]
                else:
                    embedding = encoder(data)
                    preds = [head(embedding) for head in decoder]

            N = len(preds[0]) if preds[0].dim() != 0 else 1

            loss = 0
            for i, pred in enumerate(preds):
                pred = pred.squeeze().detach()
                loss += criterion(pred, (data.y[:, target[i]] - task_mean[i]) / task_std[i])

            errors = [(pred.detach() * task_std[i] + task_mean[i] - data.y[:, target[i]]
            ) / data.y[:, target[i]] for i, pred in enumerate(preds)]
            err = torch.stack(errors, dim=0)

            loss_metric.update(loss.item(), n=N)
            mae_metric.update(torch.mean(torch.abs(err)).item(), n=N)

    return mae_metric.avg, loss_metric.avg




def compute_stats(data_loader, max_radius, logger, print_freq=1000):
    '''
        Compute mean of numbers of nodes and edges
    '''
    log_str = '\nCalculating statistics with '
    log_str = log_str + 'max_radius={}\n'.format(max_radius)
    logger.info(log_str)

    avg_node = AverageMeter()
    avg_edge = AverageMeter()
    avg_degree = AverageMeter()

    for step, data in enumerate(data_loader):

        pos = data.pos
        batch = data.batch
        edge_src, _ = radius_graph(pos, r=max_radius, batch=batch, max_num_neighbors=1000)
        batch_size = float(batch.max() + 1)
        num_nodes = pos.shape[0]
        num_edges = edge_src.shape[0]
        num_degree = torch_geometric.utils.degree(edge_src, num_nodes)
        num_degree = torch.sum(num_degree)

        avg_node.update(num_nodes / batch_size, batch_size)
        avg_edge.update(num_edges / batch_size, batch_size)
        avg_degree.update(num_degree / (num_nodes), num_nodes)

        if step % print_freq == 0 or step == (len(data_loader) - 1):
            log_str = '[{}/{}]\tavg node: {}, '.format(step, len(data_loader), avg_node.avg)
            log_str += 'avg edge: {}, '.format(avg_edge.avg)
            log_str += 'avg degree: {}, '.format(avg_degree.avg)
            logger.info(log_str)


def train_laplace(model, loader, criterion, logger):
    # Initialise the model for the laplace approximation
    la = Laplace(
        model, 
        'regression',
        subset_of_weights='all', 
        hessian_structure='kron', 
        backend=CurvlinopsGGN
    )

    # Fit the Laplace Approximation
    bayes_start_time = time.perf_counter()
    la.fit(loader)
    logger.info(f"Laplace fitted. Time taken: {time.perf_counter() - bayes_start_time:.2f}")

    # Optimise the prior precision
    la.optimize_prior_precision(method='marglik', pred_type='nn', loss=criterion)
    logger.info(f"Prior precision optimised. Time: {time.perf_counter() - bayes_start_time:.2f}\n")
    return la


def get_performance_plot(model, norm_factor, target, test_loader,
    device, logger, wandb, model_type): 

    logger.info(f"Evaluating model.")

    if model_type != "Laplace":
        model.eval()

    y = torch.tensor([], device=device, requires_grad=False)
    pred = torch.tensor([], device=device, requires_grad=False)
    sigma = torch.tensor([], device=device, requires_grad=False)

    with torch.no_grad():
        for _, data in enumerate(test_loader):
            data = data.to(device)
            pred_i, sigma_i = model(data)

            pred_i = pred_i.detach() * norm_factor[1] + norm_factor[0]
            sigma_i = torch.sqrt(sigma_i.detach()) * norm_factor[1]

            y = torch.cat([y.flatten(), data.y[:, target].flatten()])
            pred = torch.cat([pred.flatten(), pred_i.flatten()])  
            sigma = torch.cat([sigma.flatten(), sigma_i.flatten()])

    logger.info(f"Preparing calibration plot.\n")  
    calibration_plot(
        pred_mu=pred.cpu(), 
        pred_sigma=sigma.cpu(),
        obs=y.cpu(), 
        wandb=wandb,
        logger=logger
        )

    logger.info(f"Training complete.\n")


def train_gp(data, logger, lr, device, norm_factor=[0, 1], epochs=300):
    task_mean = norm_factor[0]
    task_std = norm_factor[1]

    logger.info(f"Training GP model.")

    train_x, train_y = data.X.to(device), data.Y.flatten().to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = ExactGPModel(train_x, train_y, likelihood)
    model = model.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    counter = 0
    best_loss = float('inf')
    for i in range(epochs):
        optimizer.zero_grad()

        output = model(train_x) * task_std + task_mean
        loss = -mll(output, train_y)

        loss.backward()
        optimizer.step()
        
        if loss < best_loss:
            best_loss = loss
            counter = 0 
        else:
            counter += 1
        
        if counter > 10:
            break

    logger.info(f"GP training complete.\n")

    return model
