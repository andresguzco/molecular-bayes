from typing import Iterable, Optional

from timm.utils import ModelEmaV2, dispatch_clip_grad

import time
import torch
import torch_geometric
from torch_cluster import radius_graph


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


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    norm_factor: list, 
                    target: int,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, 
                    model_ema: Optional[ModelEma] = None,  
                    amp_autocast=None,
                    loss_scaler=None,
                    clip_grad=None,
                    print_freq: int = 100, 
                    logger=None,
                    wandb=None,
                    variational=False
                    ):
    
    model.train()
    criterion.train()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    
    start_time = time.perf_counter()
    
    task_mean = norm_factor[0] #model.task_mean
    task_std  = norm_factor[1] #model.task_std

    #atomref = dataset.atomref()
    #if atomref is None:
    #    atomref = torch.zeros(100, 1)
    #atomref = atomref.to(device)
    
    for step, data in enumerate(data_loader):
        data = data.to(device)
        #data.edge_d_index = radius_graph(data.pos, r=10.0, batch=data.batch, loop=True)
        #data.edge_d_attr = data.edge_attr
        with amp_autocast():
            pred = model(data)
            pred = pred.squeeze()
            #loss = (pred - data.y[:, target])
            #loss = loss.pow(2).mean()
            #atomref_value = atomref(data.z)

            loss = criterion(pred, (data.y[:, target] - task_mean) / task_std)

            if variational:
                KL = BKLLoss(model)
                loss += KL

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), 
                    value=clip_grad, mode='norm')
            optimizer.step()
        
        #err = (pred.detach() * task_std + task_mean) - data.y[:, target]
        #err_list += [err.cpu()]
        loss_metric.update(loss.item(), n=pred.shape[0])
        err = pred.detach() * task_std + task_mean - data.y[:, target]
        mae_metric.update(torch.mean(torch.abs(err)).item(), n=pred.shape[0])
        
        if model_ema is not None:
            model_ema.update(model)
        
        torch.cuda.synchronize()
        
        # logging
        if step % print_freq == 0 or step == len(data_loader) - 1: #time.perf_counter() - wall_print > 15:
            w = time.perf_counter() - start_time
            e = (step + 1) / len(data_loader)
            info_str = 'Epoch: [{epoch}][{step}/{length}] \t loss: {loss:.5f}, MAE: {mae:.5f}, time/step={time_per_step:.0f}ms, '.format( 
                epoch=epoch, step=step, length=len(data_loader), 
                mae=mae_metric.avg, 
                loss=loss_metric.avg,
                time_per_step=(1e3 * w / e / len(data_loader))
                )
            info_str += 'lr={:.2e}'.format(optimizer.param_groups[0]["lr"])
            logger.info(info_str)
            wandb.log({
            "Loss": loss.item() - KL if variational else loss.item(), 
            "MAE": torch.mean(torch.abs(err)).item(),
            "KL": KL if variational else 0
            })
        
    return mae_metric.avg


def evaluate(model, norm_factor, target, data_loader, device, amp_autocast=None, 
    print_freq=100, logger=None):
    
    model.eval()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    criterion = torch.nn.L1Loss()
    criterion.eval()
    
    task_mean = norm_factor[0] #model.task_mean
    task_std  = norm_factor[1] #model.task_std
    
    with torch.no_grad():
            
        for data in data_loader:
            data = data.to(device)
            #data.edge_d_index = radius_graph(data.pos, r=10.0, batch=data.batch, loop=True)
            #data.edge_d_attr = data.edge_attr
            with amp_autocast():
                pred = model(data)
                pred = pred.squeeze()
            
            loss = criterion(pred, (data.y[:, target] - task_mean) / task_std)
            err = pred.detach() * task_std + task_mean - data.y[:, target]
            
            loss_metric.update(loss.item(), n=pred.shape[0])
            mae_metric.update(torch.mean(torch.abs(err)).item(), n=pred.shape[0])
        
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
        edge_src, edge_dst = radius_graph(pos, r=max_radius, batch=batch,
            max_num_neighbors=1000)
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


def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1):
    kl = log_sigma_1 - log_sigma_0 + \
    (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2) / (2 * torch.exp(log_sigma_1)**2) - 0.5
    return kl.sum(dim=1).mean()


def BKLLoss(model) :
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")

    m = model.energy_block.so3_linear_2

    bias_mu = m.bias_mu.to(device)
    prior_mu = m.prior_mu.to(device)
    weight_mu = m.weight_mu.to(device)
    bias_log_sigma = m.bias_log_sigma.to(device)
    prior_log_sigma = m.prior_log_sigma.to(device)
    weight_log_sigma = m.weight_log_sigma.to(device)

    return (_kl_loss(weight_mu, weight_log_sigma, prior_mu, prior_log_sigma) 
            + _kl_loss(bias_mu, bias_log_sigma, prior_mu, prior_log_sigma))