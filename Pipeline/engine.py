from typing import Iterable, Optional

from timm.utils import ModelEmaV2, dispatch_clip_grad
import gc
import os
import time
import torch
import torch_geometric
from torch_cluster import radius_graph

from .utils import calibration_plot
from laplace import Laplace
from laplace.curvature import CurvlinopsEF
from torch_geometric.loader import DataLoader


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
                    model_type=False
                    ):

    model.train()
    criterion.train()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    
    start_time = time.perf_counter()
    
    task_mean = norm_factor[0] 
    task_std  = norm_factor[1] 
    
    for step, data in enumerate(data_loader):
        data = data.to(device)
        with amp_autocast():
            if model_type == "Variational":
                pred, _, _ = model(data)
            else:
                pred = model(data)

            pred = pred.squeeze()

            loss = criterion(pred, (data.y[:, target] - task_mean) / task_std)

            if model_type == "Variational":
                KL = BKLLoss(model)
                loss += KL.item()

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), 
                    value=clip_grad, mode='norm')
            optimizer.step()
        
        loss_metric.update(loss.item(), n=pred.shape[0])
        err = pred.detach() * task_std + task_mean - data.y[:, target]
        mae_metric.update(torch.mean(torch.abs(err)).item(), n=pred.shape[0])
        
        if model_ema is not None:
            model_ema.update(model)
        
        # logging
        if step % print_freq == 0 or step == len(data_loader) - 1: 
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
            "Loss": loss.item() - KL if model_type == 'Variational' else loss.item(), 
            "MAE": torch.mean(torch.abs(err)).item(),
            "KL": KL if model_type == 'Variational' else None
            })

        del pred, loss, err, data
        torch.cuda.empty_cache()
        gc.collect() 
        
    return mae_metric.avg




def evaluate(model, norm_factor, target, data_loader, device, amp_autocast=None, model_type='Base'):
    
    model.eval()
    
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
                if model_type == "Variational":
                    pred, _, _ = model(data)
                else:
                    pred = model(data)

                pred = pred.squeeze().detach()
            
            loss = criterion(pred, (data.y[:, target] - task_mean) / task_std)
            err = pred.detach() * task_std + task_mean - data.y[:, target]
            
            loss_metric.update(loss.item(), n=pred.shape[0])
            mae_metric.update(torch.mean(torch.abs(err)).item(), n=pred.shape[0])
        
            del pred, loss, err, data
            torch.cuda.empty_cache()
            gc.collect() 

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
        edge_src, _ = radius_graph(pos, r=max_radius, batch=batch,
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

    bias_mu = m.bias_mu.detach().to(device)
    prior_mu = m.prior_mu.detach().to(device)
    weight_mu = m.weight_mu.detach().to(device)
    bias_log_sigma = m.bias_log_sigma.detach().to(device)
    prior_log_sigma = m.prior_log_sigma.detach().to(device)
    weight_log_sigma = m.weight_log_sigma.detach().to(device)

    return (_kl_loss(weight_mu, weight_log_sigma, prior_mu, prior_log_sigma) 
            + _kl_loss(bias_mu, bias_log_sigma, prior_mu, prior_log_sigma))




def train_laplace(model, train_loader, val_loader, criterion, target, logger, model_type, epochs=10):

    if model_type == "Laplace":                            
        last_layer_params = sum(p.numel() for p in model.energy_block.so3_linear_2.parameters())      
        logger.info(f'Laplace Layer Parameters: {last_layer_params}')                                    
                                                                                                    
        for parameter in model.sphere_embedding.parameters():                                         
            parameter.requires_grad = False                                                           
                                                                                                    
        for block in model.blocks:                                                                    
            for parameter in block.parameters():                                                      
                parameter.requires_grad = False                                                       
                                                                                                    
        for parameter in model.norm.parameters():                                                     
            parameter.requires_grad = False                                                           
                                                                                                    
        for parameter in model.edge_degree_embedding.parameters():                                    
            parameter.requires_grad = False                                                           
                                                                                                    
        for parameter in model.energy_block.so3_linear_1.parameters():                                
            parameter.requires_grad = False                                                           
                                                                                                    
        for parameter in model.energy_block.gating_linear.parameters():                               
            parameter.requires_grad = False          

    elif model_type == "BGNN":
        for parameter in model.encoder_layer.parameters():
            parameter.requires_grad = False     
        for parameter in model.second_last.parameters():
            parameter.requires_grad = False                                                              
                                                                                                
    # Initialise the model for the laplace approximation                                          
    la = Laplace(                                                                                 
        model,                                                                                    
        'regression',                                                                             
        hessian_structure='full',                                                                 
        subset_of_weights='all',                                                                  
        backend=CurvlinopsEF,                                                                      
        )                                                                                         
                                                                                                
    # Fit the Laplace Approximation                 
    bayes_start_time = time.perf_counter()                                  
    la.fit(train_loader, is_equiformer=True, target=target)                                                                                         
    logger.info(f"Laplace fitted. Time taken: {time.perf_counter() - bayes_start_time:.2f}")        
                                                                                                
    # Optimise the prior precision                                                                                                                      
    la.optimize_prior_precision_base(                                                             
        method='marglik',                                                                         
        pred_type='nn',                                                                           
        val_loader=val_loader,                                                                    
        loss=criterion                                                                       
        )                                                                                         
    logger.info(f"Prior precision optimised. Time: {time.perf_counter() - bayes_start_time:.2f}\n")    

    log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
    for _ in range(epochs):
        hyper_optimizer.zero_grad()
        neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()                        
                                                                                                                                                                
    torch.save({'model_state_dict': la.state_dict()},                                                                            
            os.path.join(os.getcwd(), f'checkpoints/Checkpoint_Laplace.pth.tar'))
    
    return la




def model_trainer(scheduler, model, criterion, 
                  norm_factor, target, train_loader,
                  val_loader, test_loader,
                  optimizer, device, epochs, model_ema, 
                  amp_autocast, loss_scaler, print_freq, 
                  logger, wandb, model_type):
    best_epoch, best_train_err, best_val_err, best_test_err = 0, float('inf'), float('inf'), float('inf')
    best_ema_epoch, best_ema_val_err, best_ema_test_err = 0, float('inf'), float('inf')               
    
    logger.info(f"Initiating training.\n")                       

    for epoch in range(epochs):                                                                  
        epoch_process(
            scheduler=scheduler, model=model, criterion=criterion, norm_factor=norm_factor, 
            target=target, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, 
            optimizer=optimizer, device=device, epoch=epoch, model_ema=model_ema, amp_autocast=amp_autocast,
            loss_scaler=loss_scaler, print_freq=print_freq, logger=logger, wandb=wandb, model_type=model_type,
            best_epoch=best_epoch, best_train_err=best_train_err, best_val_err=best_val_err, 
            best_test_err=best_test_err, best_ema_epoch=best_ema_epoch, best_ema_val_err=best_ema_val_err, 
            best_ema_test_err=best_ema_test_err) 

    if model_type == "Laplace" or model_type == "BGNN":  
        model.eval()                                                                
        trained_model = train_laplace(
            model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, 
            target=target, logger=logger, epochs=epochs, model_type=model_type
        )                          

    # Evaluate the model                                                                              
    logger.info(f"\nEvaluating model.")                                                                   

    if model_type != "Laplace":                                                                  
        model.eval()      

    y = torch.tensor([], device=device, requires_grad=False)                                          
    pred = torch.tensor([], device=device, requires_grad=False)                                                                 
                                                                                                    
    with torch.no_grad():                                                                             
        for _, data in enumerate(val_loader):                                                         

            data = data.to(device)       

            if model_type == "Laplace" or model_type == "BGNN":    
                pred_i, _ = trained_model(data)        
            elif model_type == "Variational":
                _, pred_i, _ = model(data)
            else:
                pred_i = model(data)

            pred_i = pred_i.detach() * norm_factor[1] + norm_factor[0]                      

            y = torch.cat([y.flatten(), data.y[:, target].flatten()])                            
            pred = torch.cat([pred.flatten(), pred_i.flatten()])  
            
            del data
            torch.cuda.empty_cache()
            gc.collect()                                                 
    
    logger.info(f"Preparing calibration plot.\n")  
    calibration_plot(
        pred_mu=pred.cpu(), 
        obs=y.cpu(), 
        wandb=wandb,
        logger=logger
        )

    logger.info(f"Training complete.\n")                 

    del y, pred
    torch.cuda.empty_cache()
    gc.collect()               
    
    return trained_model if model_type == "Laplace" or model_type == "BGNN" else None
 



def epoch_process(scheduler, model, criterion, 
                  norm_factor, target, train_loader,
                  val_loader, test_loader,
                  optimizer, device, epoch, model_ema, 
                  amp_autocast, loss_scaler, print_freq, 
                  logger, wandb, model_type,
                  best_epoch, best_train_err, 
                  best_val_err, best_test_err, 
                  best_ema_epoch, best_ema_val_err, 
                  best_ema_test_err):
                                                                                                
        epoch_start_time = time.perf_counter()                                                        
                                                                                                    
        scheduler.step(epoch)                                                                      
                                                                                                    
        train_err = train_one_epoch(model=model, criterion=criterion, norm_factor=norm_factor,        
            target=target, data_loader=train_loader, optimizer=optimizer,                        
            device=device, epoch=epoch, model_ema=model_ema,                                          
            amp_autocast=amp_autocast, loss_scaler=loss_scaler,                                       
            print_freq=print_freq, logger=logger, wandb=wandb, model_type=model_type)           
                                                                                                    
        val_err, _ = evaluate(model, norm_factor, target, val_loader, device,             
            amp_autocast=amp_autocast, model_type=model_type)                       
                                                                                                    
        test_err, _ = evaluate(model, norm_factor, target, test_loader, device,          
            amp_autocast=amp_autocast, model_type=model_type)                       
        wandb.log({                                                                                   
            "Validation Error": val_err,                                                              
            "Test Error": test_err,                                                                   
        })                                                                                            
                                                                                                    
        # record the best results                                                                     
        if val_err < best_val_err:                                                                    
            best_val_err = val_err                                                                    
            best_test_err = test_err                                                                  
            best_train_err = train_err                                                                
            best_epoch = epoch                                                                        
                                                                                                    
        info_str = f'Epoch: [{epoch}] Target: [{target}] train MAE: {train_err:.5f}, '           
        info_str += f'val MAE: {val_err:.5f}, '                                                       
        info_str += f'test MAE: {test_err:.5f}, '                                                     
        info_str += f'Time: {time.perf_counter() - epoch_start_time:.2f}s'                            
        logger.info(info_str)                                                                           
                                                                                                    
        info_str = f'Best -- epoch={best_epoch}, train MAE: {best_train_err:.5f}, '                   
        info_str += f'val MAE: {best_val_err:.5f}, test MAE: {best_test_err:.5f}.\n'                    
        logger.info(info_str)                                                                           
                                                                                                    
        # evaluation with EMA                                                                         
        if model_ema is not None:                                                                     
            ema_val_err, _ = evaluate(model_ema.module, norm_factor, target, val_loader, device, 
                amp_autocast=amp_autocast, model_type=model_type)                   
                                                                                                    
            ema_test_err, _ = evaluate(model_ema.module, norm_factor, target, test_loader, device,
                amp_autocast=amp_autocast, model_type=model_type)                   
                                                                                                    
            # record the best results                                                                 
            if (ema_val_err) < best_ema_val_err:                                                      
                best_ema_val_err = ema_val_err                                                        
                best_ema_test_err = ema_test_err                                                      
                best_ema_epoch = epoch                                                                
                                                                                                    
            info_str = f'Epoch: [{epoch}] Target: [{target}] '                                   
            info_str += f'EMA val MAE: {ema_val_err:.5f}, '                                           
            info_str += f'EMA test MAE: {ema_test_err:.5f}, '                                         
            info_str += f'Time: {time.perf_counter() - epoch_start_time:.2f}s'                        
            logger.info(info_str)                                                                       
                                                                                                    
            info_str = f'Best EMA -- epoch={best_ema_epoch}, '                                        
            info_str += f'val MAE: {best_ema_val_err:.5f}, '                                          
            info_str += f'test MAE: {best_ema_test_err:.5f}.\n'                                         
            logger.info(info_str)                                                                       
                                                                                                    
        if epoch % 10 == 0:                                                                           
            torch.save({                                                                              
                    'epoch': epoch,                                                                   
                    'model_state_dict': model.state_dict(),                                           
                    'optimizer_state_dict': optimizer.state_dict(),                                   
                    'scheduler_state_dict': scheduler.state_dict(),                                
                    'loss': train_err,                                                                
                    },                                                                                
                    os.path.join(os.getcwd(), f'checkpoints/Checkpoint_{model_type}.pth.tar'))   
        
        del val_err, test_err, train_err
        torch.cuda.empty_cache()
        gc.collect() 