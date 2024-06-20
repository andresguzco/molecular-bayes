import gc
import time
import math
import torch
from torch.func import vmap
from nets.Benchmarks import predict


class BayesianOptimiser(object):
    def __init__(self, model_type, alpha='EI', kappa=0.0, **kwargs):
        self.acquisition_function = AcquisitionFunction(alpha, kappa)
        if model_type not in ['Variational', 'Laplace', 'GP', 'BGNN']:
            raise NotImplementedError(f"Model type {model_type} not implemented.")
        else:
            self.model_type = model_type
    
    def query(self, model, virtual_loader, incumbent_y, device, logger=None):
        logger.info(f"Incumbent: [{incumbent_y:.2f}].\n")

        inc = incumbent_y.to(device)
        acFun = vmap(lambda a, b: self.acquisition_function(a, b, inc.item()))

        if self.model_type != 'Laplace':                                                                
                model.eval() 

        if self.model_type == 'GP':            
            virtual_data = virtual_loader.to(device)

            pred = model(virtual_data)
            mu_x, sigma_x = pred.mean.to(device), pred.variance.to(device)
            
            landscape = acFun(mu_x, sigma_x).flatten().to(device)

        else:
            landscape = torch.Tensor().to(device)
            for idx, data in enumerate(virtual_loader):

                if idx % 100 == 0:
                    batch_start_time = time.perf_counter()
                
                data = data.to(device)
                
                if self.model_type == "Variational":
                    _, mu_x, sigma_x = model(data)
                else:
                    mu_x, sigma_x = model(data)

                mu_x, sigma_x = mu_x.detach().to(device), sigma_x.detach().to(device)
                
                landscape = torch.cat((landscape.flatten(), acFun(mu_x.flatten(), sigma_x.flatten()).flatten()), dim=0)

                if idx % 100 == 0:
                    logger.info(f"Querying Batch: [{idx}]. Time: {time.perf_counter() - batch_start_time:.2f}s\n")

                del mu_x, sigma_x, data
                torch.cuda.empty_cache()
                gc.collect() 

        logger.info(f"Querying Complete.\n")
        return torch.argmax(landscape)


class AcquisitionFunction(object):                                                                         
        def __init__(self, acquision_fn, kappa) -> None:
            self.kappa = kappa
            self.acquision_fn = acquision_fn      

        def __call__(self, mu_x, sigma_x, incumbent=0.0):
            if self.acquision_fn == 'PI':
                return self.PI(
                    mu_x=mu_x, sigma_x=sigma_x, kappa=self.kappa, incumbent=incumbent
                    )
            elif self.acquision_fn == 'EI':
                return self.EI(
                    mu_x=mu_x, sigma_x=sigma_x, kappa=self.kappa, incumbent=incumbent
                    )
            elif self.acquision_fn == 'UCB':
                return self.UCB(
                    mu_x=mu_x, sigma_x=sigma_x, kappa=self.kappa, incumbent=incumbent
                    )
            else:
                raise NotImplementedError(f"Acquisition function {self.acquision_fn} not implemented")

        @staticmethod
        def PI(mu_x, sigma_x, kappa, incumbent=0.0): 
            gamma_x = (mu_x - (incumbent + kappa)) / sigma_x  
            pi = 0.5 * (1 + torch.erf(gamma_x / math.sqrt(2.0)))                                                                             
            return pi
        
        @staticmethod
        def EI(mu_x, sigma_x, kappa, incumbent=0.0):
            gamma_x = (mu_x - (incumbent + kappa)) / sigma_x                                              
            tmp_erf = torch.erf(gamma_x / math.sqrt(2))                                                                
            tmp_ei_no_std = (
                0.5*gamma_x * (1 + tmp_erf) 
                + torch.exp( -gamma_x**2 / 2) / math.sqrt(2 * torch.pi)
                )                                                  
            ei = sigma_x * tmp_ei_no_std                                                                       
            return ei
        
        @staticmethod
        def UCB(mu_x, sigma_x, kappa, incumbent=0.0):
            ucb = (mu_x - incumbent) + kappa * sigma_x                                                    
            return ucb
                                                                                                           
                                                                                                        