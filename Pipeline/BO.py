import gc
import time
import math
import torch
from torch.func import vmap
from scipy.stats import norm


class BayesianOptimiser(object):
    def __init__(self, model_type, alpha='EI', kappa=None, **kwargs):

        if model_type not in ['Variational', 'Laplace', 'GP', 'BGNN']:
            raise NotImplementedError(f"Model type {model_type} not implemented.")
        else:
            self.model_type = model_type

        if alpha not in ['PI', 'EI', 'UCB', 'ES', 'KG']:
            raise NotImplementedError(f"Acquisition function {alpha} not implemented.")

        if kappa is None and alpha == 'UCB':
            kappa = 1.0
        elif kappa is None and alpha in ['PI', 'ES', 'KG', 'EI']:
            kappa = 0.0

        self.acquisition_function = AcquisitionFunction(alpha, kappa)
    
    def query(self, model, virtual_loader, incumbent_y, device, norm_factor=None, logger=None):
        logger.info(f"Incumbent: [{incumbent_y:.2f}].")

        inc = incumbent_y.to(device)
        acFun = vmap(lambda a, b: self.acquisition_function(a, b, inc.item()))

        if self.model_type != 'Laplace' and self.model_type != 'BGNN':                                                                
                model.eval() 

        if self.model_type == 'GP':            
            virtual_data = virtual_loader.to(device)

            pred = model(virtual_data)
            mu_x, sigma_x = pred.mean.to(device), torch.sqrt(pred.variance).to(device)
            
            landscape = acFun(mu_x, sigma_x).flatten().to(device)

        else:
            landscape = torch.Tensor().to(device)
            for idx, data in enumerate(virtual_loader):

                if idx % 10 == 0:
                    batch_start_time = time.perf_counter()
                
                data = data.to(device)
                if self.model_type == "Variational":
                    _, mu_x, sigma_x = model(data)
                else:
                    mu_x, sigma_x = model(data)
                
                
                mu_x, sigma_x = mu_x * norm_factor[1] + norm_factor[0], torch.sqrt(sigma_x) * norm_factor[1]

                landscape = torch.cat([landscape.flatten(), acFun(mu_x.flatten(), sigma_x.flatten()).flatten()], dim=0)

                if idx % 100 == 0:
                    logger.info(f"Querying Batch: [{idx}]. Time: {time.perf_counter() - batch_start_time:.2f}s.")

                del mu_x, sigma_x, data
                torch.cuda.empty_cache()
                gc.collect() 

        logger.info(f"Querying Complete.\n")
        return torch.argmax(landscape)


class AcquisitionFunction(object):                                                                         
        def __init__(self, acquisition_fn, kappa) -> None:
            self.kappa = kappa
            self.acquisition_fn = acquisition_fn      

        def __call__(self, mu_x, sigma_x, incumbent=0.0):
            if self.acquisition_fn == 'PI':
                return self.PI(
                    mu_x=mu_x, sigma_x=sigma_x, kappa=self.kappa, incumbent=incumbent
                    )
            elif self.acquisition_fn == 'EI':
                return self.EI(
                    mu_x=mu_x, sigma_x=sigma_x, kappa=self.kappa, incumbent=incumbent
                    )
            elif self.acquisition_fn == 'UCB':
                return self.UCB(
                    mu_x=mu_x, sigma_x=sigma_x, kappa=self.kappa, incumbent=incumbent
                    )
            elif self.acquisition_fn == 'ES':
                return self.ES(
                    mu_x=mu_x, sigma_x=sigma_x, kappa=self.kappa, incumbent=incumbent
                )
            elif self.acquisition_fn == 'KG':
                return self.KG(
                    mu_x=mu_x, sigma_x=sigma_x, kappa=self.kappa, incumbent=incumbent
                )
            else:
                raise NotImplementedError(f"Acquisition function {self.acquision_fn} not implemented")

        @staticmethod
        def PI(mu_x, sigma_x, kappa, incumbent): 
            gamma_x = (mu_x - (incumbent + kappa)) / sigma_x  
            pi = 0.5 * (1 + torch.erf(gamma_x / math.sqrt(2)))                                                                             
            return pi
        
        @staticmethod
        def EI(mu_x, sigma_x, kappa, incumbent):
            gamma_x = (mu_x - (incumbent + kappa)) / sigma_x                                              
            tmp_erf = torch.erf(gamma_x / math.sqrt(2))                                                                
            tmp_ei_no_std = (
                0.5*gamma_x * (1 + tmp_erf) 
                + torch.exp( -gamma_x**2 / 2) / math.sqrt(2 * torch.pi)
                )                                                  
            ei = sigma_x * tmp_ei_no_std                                                                       
            return ei
        
        @staticmethod
        def UCB(mu_x, sigma_x, kappa, incumbent):
            ucb = (mu_x - incumbent) + kappa * sigma_x                                                    
            return ucb

        @staticmethod
        def ES(mu_x, sigma_x, kappa, incumbent):
            std_improvement = (mu_x - incumbent - kappa) / sigma_x
            pdf = torch.exp(-0.5 * std_improvement**2) / math.sqrt(2 * torch.pi)
            cdf = 0.5 * (1 + torch.erf(std_improvement / math.sqrt(2)))
            ei = sigma_x * (std_improvement * cdf + pdf)
        
            return ei

        @staticmethod
        def KG(mu_x, sigma_x, kappa, incumbent):
            gamma = (mu_x - incumbent) / sigma_x
            Z = (mu_x - incumbent - kappa) / sigma_x
            cdf = 0.5 * (1 + torch.erf(Z / math.sqrt(2)))
            pdf = torch.exp(-0.5 * Z**2) / math.sqrt(2 * torch.pi)
            KG = (mu_x - incumbent) * cdf + sigma_x * pdf
            return KG
                                                                                                           
                                                                                                        
