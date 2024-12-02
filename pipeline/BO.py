import gc
import time
import math
import torch
from torch.func import vmap



class BayesianOptimiser(object):
    def __init__(self, model_type, alpha='EI', kappa=0.0, **kwargs):

        if model_type not in ['Variational', 'Laplace', 'GP']:
            raise NotImplementedError(f"Model type {model_type} not implemented.")
        else:
            self.model_type = model_type

        if kappa is None and alpha == 'UCB':
            kappa = 1.0

        self.acquisition_function = AcquisitionFunction(alpha, kappa)
    
    def query(
        self, 
        decoder, 
        virtual_loader, 
        incumbent_y, 
        device, 
        logger=None, 
        norm_factor=[0, 1]
        ):
        logger.info(f"Incumbent: [{incumbent_y:.2f}].")

        task_mean = norm_factor[0]
        task_std = norm_factor[1]

        acFun = vmap(lambda a, b: self.acquisition_function(a, b, incumbent_y))

        if self.model_type != 'Laplace':
                decoder.eval() 

        landscape = torch.Tensor().to(device)
        idx = 0
        for x, _ in virtual_loader:

            if idx % 100 == 0:
                batch_start_time = time.perf_counter()
            
            x = x.to(device)
            
            if self.model_type == 'Laplace':
                mu_x, var_x = decoder(x)
                mu_x = mu_x.detach() * task_std + task_mean
                sigma_x = var_x.detach().sqrt() * task_std
            else:
                pred = decoder(x)
                mu_x = pred.mean * task_std + task_mean
                sigma_x = pred.variance.sqrt() * task_std

            landscape = torch.cat([landscape.flatten(), acFun(mu_x.flatten(), sigma_x.flatten()).flatten()], dim=0)
            idx +=1

            if idx % 100 == 0:
                logger.info(f"Querying Batch: [{idx}]. Time: {time.perf_counter() - batch_start_time:.2f}s.")   

        logger.info(f"Querying Complete.\n")
        return torch.argmax(landscape)


class AcquisitionFunction(object):                                                                         
        def __init__(self, acquisition_fn, kappa) -> None:
            self.kappa = kappa
            self.acquisition_fn = acquisition_fn      

        def __call__(self, mu_x, sigma_x, incumbent=0.0):
            if self.acquisition_fn == 'EI':
                return self.EI(
                    mu_x=mu_x, sigma_x=sigma_x, kappa=self.kappa, incumbent=incumbent
                    )
            elif self.acquisition_fn == 'UCB':
                return self.UCB(
                    mu_x=mu_x, sigma_x=sigma_x, kappa=self.kappa, incumbent=incumbent
                    )
            else:
                raise NotImplementedError(f"Acquisition function {self.acquisition_fn} not implemented")

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
