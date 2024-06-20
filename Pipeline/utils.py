'''
    Distributed training related functions.

    From DeiT.
'''
import torch
import matplotlib.pyplot as plt

from scipy.stats import ecdf

def reset_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def plotter(incumbents, selections, maxtrain, maxvirt, maxall, wandb):    
    x = range(len(incumbents))

    plt.figure(figsize=(10, 6))
    plt.plot(x, incumbents.numpy(), label='Incumbent', color='black')
    plt.plot(x, selections.numpy(), label='Selected', color='red')
    plt.axhline(y=maxtrain, color='y', linestyle='dashed', label=f'Max Train data: {maxtrain}')
    plt.axhline(y=maxvirt, color='g', linestyle='dashed', label=f'Max Virtual data: {maxvirt}')
    plt.axhline(y=maxall, color='b', linestyle='dashed', label=f'Max All data: {maxall}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('BO Results')
    plt.legend()
    wandb.log({"BO Results": plt})
    plt.close()


def calibration_plot(
        pred_mu: torch.Tensor, 
        obs: torch.Tensor,
        wandb=None, 
        logger=None
        ):

    obs_ecf = ecdf(obs.numpy())
    obs_ci = obs_ecf.cdf.confidence_interval(confidence_level=0.95)

    pred_ecf = ecdf(pred_mu.numpy())
    pred_ci = pred_ecf.cdf.confidence_interval(confidence_level=0.95)

    if wandb:
        plt.figure(figsize=(10, 10))
        plt.plot(obs_ecf.cdf.quantiles, obs_ecf.cdf.probabilities, color="black", label="Observations")
        plt.plot(obs_ecf.cdf.quantiles, obs_ci.low.probabilities, color="black", linestyle='dashed', label="Observations CI")
        plt.plot(obs_ecf.cdf.quantiles, obs_ci.high.probabilities, color="black", linestyle='dashed')
        plt.plot(pred_ecf.cdf.quantiles, pred_ecf.cdf.probabilities, color="red", label="Predictions")
        plt.plot(pred_ecf.cdf.quantiles, pred_ci.low.probabilities, color="red", linestyle='dashed', label="Predictions CI")
        plt.plot(pred_ecf.cdf.quantiles, pred_ci.high.probabilities, color="red", linestyle='dashed')
        plt.title("Calibration Plot")
        plt.legend()
        wandb.log({"Comparison Quantiles": plt})
        plt.close()

    # if logger:
    #     logger.info(f"Observations Quantiles: {obs_ecf.cdf.quantiles}")
    #     logger.info(f"Observations CI: {obs_ci}")
    #     logger.info(f"Prediction Quantiles: {pred_ecf.cdf.quantiles}")
    #     logger.info(f"Prediction CI: {pred_ci}")
