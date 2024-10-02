import os
import torch
import matplotlib.pyplot as plt

from scipy.stats import ecdf
from sklearn.metrics import r2_score


def calibration_plot(
        pred_mu: torch.Tensor,
        pred_sigma: torch.Tensor,
        obs: torch.Tensor,
        wandb=None, 
        logger=None
        ):

    obs_ecf = ecdf(obs.numpy())
    obs_ci = obs_ecf.cdf.confidence_interval(confidence_level=0.95)

    pred_ecf = ecdf(pred_mu.numpy())
    pred_ci = pred_ecf.cdf.confidence_interval(confidence_level=0.95)

    sigma_ecf = ecdf(pred_sigma.numpy())
    
    R2 = r2_score(obs.numpy(), pred_mu.numpy())
    MAE = torch.nn.L1Loss()(obs, pred_mu)

    logger.info(f"R2 Score: {R2}")
    logger.info(f"Mean Absolute Error: {MAE}")

    if wandb:
        wandb.log({
            "R2 Score": R2,
            "Mean Absolute Error": MAE
        })

    if sigma_ecf.cdf.quantiles.shape[0] != pred_ecf.cdf.quantiles.shape[0]:
        pred_ci_upper = pred_ecf.cdf.quantiles + sigma_ecf.cdf.quantiles.mean() * 1.96
        pred_ci_lower = pred_ecf.cdf.quantiles - sigma_ecf.cdf.quantiles.mean() * 1.96
    else:
        pred_ci_upper = pred_ecf.cdf.quantiles + sigma_ecf.cdf.quantiles * 1.96
        pred_ci_lower = pred_ecf.cdf.quantiles - sigma_ecf.cdf.quantiles * 1.96
    
    if wandb:
        plt.figure(figsize=(10, 10))
        plt.plot(obs_ecf.cdf.quantiles, obs_ecf.cdf.probabilities, color="black", label="Observations")
        plt.plot(obs_ecf.cdf.quantiles, obs_ci.low.probabilities, color="black", linestyle='dashed', label="Observations ProbabilityCI")
        plt.plot(obs_ecf.cdf.quantiles, obs_ci.high.probabilities, color="black", linestyle='dashed')
        plt.plot(pred_ecf.cdf.quantiles, pred_ecf.cdf.probabilities, color="red", label="Predictions") 
        plt.plot(pred_ecf.cdf.quantiles, pred_ci.low.probabilities, color="red", linestyle='dashed', label="Prediction Probability CI")
        plt.plot(pred_ecf.cdf.quantiles, pred_ci.high.probabilities, color="red", linestyle='dashed')
        plt.plot(pred_ci_upper, pred_ecf.cdf.probabilities, color="blue", linestyle='dashed', label="Predictions CI")
        plt.plot(pred_ci_lower, pred_ecf.cdf.probabilities, color="blue", linestyle='dashed')
        plt.title("Calibration Plot")
        plt.legend()
        wandb.log({"Comparison Quantiles": plt})
        plt.close()


def save_checkpoint(encoder, decoder, optimizer, scheduler, epoch, train_loss, args):
    path = os.getcwd()
    path += f'/checkpoints/Checkpoint_{args.model_type}_{args.dataset_size // 1000}_'
    path += f'{"Single" if len(args.target) == 1 else "Multiple"}.pth.tar'

    torch.save({
        'epoch': epoch,
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': train_loss,
        'decoder_state_dict': decoder.state_dict()
        }, path)
