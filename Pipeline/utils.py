'''
    Distributed training related functions.

    From DeiT.
'''

import os
import torch
import matplotlib.pyplot as plt
import torch.distributed as dist


def calibration_plot(predictions: torch.Tensor, observations: torch.Tensor, num_bins: int = 10, wandb=None, logger=None):
    if predictions.device != observations.device:
        observations = observations.to(predictions.device)

    deciles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    obs_quantiles = [
        torch.quantile(observations, decile, dim=0).item() for decile in deciles
    ]
    pred_deciles = []
    cumulative = 0

    logger.info(f"Observations Quantiles: {obs_quantiles}")

    for i in range(len(deciles) - 1):
        count = (
            (predictions > obs_quantiles[i]) & (predictions < obs_quantiles[i + 1])
            ).sum().item() / predictions.shape[0]
        cumulative += count
        pred_deciles.append(count)
    logger.info(f"Prediction Quantiles: {pred_deciles}")

    plt.figure(figsize=(8, 8))

    plt.plot(deciles[:-1], pred_deciles, marker='o', linestyle='-', label='Calibration curve')
    plt.plot([0, 0], [1, 1], linestyle='--', label='Perfect calibration')

    plt.xlabel('Mean Predicted Value')
    plt.ylabel('Mean Observed Value')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)

    wandb.log({"Calibration Plot": plt})
