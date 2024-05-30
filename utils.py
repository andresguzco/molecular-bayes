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


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.local_rank = 0
        return

    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
