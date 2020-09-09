# Source:
# https://raw.githubusercontent.com/facebookresearch/SlowFast/0cc82440fee6e51a5807853b583be238bf26a253/slowfast/utils/misc.py

import numpy as np
import psutil
import torch
from fvcore.nn.flop_count import flop_count

from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total


def calculate_model_info(model: torch.nn.Module, image_size: int, color_channels: int = 3):
    """
    Log info, includes number of parameters, gpu usage and gflops.
    Args:
        :param model: model to log the info.
        :param color_channels: number of color channels in the data
        :param image_size: size of the image, it is important in flops counting

        :return pd.Series with logged values
    """
    example_batch_input = torch.rand([1, color_channels, image_size, image_size]).to(model.device)
    flop_results = flop_count(model, (example_batch_input,))

    gpu_mem = gpu_mem_usage()
    total_params = params_count(model)
    total_flops = sum(flop_results[0].values())

    log.info("Model:\n{}".format(model))
    log.info("Params: {:,}".format(total_params))
    log.info("Mem: {:.3f} MB".format(gpu_mem))
    log.info("Flops: {:,} G".format(total_flops))

    return dict(gpu_mem_usage_MB=gpu_mem, params_count=total_params, total_flops=total_flops)
