from collections import Counter
from functools import reduce

import pytorch_lightning as pl
import torch
from fvcore.nn import flop_count
from fvcore.nn.jit_handles import get_shape
from pytorch_lightning.callbacks import ModelCheckpoint
from thop import profile
from thop.vision.basic_hooks import zero_ops, count_convNd, count_convNd_ver2
from torch import nn

from models.efficient_net.conv_2d import Conv2dStaticSamePadding, Conv2dDynamicSamePadding
from models.efficient_net.efficient_net import EfficientNet
from models.efficient_net.efficient_nets import EfficientNets
from training.cars_dataset_callback import StanfordCarsDatasetCallback
from training.trial_info import TrialInfo


def perform_training(
        model_info: EfficientNets,
        load_weights=True,
        advprop=False
):
    IMAGE_SIZE = model_info.value.network_params.compound_scalars.resolution
    model = EfficientNet(
        batch_size=3,
        net_info=model_info.value,
        load_weights=load_weights,
        advprop=advprop)

    trial_info = TrialInfo(model_info, load_weights, advprop)
    # neptune_logger = NeptuneLogger(
    #     project_name="matkalinowski/sandbox",
    #     experiment_name="e0"
    # )

    checkpoint = ModelCheckpoint(filepath=str(trial_info.output_folder), period=2, mode='min')
    trainer = pl.Trainer(max_epochs=20, gpus=1,
                         fast_dev_run=True,
                         # logger=neptune_logger, save_last=True,
                         callbacks=[(StanfordCarsDatasetCallback(trial_info))], checkpoint_callback=checkpoint)

    # onnx_file_name = "EfficientNet_b0.onnx"
    # torch_out = torch.onnx.export(model, example_batch_input, onnx_file_name, export_params=True)

    # example_batch_input = torch.rand([1, 3, 224, 224], requires_grad=True)
    # with torch.autograd.profiler.profile() as prof:
    #     model(example_batch_input)
    # # NOTE: some columns were removed for brevity
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    example_batch_input = torch.rand([1, 3,
                                      IMAGE_SIZE,
                                      IMAGE_SIZE])

    flop_results = flop_count(model, (example_batch_input,))
    print(f'This model has: {flop_results}B flops')

    trainer.test(model)


if __name__ == '__main__':
    perform_training(EfficientNets.b3, load_weights=False)
