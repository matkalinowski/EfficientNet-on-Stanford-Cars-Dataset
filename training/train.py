import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from thop import profile
from thop.vision.basic_hooks import zero_ops
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
        batch_size=10,
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

    # In every epoch, the loop methods are called in this frequency:
    # validation_step() called every batch
    # validation_epoch_end() called every epoch
    # test_dataloader() is only called with .test()
    trainer.fit(model)

    example_batch_input = torch.rand([2, 3,
                                      IMAGE_SIZE,
                                      IMAGE_SIZE])
    custom_flops_calculations(model, example_batch_input)

    # onnx_file_name = "EfficientNet_b0.onnx"
    # torch_out = torch.onnx.export(model, example_batch_input, onnx_file_name, export_params=True)

    # example_batch_input = torch.rand([1, 3, 224, 224], requires_grad=True)
    # with torch.autograd.profiler.profile() as prof:
    #     model(example_batch_input)
    # # NOTE: some columns were removed for brevity
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    print(model)
    trainer.test(model)


def custom_flops_calculations(model, example_batch_input):
    # def count_Conv2dStaticSamePadding(layer: nn.Conv2d, x: (torch.Tensor,), y: torch.Tensor):
    #     count_convNd(layer, x, y)

    def count_linear(layer, x: (torch.Tensor,), y: torch.Tensor):
        batch_size = x[0].size(0)

        weight_ops = layer.weight.nelement()
        bias_ops = layer.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)

        layer.total_ops += torch.DoubleTensor([int(flops)])

    def count_Conv2dStaticSamePadding(layer: nn.Conv2d, x: (torch.Tensor,), y: torch.Tensor):
        batch_size, input_channels, input_height, input_width = x[0].size()
        output_channels, output_height, output_width = y[0].size()

        kernel_ops = layer.kernel_size[0] * layer.kernel_size[1] * (layer.in_channels / layer.groups)
        bias_ops = 1 if layer.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        print(layer)
        print(batch_size, input_channels, input_height, input_width, output_channels, output_height, output_width, kernel_ops, bias_ops, params, flops)
        layer.total_ops += torch.DoubleTensor([int(flops)])

    def count_swish(layer, x: (torch.Tensor,), y: torch.Tensor):
        n_elements = x[0].numel()

        total_exp = n_elements
        total_add = n_elements - 1
        total_div = n_elements
        total_mult = n_elements
        sign_change = n_elements

        # we count activation functions using flops, but library returns macs, macs ~= flops*2
        total_flops = (total_exp + total_add + total_div + total_mult + sign_change) * 2

        layer.total_ops += torch.DoubleTensor([int(total_flops)])

    def default_no_elements(layer, x: (torch.Tensor,), y: torch.Tensor):
        layer.total_ops += torch.DoubleTensor([int(x[0].nelement())])

    print('Custom calculations:')
    macs, params = profile(model.cpu(), inputs=(example_batch_input,),
                           custom_ops={
                               # Swish: count_swish,
                               nn.Linear: count_linear,
                               # nn.ReLU: default_no_elements,
                               nn.ReLU: zero_ops,
                               nn.AdaptiveAvgPool2d: zero_ops,
                               nn.BatchNorm2d: default_no_elements,
                               Conv2dDynamicSamePadding: count_Conv2dStaticSamePadding,
                               Conv2dStaticSamePadding: count_Conv2dStaticSamePadding})
    # in billions
    flops = macs / 2e9
    print(f'{macs=}, {params=}, {flops=}')


if __name__ == '__main__':
    perform_training(EfficientNets.b1, load_weights=False)
