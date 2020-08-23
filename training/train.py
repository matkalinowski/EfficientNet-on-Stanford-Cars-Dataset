from collections import Counter
from functools import reduce

import pytorch_lightning as pl
import torch
from fvcore.nn import flop_count
from fvcore.nn.jit_handles import get_shape
from pytorch_lightning.callbacks import ModelCheckpoint
from thop import profile
from thop.vision.basic_hooks import count_convNd

from models.efficient_net.Swish import Swish
from models.efficient_net.conv_2d import Conv2dStaticSamePadding
from models.efficient_net.efficient_net import EfficientNet
from models.efficient_net.efficient_nets import EfficientNets
from training.cars_dataset_callback import StanfordCarsDatasetCallback
from training.trial_info import TrialInfo
from utils.xtils import calculate_FLOPs_scale, get_model_summary


def perform_training(
        model_info: EfficientNets,
        load_weights=True,
        advprop=False
):
    model = EfficientNet(
        batch_size=10,
        image_size=model_info.value.network_params.global_params.image_size,
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

    example_batch_input = torch.rand([1, 3, 224, 224])

    custom_flops_calculations(model, example_batch_input)

    # onnx_file_name = "EfficientNet_b0.onnx"
    # torch_out = torch.onnx.export(model, example_batch_input, onnx_file_name, export_params=True)

    fvcore_flops(example_batch_input, model)

    calculate_FLOPs_scale(model, input_size=224, use_gpu=False, multiply_adds=False)

    # example_batch_input = torch.rand([1, 3, 224, 224], requires_grad=True)
    # with torch.autograd.profiler.profile() as prof:
    #     model(example_batch_input)
    # # NOTE: some columns were removed for brevity
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    trainer.test(model)


def prod_of_list(values):
    return reduce(lambda x, y: x * y, values)


def get_total_number_of_elements(input):
    shp = get_shape(input)
    return prod_of_list(shp)


def fvcore_flops(example_batch_input, model):
    def count_sigmoid(inputs, outputs):
        n_elements = get_total_number_of_elements(inputs[0])

        total_exp = n_elements
        total_add = n_elements - 1
        total_div = n_elements
        sign_change = n_elements

        total = total_exp + total_add + total_div + sign_change

        return Counter({"sigmoid": total})

    def build_flop_result(operation_name):
        def flop_of_input_size(inputs, outputs):
            n_elements = get_total_number_of_elements(inputs[0])
            return Counter({operation_name: n_elements})

        return flop_of_input_size

    def count_adaptive_avg_pool2d(inputs, outputs):
        shp_x = get_shape(inputs[0])
        shp_y = get_shape(outputs[0])

        kernel = map(lambda x: x // shp_y[1], shp_x[2:])
        total_add = prod_of_list(kernel)
        total_div = 1
        kernel_ops = total_add + total_div
        num_elements = prod_of_list(shp_y)

        total = kernel_ops * num_elements

        return Counter({"adaptive_avg_pool2d": total})

    def count_rand(inputs, outputs):
        return Counter({"rand": get_total_number_of_elements(outputs[0])})

    def count_batch_norm(inputs, outputs):
        n_elements = get_total_number_of_elements(inputs[0])

        total_mult = n_elements
        total_add = (n_elements - 1) * 2
        total_sub = n_elements
        total_div = n_elements
        total_sqrt = n_elements

        # z = gamma * (y - mean) / sqrt(variance + epsilon) + beta
        total = total_mult + total_add + + total_sub + total_div + total_sqrt

        return Counter({"batch_norm": total})

    flop_results = flop_count(model, (example_batch_input,),
                              supported_ops={
                                  # 'aten::sigmoid': count_sigmoid,
                                  # 'aten::rand': count_rand,
                                  'aten::adaptive_avg_pool2d': count_adaptive_avg_pool2d,
                                  # 'aten::mul': build_flop_result('mul'),
                                  # 'aten::add': build_flop_result('add'),
                                  # 'aten::div': build_flop_result('div'),
                                  # 'aten::batch_norm': count_batch_norm,
                                  # 'aten::dropout': build_flop_result('dropout'),
                              })
    flops = sum(flop_results[0].values())
    print(flops)


def custom_flops_calculations(model, example_batch_input):
    def count_Conv2dStaticSamePadding(layer: Conv2dStaticSamePadding, x: (torch.Tensor,), y: torch.Tensor):
        count_convNd(layer, x, y)

    def count_swish(layer, x: (torch.Tensor,), y: torch.Tensor):
        n_elements = x[0].numel()

        total_exp = n_elements
        total_add = n_elements - 1
        total_div = n_elements
        total_mult = n_elements
        sign_change = n_elements

        # we count activation functions using flops, but library returns macs, macs ~= flops*2
        total_flops = (total_exp + total_add + total_div + total_mult+ sign_change) * 2

        layer.total_ops += torch.DoubleTensor([int(total_flops)])

    print('Custom calculations:')
    macs, params = profile(model.cpu(), inputs=(example_batch_input,),
                           custom_ops={
                               Swish: count_swish,
                               Conv2dStaticSamePadding: count_Conv2dStaticSamePadding})
    # in billions
    flops = macs / 2e9
    print(f'{macs=}, {params=}, {flops=}'),
    return example_batch_input


if __name__ == '__main__':
    perform_training(EfficientNets.b0, load_weights=False)
