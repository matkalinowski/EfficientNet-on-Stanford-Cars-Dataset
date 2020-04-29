from dataclasses import dataclass
from datetime import time

import torch
from fastai.basic_train import Learner, LearnerCallback
from fastai.callback import annealing_linear, annealing_exp, CallbackList
from fastai.callbacks import TrainingPhase, GeneralScheduler
from fastai.core import Floats, defaults, Tuple, Optional, listify, Any, plt
from fastai.metrics import accuracy, LabelSmoothingCrossEntropy

import pandas as pd
from config.structure import data_sources, project_structure
from fastai.vision import (get_transforms, ImageList, ResizeMethod, annealing_cos)

from utils.telegram import send_learning_update, TelegramUpdater, training_updater


def main():
    dataset = data_sources['stanford']

    labels = pd.read_csv(str(dataset['labels']['location']))

    imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    car_tfms = get_transforms()
    SZ = 300

    data = ImageList.from_df(labels[labels.is_test == 0], dataset['train']['location'],
                             cols='filename').split_by_rand_pct(.2).label_from_df(cols='class_name')

    data = (data.transform(car_tfms,
                           size=SZ,
                           resize_method=ResizeMethod.SQUISH,
                           padding_mode='reflection')
            .databunch()
            .normalize(imagenet_stats))

    # By @muellerzr on the fastai forums:
    # https://forums.fast.ai/t/meet-mish-new-activation-function-possible-successor-to-relu/53299/133
    def FlatCosAnnealScheduler(learn, lr: float = 4e-3, tot_epochs: int = 1, moms: Floats = (0.95, 0.999),
                               start_pct: float = 0.72, curve='cosine'):
        "Manage FCFit trainnig as found in the ImageNette experiments"
        n = len(learn.data.train_dl)
        anneal_start = int(n * tot_epochs * start_pct)
        batch_finish = ((n * tot_epochs) - anneal_start)
        if curve == "cosine":
            curve_type = annealing_cos
        elif curve == "linear":
            curve_type = annealing_linear
        elif curve == "exponential":
            curve_type = annealing_exp
        else:
            raise Exception(f"annealing type not supported {curve}")

        phase0 = TrainingPhase(anneal_start).schedule_hp('lr', lr).schedule_hp('mom', moms[0])
        phase1 = TrainingPhase(batch_finish).schedule_hp('lr', lr, anneal=curve_type).schedule_hp('mom', moms[1])
        phases = [phase0, phase1]
        return GeneralScheduler(learn, phases)

    def fit_fc(learn: Learner, tot_epochs: int = None, lr: float = defaults.lr,
               moms: Tuple[float, float] = (0.95, 0.85), start_pct: float = 0.72,
               wd: float = None, callbacks: Optional[CallbackList] = None, show_curve: bool = False) -> None:
        "Fit a model with Flat Cosine Annealing"
        max_lr = learn.lr_range(lr)
        callbacks = listify(callbacks)
        callbacks.append(FlatCosAnnealScheduler(learn, lr, moms=moms, start_pct=start_pct, tot_epochs=tot_epochs))
        learn.fit(tot_epochs, max_lr, wd=wd, callbacks=callbacks)

    @dataclass
    class SimpleRecorder(LearnerCallback):
        learn: Learner
        telegram_updater: TelegramUpdater = TelegramUpdater()

        def on_train_begin(self, **kwargs: Any) -> None:
            self.losses = []

        def on_step_end(self, iteration: int, last_loss, **kwargs):
            self.losses.append(last_loss)

        def on_epoch_end(self, last_loss, smooth_loss, **kwarg):
            print('Epoch ended', last_loss, smooth_loss)
            send_learning_update(self.telegram_updater, self.learn)

        def plot(self, **kwargs):
            losses = self.losses
            iterations = range(len(losses))
            fig, ax = plt.subplots(1, 1)
            ax.set_ylabel('Loss')
            ax.set_xlabel('Iteration')
            ax.plot(iterations, losses)

    @training_updater
    def perform_EfficientNet_training(model, epochs=40):

        learn = Learner(data,
                        model=model,
                        wd=1e-3,
                        bn_wd=False,
                        true_wd=True,
                        metrics=[accuracy],
                        loss_func=LabelSmoothingCrossEntropy(),
                        callback_fns=SimpleRecorder
                        ).to_fp16()

        fit_fc(learn, tot_epochs=epochs, lr=15e-4, start_pct=0.1, wd=1e-3, show_curve=True)

        model_id = f'{model.net_info.name}-{time.strftime("%Y_%m_%d-%H_%M_%S")}'
        torch.save(model.state_dict(), project_structure['models_location'] / model_id)

        result = {
            'model_name': model_name,
            'model_id': model_id,
            'model': model,
            'learner': learn
        }
        return result

    model_name = 'efficientnet-b0'
    from model.efficient_net import EfficientNet
    model = EfficientNet.from_name(model_name, load_weights=True)

    data.batch_size = 48
    result = perform_EfficientNet_training(model, epochs=1)
    # zobacz co sensownie jest zapisywac, jak to wczytywac itd itp
    # uprosc trening, porozbijaj na pakiety funkcje

    # naprawic:
    # learning_updater, [29.04.20 13: 29]
    # Training
    # ended.Took
    # 268.47143619999997
    # minutes.
    print('Finished')


if __name__ == '__main__':
    main()
