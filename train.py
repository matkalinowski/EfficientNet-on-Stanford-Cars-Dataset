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
from timeit import default_timer as timer


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


    from telegram.ext import Updater
    from telegram.ext import CommandHandler
    TELEGRAM_TOKEN = '1069361426:AAH21f3L9g1PD_CJKe3ckKcVqlC00JdAI6c'
    CHAT_ID = '368109717'

    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    def howis(update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text="Your training is fine, wait 10 more hours")

    def getid(update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.chat_id)

    dispatcher.add_handler(CommandHandler('howis', howis))
    dispatcher.add_handler(CommandHandler('getid', getid))
    updater.start_polling()
    bot = updater.bot

    # progress_msg_res = bot.send_message(chat_id = CHAT_ID, text = 'Test message')
    # progress_msg_res.edit_text('Iteration 1: loss: 1')

    # embedding_img_path = './images/embedding.png'
    # plot_embeddings(train_embeddings_baseline, train_labels_baseline, plt_store_path=embedding_img_path)
    # bot.send_photo(chat_id = CHAT_ID, photo=open(embedding_img_path, 'rb'))

    def send_message(msg, chat_id=CHAT_ID, bot=bot):
        bot.send_message(chat_id=CHAT_ID, text=msg)

    def telegram_updater(func):
        def wrapper(*args, **kwargs):
            send_message('Training started.')
            start = timer()
            result = func(*args, **kwargs)
            send_message(f'Training ended. Took {timer() - start} minutes.')
            return result

        return wrapper

    def send_update(learn):
        if learn and hasattr(learn, 'recorder'):
            send_message(f'val_loss: {learn.recorder.val_losses}.\nacc {[m[0].item() for m in learn.recorder.metrics]}')
        else:
            print('learn object is not valid')


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

        def on_train_begin(self, **kwargs: Any) -> None:
            self.losses = []

        def on_step_end(self, iteration: int, last_loss, **kwargs):
            self.losses.append(last_loss)

        def on_epoch_end(self, last_loss, smooth_loss, **kwarg):
            print('Epoch ended', last_loss, smooth_loss)
            send_update(self.learn)

        def plot(self, **kwargs):
            losses = self.losses
            iterations = range(len(losses))
            fig, ax = plt.subplots(1, 1)
            ax.set_ylabel('Loss')
            ax.set_xlabel('Iteration')
            ax.plot(iterations, losses)

    @telegram_updater
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
        torch.save(model.state_dict, project_structure.models_location / model_id)

        result = {
            'model_name': model_name,
            'model_id': model_id,
            'model': model,
            'learner': learn
        }

        return result, learn

    model_name = 'efficientnet-b0'

    from model.efficient_net import EfficientNet
    model = EfficientNet.from_name(model_name, load_weights=True)

    data.batch_size = 48
    result, learn = perform_EfficientNet_training(model, epochs=3)
    print('Finished')


if __name__ == '__main__':
    main()
