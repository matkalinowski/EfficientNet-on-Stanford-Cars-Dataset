from fastai.basic_train import Learner
from telegram.ext import Updater
from telegram.ext import CommandHandler
from timeit import default_timer as timer
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)

from config.structure import telegram


class TelegramUpdater:
    def __init__(self, chat_id=telegram['CHAT_ID']):
        log.info('Initializing telegram updater.')
        self.updater = Updater(token=telegram['TELEGRAM_TOKEN'], use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.bot = self.updater.bot
        self.chat_id = chat_id

        self.dispatcher.add_handler(CommandHandler('getid', self._get_id))

        self.updater.start_polling()

    # def howis(self, update, context):
    #     context.bot.send_message(chat_id=update.effective_chat.id, text="Your training is fine, wait 10 more hours")

    def _get_id(self, update):
        self.send_message(update.message.chat_id)

    def send_photo(self, image):
        self.bot.send_photo(chat_id=self.chat_id, photo=image)

    def send_message(self, msg):
        self.bot.send_message(chat_id=self.chat_id, text=msg)


def training_updater(func):
    tu = TelegramUpdater()

    def wrapper(*args, **kwargs):
        tu.send_message('Training started.')
        start = timer()
        # try:
        result = func(*args, **kwargs)
        # except Exception as e:
            # tu.send_message(f'Your training failed, exc: {e}')
            # raise Exception(e)
        tu.send_message(f'Training ended. Took about {(timer() - start)//60} minutes.')
        return result
    return wrapper


def send_learning_update(tu: TelegramUpdater, learn: Learner):
    tu.send_message(f'val_loss: {learn.recorder.val_losses}.\nacc {[m[0].item() for m in learn.recorder.metrics]}')