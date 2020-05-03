from functools import wraps

from telegram.ext import Updater
from timeit import default_timer as timer
from utils.default_logging import configure_default_logging
from config.structure import get_telegram_settings

telegram_settings = get_telegram_settings()

log = configure_default_logging(__name__)


class TelegramUpdater:
    def __init__(self, chat_id=telegram_settings['CHAT_ID']):
        self.updater = Updater(token=telegram_settings['TELEGRAM_TOKEN'], use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.bot = self.updater.bot
        self.chat_id = chat_id

    def send_photo(self, image):
        self.bot.send_photo(chat_id=self.chat_id, photo=image)

    def send_message(self, msg):
        self.bot.send_message(chat_id=self.chat_id, text=msg)


def training_updater(func):
    telegram_updater = TelegramUpdater()

    @wraps(func)
    def inner(*args, **kwargs):
        telegram_updater.send_message('Training started.')
        start = timer()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            telegram_updater.send_message(f'Your training failed, exc: {e}')
            raise Exception(e)
        elapsed_seconds = timer() - start
        telegram_updater.send_message(f'Training ended. Took about {elapsed_seconds // 60} minutes '
                                      f'({elapsed_seconds} seconds).')
        return result
    return inner


