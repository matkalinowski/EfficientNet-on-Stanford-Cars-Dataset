import time
from pathlib import Path
from typing import Union, Optional

import pandas as pd

from utils.default_logging import configure_default_logging
from utils.folders import mkdir_if_not_exists

log = configure_default_logging(__name__)


def form_final_path(directory, filename, compression: Optional[str]):
    extension = 'csv'
    if compression:
        extension += compression
    mkdir_if_not_exists(directory)
    final_path = directory / f'{filename}.{extension}'
    return final_path


def get_file_path_with_timestamp(directory: Union[Path, str], filename: str, extension: str):
    mkdir_if_not_exists(directory)

    new_file_name = f'{filename}-{time.strftime("%Y_%m_%d-%H_%M_%S")}.{extension}'
    return directory / new_file_name


def save_csv(df: pd.DataFrame, directory: Path, filename: str, compression: Optional[str] = 'bz2',
             append=True) -> None:
    final_path = form_final_path(directory, filename, compression)
    if final_path.exists() and append:
        log.info(f'Appending file {final_path}')
        df.to_csv(final_path, compression=compression, mode='a', header=False)
    else:
        log.info(f'Storing file in: {final_path}')
        df.to_csv(final_path, compression=compression)


def read_csv(directory: Path, filename: str, compression: str = 'bz2') -> pd.DataFrame:
    final_path = form_final_path(directory, filename)
    log.info(f'Reading file from: {final_path}')
    return pd.read_csv(final_path, compression=compression)
