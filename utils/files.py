import time
from pathlib import Path
from typing import Union

from utils.folders import mkdir_if_not_exists


def get_file_path_with_timestamp(directory: Union[Path, str], filename: str, extension: str):
    mkdir_if_not_exists(directory)

    new_file_name = f'{filename}-{time.strftime("%Y_%m_%d-%H_%M_%S")}.{extension}'
    return directory / new_file_name
