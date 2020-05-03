import os
import time
from pathlib import Path
from typing import Union


def mkdir_if_not_exists(directory: Union[Path, str]) -> None:
    if isinstance(directory, Path):
        directory = str(directory)
    if not os.path.exists(directory):
        os.makedirs(name=directory)


def create_date_folder(directory: Path) -> Path:
    folder_dir = str(directory / time.strftime("%Y_%m_%d"))
    mkdir_if_not_exists(folder_dir)
    return Path(folder_dir)
