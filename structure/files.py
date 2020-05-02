import time
from pathlib import Path


def generate_file_path_with_timestamp(directory: Path, filename: str, extension: str):
    return directory / f'{filename}-{time.strftime("%Y_%m_%d-%H_%M_%S")}.{extension}'