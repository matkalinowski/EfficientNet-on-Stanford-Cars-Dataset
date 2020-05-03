import time
from pathlib import Path
import os


def get_file_path_with_timestamp(directory: Path, filename: str, extension: str, date_structure=True):
    new_file_name = f'{filename}-{time.strftime("%Y_%m_%d-%H_%M_%S")}.{extension}'
    if date_structure:
        curr_time = time.strftime("%Y_%m_%d-%H_%M_%S")
        folder_dir = directory / curr_time[:10]
        if not os.path.exists(folder_dir):
            os.makedirs(name=str(folder_dir))
        return folder_dir / new_file_name
    else:
        return directory / new_file_name
