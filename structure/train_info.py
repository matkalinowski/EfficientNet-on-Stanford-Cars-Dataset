import os
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid1

from config.structure import project_structure
from utils.folders import create_date_folder, mkdir_if_not_exists


@dataclass
class TrialInfo:
    model_type: str
    output_folder: Path = None
    trial_id: UUID = uuid1()

    def __post_init__(self):
        date_folder = create_date_folder(project_structure['training_trials'])

        self.output_folder = date_folder / str(self.trial_id)
        mkdir_if_not_exists(self.output_folder)


