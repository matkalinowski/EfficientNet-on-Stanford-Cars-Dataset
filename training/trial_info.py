import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from uuid import UUID, uuid1

import pandas as pd
from torch.optim import Optimizer

from config.structure import get_project_structure
from models.efficient_net.efficient_nets import EfficientNets
from utils.folders import create_date_folder, mkdir_if_not_exists


@dataclass
class TrialInfo:
    model_info: EfficientNets

    load_weights: bool
    advprop: bool
    freeze_pretrained_weights: bool

    epochs: int
    batch_size: int
    initial_lr: float
    optimizer: Optimizer
    optimizer_settings: Dict
    scheduler_settings: Dict
    custom_dropout_rate: float

    num_classes: int
    in_channels: int

    # auto-generated
    output_folder: Path = None
    trial_id: UUID = uuid1()

    def __post_init__(self):
        date_folder = create_date_folder(get_project_structure()['training_trials'])

        self.output_folder = date_folder / str(self.trial_id)
        mkdir_if_not_exists(self.output_folder)

    def get_trial_info(self):
        return dataclasses.asdict(self)

    def drop_trial_info(self, path=None):
        if not path:
            path = get_project_structure()['training_trials'] / 'trials_info.csv'

        dictionary = self.get_trial_info()
        dictionary['index'] = [0]
        df = pd.DataFrame.from_records(dictionary, index='index')
        if path.exists():
            df.to_csv(path, mode='a', header=False)
        else:
            df.to_csv(path, header=True)

    def __str__(self):
        return f'{self.model_info.name}-{str(self.trial_id)}'
