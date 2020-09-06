from pathlib import Path

import pandas as pd
from mat4py import loadmat


def prepare_raw_data(data_root=Path('data/input/stanford')):
    devkit = data_root / 'devkit'
    cars_meta = pd.DataFrame(loadmat(devkit / 'cars_meta.mat'))
    cars_meta.to_csv(devkit / 'cars_meta.csv', index=False)

    cars_annos = pd.DataFrame(loadmat(devkit / 'cars_annos.mat')['annotations'])
    cars_annos['class'] -= 1
    cars_annos.to_csv(devkit / 'cars_annos.csv', index=False)


if __name__ == '__main__':
    prepare_raw_data()