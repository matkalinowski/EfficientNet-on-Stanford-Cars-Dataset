from pathlib import Path


def get_project_structure(root_path=Path('.')):
    data_dir = root_path / 'data'

    input_data = data_dir / 'input'
    output_data = data_dir / 'output'
    training_trials = output_data / 'trials'

    return dict(
        input_data=input_data,
        output_data=output_data,
        logging_dir=output_data / 'logs',
        training_trials=training_trials,
        stanford_data_source=input_data / 'stanford',
    )


def get_data_sources(root_path):
    structure = get_project_structure(root_path)
    devkit = structure['stanford_data_source'] / 'devkit'

    return dict(
        stanford=dict(
            source='http://imagenet.stanford.edu/internal/car196/car_ims.tgz',
            data_dir=structure['stanford_data_source'],
            annotations=dict(
                original_file_path=devkit / 'cars_annos.mat',
                csv_file_path=devkit / 'cars_annos.csv',
                source='https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz',
            ),
            class_names=dict(
                original_file_path=devkit / 'cars_meta.mat',
                csv_file_path=devkit / 'cars_meta.csv',
                source='https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz',
            ),
            num_classes=196,
            in_channels=3,
        ),
    )
