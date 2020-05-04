from pathlib import Path


def get_project_structure():
    data_dir = Path('.') / 'data'

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


def get_telegram_settings():
    return dict(
        TELEGRAM_TOKEN='1069361426:AAH21f3L9g1PD_CJKe3ckKcVqlC00JdAI6c',
        CHAT_ID='368109717'
    )


def get_data_sources():
    structure = get_project_structure()

    return dict(
        stanford=dict(
            data_source='https://ai.stanford.edu/~jkrause/cars/car_dataset.html',
            train=dict(
                location=structure['stanford_data_source'] / 'cars_train',
                source='http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
            ),
            test=dict(
                location=structure['stanford_data_source'] / 'cars_test',
                source='http://imagenet.stanford.edu/internal/car196/cars_test.tgz'
            ),
            labels=dict(
                location=structure['stanford_data_source'] / 'devkit' / 'labels_df.csv',
                source='https://raw.githubusercontent.com/morganmcg1/stanford-cars/master/labels_df.csv'
            ),
        ),
    )
