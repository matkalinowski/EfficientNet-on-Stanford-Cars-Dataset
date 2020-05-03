from pathlib import Path

project_main_dir = Path('.')
data_dir = project_main_dir / 'data'

input_data = data_dir / 'input'
output_data = data_dir / 'output'
training_trials = output_data / 'trials'

# train_location = output_data / 'cars_train'

project_structure = dict(
    input_data=input_data,
    output_data=output_data,
    logging_dir=output_data / 'logs',
    training_trials=training_trials,
    stanford_data_source=input_data / 'stanford',
)


telegram = dict(
    TELEGRAM_TOKEN='1069361426:AAH21f3L9g1PD_CJKe3ckKcVqlC00JdAI6c',
    CHAT_ID='368109717'
)

data_sources = dict(
    stanford=dict(
        data_source='https://ai.stanford.edu/~jkrause/cars/car_dataset.html',
        train=dict(
            location=project_structure['stanford_data_source'] / 'cars_train',
            source='http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
        ),
        test=dict(
            location=project_structure['stanford_data_source'] / 'cars_test',
            source='http://imagenet.stanford.edu/internal/car196/cars_test.tgz'
        ),
        labels=dict(
            location=project_structure['stanford_data_source'] / 'devkit' / 'labels_df.csv',
            source='https://raw.githubusercontent.com/morganmcg1/stanford-cars/master/labels_df.csv'
        ),
    ),
)
