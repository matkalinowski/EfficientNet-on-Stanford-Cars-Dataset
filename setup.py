import setuptools
setuptools.setup(
     name='efficientnet',
     version='0.0.1',
     author="Mateusz Kalinowski",
     description="Minimal implementation of EfficientNet",
     url="https://github.com/matkalinowski/dnn",
     install_requires=[
        'pandas>=1.0.1',
        'python-telegram-bot>=12.5',
        'fastai>=1.0.61',
        'mlflow>=1.8.0'
     ],
     extras_require={
        'dev': [
            'pytorch_memlab',
        ]
     }
 )