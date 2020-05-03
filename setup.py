import setuptools
setuptools.setup(
     name='efficientnet',
     version='0.0.1',
     author="Mateusz Kalinowski",
     description="Minimal implementation of EfficientNet",
     url="https://github.com/matkalinowski/dnn",
     install_requires=[
        'pandas>=1.0.1'
     ],
     extras_require={
        'dev': [
            'pytorch_memlab',
        ]
     }
 )