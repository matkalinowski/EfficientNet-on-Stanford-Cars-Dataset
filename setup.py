import setuptools
setuptools.setup(
     name='efficientNet',
     version='0.0.1',
     author="Mateusz Kalinowski",
     description="Minimal implementation of EfficientNet",
     url="https://github.com/matkalinowski/dnn",
     packages=setuptools.find_namespace_packages(where='.'),
     install_requires=[
        'pandas>=1.0.1'
     ],
     extras_require={
        'dev': [
            'pytorch_memlab',
        ]
     }
 )