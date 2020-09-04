import setuptools
setuptools.setup(
     name='efficientnet',
     version='0.0.1',
     author="Mateusz Kalinowski",
     description="Minimal implementation of EfficientNet",
     url="https://github.com/matkalinowski/dnn",
     packages=setuptools.find_packages(),
     # python_requires='>=3.8',
     install_requires=[
          'torch==1.6.0',
          'torchvision==0.7.0',
          'matplotlib==3.3.1',
          'pandas==1.1.1',
          'pytorch-lightning==0.9.0',
          'scikit-learn==0.23.2',
          'fvcore==0.1.1.post20200716',
          'psutil==5.7.2',
          'neptune-client==0.4.119'
     ]
 )