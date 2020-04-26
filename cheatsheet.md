
Markdown:
* [cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)


### Jupyter notebook:

#### Installing packages
* [Installing Python Packages from a Jupyter Notebook](https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/)
```bash
!{sys.executable} -m pip install efficientnet-pytorch
```

#### notebook settings
```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

%reload_ext autoreload
%autoreload 1
%matplotlib inline
```

Add kernel to jupyter
```bash
conda install -c conda-forge jupyterlab
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```


conda setup:
```bash
conda env create -n 'name'
```
```bash
conda activate 'name'
```


docker:

stop all containers:
```docker kill $(docker ps -q)```

remove all containers
```docker rm $(docker ps -a -q)```

remove all docker images
```docker rmi $(docker images -q)```