alias python=python3

sudo apt-get install python3-distutils
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py & python get-pip.py

git clone -q https://github.com/matkalinowski/dnn.git

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3-venv -y

cd ~/dnn || exit
python -m venv ./venv
source ./venv/bin/activate

./venv/bin/pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html