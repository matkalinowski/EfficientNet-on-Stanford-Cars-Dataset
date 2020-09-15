alias python=python3

sudo apt-get update
sudo apt-get upgrade -y

sudo apt-get install python3-distutils -y
sudo apt-get install python3-venv -y
sudo apt-get install gcc python3-dev -y

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py & python get-pip.py

git clone -q https://github.com/matkalinowski/dnn.git

cd ~/dnn || exit
python -m venv ./venv
source ./venv/bin/activate

./venv/bin/pip install grpcio==1.26
./venv/bin/pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
