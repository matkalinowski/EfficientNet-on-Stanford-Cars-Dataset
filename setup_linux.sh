alias python=python3

sudo apt-get update -y
sudo apt-get upgrade -y

sudo apt-get install python3-distutils -y
sudo apt-get install python3-venv -y
sudo apt-get install gcc python3-dev -y

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

git clone -q https://github.com/matkalinowski/dnn.git

cd ~/dnn || exit
python -m venv ./venv
source ./venv/bin/activate

#./venv/bin/pip install grpcio==1.26
./venv/bin/pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

./venv/bin/pip install gdown
gdown https://drive.google.com/uc?id=1zBJEVYyPa4UteDv0vqxj2cjq0JiiN1vi
tar -zxvf "./car_ims.tgz" -C "data/input/stanford/"

#./venv/bin/pip install dataclasses
export NEPTUNE_API_TOKEN=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYzhhMzc3ODktZTNmNi00Zjg2LTgxNDgtNDQwODBiOTg4ZDAzIn0=