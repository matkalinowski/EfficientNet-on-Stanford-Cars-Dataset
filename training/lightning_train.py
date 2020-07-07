import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
from torchvision import transforms

from config.structure import get_data_sources
from model.efficient_net import EfficientNet
from model.efficient_net_lightning import EfficientNetLightning
from structure.efficient_nets import EfficientNets
from training.cars_dataset import CarsDataset
import numpy as np


def perform_training(
        model_info: EfficientNets,
        load_weights=True,
        advprop=False
):
    # early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.0,
    #     patience=10,
    #     verbose=False,
    #     mode='min'
    # )
    model = EfficientNetLightning(model_info.value,
                                  batch_size=16,
                                  load_weights=load_weights,
                                  advprop=advprop)

    trainer = pl.Trainer(fast_dev_run=True,
        # early_stop_callback=early_stop_callback,
                         max_epochs=10)
    trainer.fit(model)


def train(model, loss, device, train_loader, optimizer, epoch, log_interval=10000):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        obj = loss(output, torch.tensor(target, dtype=torch.long, device=device))
        obj.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), obj.item()))


def perform_manual_training(
        model_info: EfficientNets,
        load_weights=True,
        advprop=False
):
    dataset_info = get_data_sources()['stanford']
    dataset_type = 'train'
    image_size = 300
    dataset_location = dataset_info[dataset_type]['location']
    dataset = CarsDataset(dataset_location, dataset_info, image_size)

    split_sizes = (len(dataset) * np.array([.8, .1, .1])).astype(np.int)
    split_sizes[-1] = split_sizes[-1] + (len(dataset) - sum(split_sizes))
    train_data, val, test = random_split(dataset, split_sizes.tolist())

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=25, shuffle=True)

    model = EfficientNet(model_info.value, load_weights, advprop).cuda()
    optimizer = torch.optim.Adam(lr=1e-3, params=model.parameters())
    loss = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda")
    for epoch in range(1):
        print(epoch)
        train(model, loss, device, train_data_loader, optimizer, epoch, )
    print(model)


def main():
    perform_training(EfficientNets.b0)
    # perform_manual_training(EfficientNets.b0)


if __name__ == '__main__':
    main()
