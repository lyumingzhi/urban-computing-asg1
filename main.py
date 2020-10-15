import json
import os
from pathlib import Path
import get_data
import numpy as np
import matplotlib
import visualize_wp

from dataset import FloorData
from utils import *
from network import MLP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
from tqdm import tqdm, trange


def train(network, train_dataset):
    sampler = RandomSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset,
                                   sampler=sampler,
                                   batch_size=10,
                                   collate_fn=train_dataset.collate_fn)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    for epoch in range(10):
        epoch_iterator = tqdm(train_data_loader, dec='Interation', disable=False)
        for step, batch in enumerate(epoch_iterator):
            network.train()
            example, label = batch[0], batch[1]
            example.requires_grad_()
            preds = network(example)
            optimizer.zero_grad()
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()


def main():
    dataset = FloorData('./output/site1/B1', './data/site1/B1')
    # floor.parse_date()
    # floor.draw_magnetic()
    # floor.draw_way_points()

    net = MLP(dataset.example[list(dataset.example.keys())[0]].shape[1], 128, 128, dataset.gt.shape[1])
    train(net, dataset)
