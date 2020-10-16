import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from dataset import FloorData, UrbanDataset
from network import MLP
from utils import visualize_heatmap

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
 

def train(network, train_dataset, test_dataset):
    device = 'cuda' if torch.cuda.is_initialized() else 'cpu'
    network.apply(init_weights)
    network.to(device=device)
    sampler = RandomSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset,
                                   sampler=sampler,
                                   batch_size=64,
                                   pin_memory=torch.cuda.is_initialized())
    test_data_loader = DataLoader(test_dataset, batch_size=16,
                                  pin_memory=torch.cuda.is_initialized())
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = torch.nn.MSELoss()
    avg_loss = 0.0
    # exit()
    for epoch in range(100):
        epoch_iterator = tqdm(train_data_loader, desc='Iteration', disable=False)
        network.train()
        loss_sum = 0.0
        count = 0
        for step, batch in enumerate(epoch_iterator):
            example, label, image = map(lambda x: x.to(device), batch)
            preds = network(example, image)
            optimizer.zero_grad()
            network.zero_grad()
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().cpu().item()
            count += 1

        scheduler.step()
        print('[Train] Epoch %d | loss: %.3f' % (epoch, loss_sum / count))

        test_iter = tqdm(test_data_loader, desc='Iteration', disable=False)
        network.eval()
        loss_sum = 0.0
        count = 0
        for step, batch in enumerate(test_iter):
            example, label = batch[0], batch[1]
            preds = network(example)
            loss = criterion(preds, label)

            loss_sum += loss.detach().cpu().item()
            count += 1
        avg_loss = loss_sum / count
        print('[Test] Epoch %d | loss: %.3f' % (epoch, avg_loss))
    print("Finish training, save model")
    torch.save(network.state_dict(), "urban.%d.%.3f.pth" % (epoch, avg_loss))

    return network


def evaluate(network, test_dataset):
    test_data_loader = DataLoader(test_dataset, batch_size=1,
                                  pin_memory=torch.cuda.is_initialized())
    labels = np.zeros((len(test_dataset), 2), dtype=np.float)
    losses = np.zeros((len(test_dataset)), dtype=np.float)
    print(losses.shape)

    criterion = torch.nn.MSELoss()
    test_iter = tqdm(test_data_loader, desc='Iteration', disable=False)
    network.eval()
    loss_sum = 0.0
    count = 0
    for step, batch in enumerate(test_iter):
        example, label = batch[0], batch[1]
        preds = network(example)
        loss = criterion(preds, label)
        labels[step] = label[0].detach().numpy()
        losses[step] = loss.detach().item()

        loss_sum += loss.detach().cpu().item()
        count += 1
    print('[Test] loss: %.3f' % (loss_sum / count))
    return labels, losses


def visualize(labels, losses, dataset, show=False):
    for idx, val in enumerate(labels):
        labels[idx] = np.array([val[0] * dataset.width_meter, val[1] * dataset.height_meter])
    fig = visualize_heatmap(labels, losses,
                            dataset.floor_plan_filename, dataset.width_meter,
                            dataset.height_meter, colorbar_title='MSE Loss',
                            title='DL-based Localization Error Map', show=show)
    dataset.save_figure(fig, 'output_visualize.jpg')


def main():
    dataset = FloorData('./output/site1/B1', './data/site1/B1', shuffle=False)
    # dataset.parse_data()
    dataset.draw_magnetic()
    dataset.draw_way_points()
    dataset.draw_wifi_rssi()
    # print(dataset.example[list(dataset.example.keys())[0]].shape, dataset.gt.shape)
    # print(dataset.gt)

    train_ds = UrbanDataset(dataset, type='train', shuffle=True)
    test_ds = UrbanDataset(dataset, type='test', shuffle=False)
    all_ds = UrbanDataset(dataset, type='all', shuffle=False)

    net = MLP(dataset.feature_length, 256, 128, dataset.output_length)
    net = train(net, train_ds, test_ds)

    labels, losses = evaluate(net, all_ds)
    visualize(labels, losses, dataset)


if __name__ == '__main__':
    main()
