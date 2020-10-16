import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from dataset import FloorData
from network import MLP

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
 

def train(network, train_dataset):
    network.apply(init_weights)
    network.train()
    sampler = RandomSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset,
                                   sampler=sampler,
                                   batch_size=16,
                                   collate_fn=train_dataset.collate_fn)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = torch.nn.MSELoss()
    # exit()
    for epoch in range(50):
        epoch_iterator = tqdm(train_data_loader, desc='Interation', disable=False)
        for step, batch in enumerate(epoch_iterator):
            example, label = batch[0], batch[1]
            preds = network(example)
            optimizer.zero_grad()
            network.zero_grad()
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print('Epoch %d | loss: %.3f' % (epoch, loss.item()))
    print("Finish training, save model")
    torch.save(network.state_dict(), "urban.%d.%.3f.pth" % (epoch, loss.item()))


def main():
    dataset = FloorData('./output/site1/B1', './data/site1/B1')
    # floor.parse_date()
    # floor.draw_magnetic()
    # floor.draw_way_points()
    # print(dataset.example[list(dataset.example.keys())[0]].shape,dataset.gt.shape[1])
    # print(dataset.gt)
    net = MLP(dataset.example[list(dataset.example.keys())[0]].shape[0],
              256, 128, dataset.gt.shape[1])
    train(net, dataset)

main()
