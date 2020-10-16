import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from dataset import FloorData, UrbanDataset
from network import MLP

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
 

def train(network, train_dataset, test_dataset):
    network.apply(init_weights)
    network.train()
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
    # exit()
    for epoch in range(50):
        epoch_iterator = tqdm(train_data_loader, desc='Interation', disable=False)
        network.train()
        loss_sum = 0.0
        count = 0
        for step, batch in enumerate(epoch_iterator):
            example, label = batch[0], batch[1]
            preds = network(example)
            optimizer.zero_grad()
            network.zero_grad()
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().cpu().item()
            count += 1
        scheduler.step()
        print('[Train] Epoch %d | loss: %.3f' % (epoch, loss_sum / count))

        test_iter = tqdm(test_data_loader, desc='Interation', disable=False)
        network.eval()
        loss_sum = 0.0
        count = 0
        for step, batch in enumerate(test_iter):
            example, label = batch[0], batch[1]
            preds = network(example)
            loss = criterion(preds, label)

            loss_sum += loss.detach().cpu().item()
            count += 1
        print('[Test] Epoch %d | loss: %.3f' % (epoch, loss_sum / count))
    print("Finish training, save model")
    torch.save(network.state_dict(), "urban.%d.%.3f.pth" % (epoch, loss.item()))


def main():
    dataset = FloorData('./output/site1/B1', './data/site1/B1', shuffle=False)
    # floor.parse_date()
    # floor.draw_magnetic()
    # floor.draw_way_points()
    # print(dataset.example[list(dataset.example.keys())[0]].shape,dataset.gt.shape[1])
    # print(dataset.gt)
    train_ds = UrbanDataset(dataset, is_training=True, shuffle=True)
    test_ds = UrbanDataset(dataset, is_training=False, shuffle=False)
    net = MLP(dataset.feature_length, 256, 128, dataset.output_length)
    train(net, train_ds, test_ds)

main()
