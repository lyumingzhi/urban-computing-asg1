import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from dataset import FloorData, UrbanDataset
from network import MLP
from utils import visualize_heatmap

from sklearn import svm as SVM


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
    # dataset.draw_magnetic()
    # dataset.draw_way_points()
    # dataset.draw_wifi_rssi()
    # print(dataset.example[list(dataset.example.keys())[0]].shape, dataset.gt.shape)
    # print(dataset.gt)

    train_ds = UrbanDataset(dataset, type='train', shuffle=True)
    test_ds = UrbanDataset(dataset, type='test', shuffle=False)
    all_ds = UrbanDataset(dataset, type='all', shuffle=False)

    train_data=train_ds.feature
    train_label=train_ds.label
    test_data=test_ds.feature
    test_label=test_ds.label
    preds=np.zeros((test_data.shape))
    svm=SVM.SVR()
    svm.fit(train_data,train_label[:,0])
    preds[:,0]=svm.predict(test_data)
    svm=SVM.SVR()
    svm.fit(train_data,train_label[:,1])
    preds[:,1]=svm.predict(test_data)
    
    # net = MLP(dataset.feature_length, 256, 128, dataset.output_length)
    # net = train(net, train_ds, test_ds)

    # labels, losses = evaluate(net, all_ds)
    losses=[0 for _ in range(len(preds))]
    visualize(preds, losses, dataset)


if __name__ == '__main__':
    main()
