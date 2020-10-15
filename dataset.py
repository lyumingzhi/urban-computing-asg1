import glob
import os
import random

import numpy as np
import json
import matplotlib.pyplot as plt
import logging

import torch

from utils import *


class FloorData(object):
    """Class represents one floor."""

    def __init__(self, output_path, path='./data/site1/B1',
                 logger=logging):
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        self.path = path
        self.output_path = output_path
        self.path_data_dir = self.path = path + '/path_data_files'
        self.floor_plan_filename = self.path = path + '/floor_image.png'
        floor_info_filename = self.path = path + '/floor_info.json'
        with open(floor_info_filename) as f:
            self.floor_info = json.load(f)

        self.width_meter = self.floor_info["map_info"]["width"]
        self.height_meter = self.floor_info["map_info"]["height"]
        self.logger = logger
        self.parse_data()

    def parse_data(self):
        self.logger.info("Parse data from %s" % self.path)
        path_filenames = glob.glob(os.path.join(self.path_data_dir, "*.txt"))
        # Raw data, used to plot positions
        self.raw_data = read_data_files(path_filenames)
        # Calibrated data, used to draw magnetic, wifi, etc.
        self.data = calibrate_magnetic_wifi_ibeacon_to_position(self.raw_data)

        magnetic_strength = extract_magnetic_strength(self.data)
        wifi_rssi = extract_wifi_rssi(self.data)  # {wifi id: {pos1: value,pos2:value...}...}
        ibeacon_rssi = extract_ibeacon_rssi(self.data)  # {ibeacon id: {pos1: value,pos2:value...}...}
        wifi_count = extract_wifi_count(self.data)

        example = {}
        for position, magnet_s in magnetic_strength.items():
            if position not in example.keys():
                example[position] = []
            example[position].append(magnet_s)

        # wifi_rssi_matrix=np.zeros((len(list(wifi_rssi.keys()))))
        position_wifi_rssi = {}
        for wifi_id, pos_rssi_dict in wifi_rssi.items():
            for pos, rssi in pos_rssi_dict.items():
                if pos not in position_wifi_rssi:
                    position_wifi_rssi[pos] = np.zeros((len(list(wifi_rssi.keys())),))
                position_wifi_rssi[pos][list(wifi_rssi.keys()).index(wifi_id)] = rssi[
                    0]  # rssi[0]=strength rssi[1]=count

        position_ibeacon_rssi = {}
        for ibeacon_id, pos_rssi_dict in ibeacon_rssi.items():
            for pos, rssi in pos_rssi_dict.items():
                if pos not in position_ibeacon_rssi:
                    position_ibeacon_rssi[pos] = np.zeros((len(list(ibeacon_rssi.keys())),))
                position_ibeacon_rssi[pos][list(ibeacon_rssi.keys()).index(ibeacon_id)] = rssi[
                    0]  # rssi[0]=strength rssi[1]=count

        for pos in example.keys():
            if pos not in position_wifi_rssi.keys() or pos not in position_ibeacon_rssi.keys():
                continue
            example[pos] = np.append(np.array(example[pos]), np.array(position_wifi_rssi[pos]))
            example[pos] = np.append(np.array(example[pos]), np.array(position_ibeacon_rssi[pos]))
        self.example = example
        self.gt = np.array(list(example.keys()))

    def save_figure(self, fig, name):
        filename = os.path.abspath(os.path.join(
            self.output_path, name))
        # fig.set_size_inches(18.5, 10.5)
        fig.savefig(filename, dpi=500, quality=30)

    def draw_magnetic(self, show=False):
        if not hasattr(self, 'data'):
            self.parse_data()

        magnetic_strength = extract_magnetic_strength(self.data)
        heat_positions = np.array(list(magnetic_strength.keys()))
        heat_values = np.array(list(magnetic_strength.values()))
        fig = visualize_heatmap(heat_positions, heat_values,
                                self.floor_plan_filename, self.width_meter,
                                self.height_meter, colorbar_title='mu tesla',
                                title='Magnetic Strength', show=show)
        self.save_figure(fig, 'Magnetic.jpg')

    def draw_way_points(self, show=False):
        fig, ax = plt.subplots()
        waypoint = []

        for filename, path_data in self.raw_data.items():
            waypoint.append(path_data.waypoint[:, 1:3])
        for wp in waypoint:
            ax.scatter(wp[:, 0], wp[:, 1], linewidths=0.5)
        ax.title.set_text('WayPoints')
        im = plt.imread(self.floor_plan_filename)
        ax.imshow(im, extent=[0, self.width_meter, 0, self.height_meter])
        if show:
            plt.show()
        self.save_figure(fig, 'WayPoints.jpg')

    def draw_wifi_rssi(self, show=False):
        wifi_rssi = extract_wifi_rssi(self.data)
        wifi_bssids = list(wifi_rssi.keys())
        target_wifi_list = random.sample(wifi_bssids, k=3)
        for target_wifi in target_wifi_list:
            heat_positions = np.array(list(wifi_rssi[target_wifi].keys()))
            heat_values = np.array(list(wifi_rssi[target_wifi].values()))[:, 0]
            fig = visualize_heatmap(heat_positions, heat_values, self.floor_plan_filename, self.width_meter,
                                    self.height_meter,
                                    colorbar_title='dBm', title=f'Wifi: {target_wifi} RSSI', show=show)
            self.save_figure(fig, f'Wifi_RSSI_{target_wifi.replace(":", "-")}.jpg')

    # def create_example(self):

    def __len__(self):
        return len(list(self.data.keys()))

    def __getitem__(self, index):
        return self.example[list(self.example.keys())[index]], self.gt[index]

    def collate_fn(self, batch):
        examples, gts = batch[0], batch[1]
        examples = torch.Tensor(examples).view(len(batch), -1)
        gts = torch.Tensor(gts)


if __name__ == '__main__':
    floor = FloorData('./output/site1/B1', './data/site1/B1')
    floor.parse_data()
    floor.draw_magnetic()
    floor.draw_way_points()
    floor.draw_wifi_rssi()
