import json
import os
from pathlib import Path

import re
import numpy as np

floor_list = ['B1', 'F1', 'F2']
floor_data_dir = './data/site1/F1'
path_data_dir = floor_data_dir + '/path_data_files'
floor_plan_filename = floor_data_dir + '/floor_image.png'
floor_info_filename = floor_data_dir + '/floor_info.json'

save_dir = './output/site1/F1'
path_image_save_dir = save_dir + '/path_images'
step_position_image_save_dir = save_dir
magn_image_save_dir = save_dir
wifi_image_save_dir = save_dir + '/wifi_images'
ibeacon_image_save_dir = save_dir + '/ibeacon_images'
wifi_count_image_save_dir = save_dir


class Floor_data(object):
    """docstring for Floor_data"""

    def __init__(self, floor):
        super(Floor_data, self).__init__()
        self.floor = floor if floor in floor_list else None
        self.acce = {}
        self.acce_uncali = {}
        self.gyro = {}
        self.gyro_uncali = {}
        self.magn = {}
        self.magn_uncali = {}
        self.ahrs = {}
        self.wifi = {}
        self.ibeacon = {}
        self.waypoint = {}
        self.width = None
        self.high = None
        self.get_data()

    def get_data(self):
        path_data_files = list(Path(path_data_dir).resolve().glob('*.txt'))
        for path_file in path_data_files:
            # print('loading file',path_data_files)
            with open(path_file, 'r', encoding='utf-8') as f:
                path_data = f.readlines()
            if path_file in self.acce.keys():
                continue
            self.acce[path_file] = []
            self.acce_uncali[path_file] = []
            self.gyro[path_file] = []
            self.gyro_uncali[path_file] = []
            self.magn[path_file] = []
            self.magn_uncali[path_file] = []
            self.ahrs[path_file] = []
            self.wifi[path_file] = []
            self.ibeacon[path_file] = []
            self.waypoint[path_file] = []
            for entry in path_data:
                entry = entry.strip()
                if entry[0] == '#':
                    # print(entry)
                    # exit()
                    continue
                # print(entry)
                entry = re.split('[ \t]', entry)
                # entry=entry.split('')
                # print(entry)
                # exit()
                if entry[1] == 'TYPE_ACCELEROMETER':
                    self.acce[path_file].append([int(entry[0]), float(entry[2]), float(entry[3]), float(entry[4])])
                    continue

                if entry[1] == 'TYPE_ACCELEROMETER_UNCALIBRATED':
                    self.acce_uncali[path_file].append(
                        [int(entry[0]), float(entry[2]), float(entry[3]), float(entry[4])])
                    continue

                if entry[1] == 'TYPE_GYROSCOPE':
                    self.gyro[path_file].append([int(entry[0]), float(entry[2]), float(entry[3]), float(entry[4])])
                    continue

                if entry[1] == 'TYPE_GYROSCOPE_UNCALIBRATED':
                    self.gyro_uncali[path_file].append(
                        [int(entry[0]), float(entry[2]), float(entry[3]), float(entry[4])])
                    continue

                if entry[1] == 'TYPE_MAGNETIC_FIELD':
                    self.magn[path_file].append([int(entry[0]), float(entry[2]), float(entry[3]), float(entry[4])])
                    continue

                if entry[1] == 'TYPE_MAGNETIC_FIELD_UNCALIBRATED':
                    self.magn_uncali[path_file].append(
                        [int(entry[0]), float(entry[2]), float(entry[3]), float(entry[4])])
                    continue

                if entry[1] == 'TYPE_ROTATION_VECTOR':
                    self.ahrs[path_file].append([int(entry[0]), float(entry[2]), float(entry[3]), float(entry[4])])
                    continue

                if entry[1] == 'TYPE_WIFI':
                    sys_ts = entry[0]
                    ssid = entry[2]
                    bssid = entry[3]
                    rssi = entry[4]
                    lastseen_ts = entry[6]
                    wifi_data = [sys_ts, ssid, bssid, rssi, lastseen_ts]
                    self.wifi[path_file].append(wifi_data)
                    continue

                if entry[1] == 'TYPE_BEACON':
                    ts = entry[0]
                    uuid = entry[2]
                    major = entry[3]
                    minor = entry[4]
                    rssi = entry[6]
                    ibeacon_data = [ts, '_'.join([uuid, major, minor]), rssi]
                    self.ibeacon[path_file].append(ibeacon_data)
                    continue

                if entry[1] == 'TYPE_WAYPOINT':
                    self.waypoint[path_file].append([int(entry[0]), float(entry[2]), float(entry[3])])
                    continue
            self.acce[path_file] = np.array(self.acce[path_file])
            self.acce_uncali[path_file] = np.array(self.acce_uncali[path_file])
            self.gyro[path_file] = np.array(self.gyro[path_file])
            self.gyro_uncali[path_file] = np.array(self.gyro_uncali[path_file])
            self.magn[path_file] = np.array(self.magn[path_file])
            self.magn_uncali[path_file] = np.array(self.magn_uncali[path_file])
            self.ahrs[path_file] = np.array(self.ahrs[path_file])
            self.wifi[path_file] = np.array(self.wifi[path_file])
            self.ibeacon[path_file] = np.array(self.ibeacon[path_file])
            self.waypoint[path_file] = np.array(self.waypoint[path_file])

        with open(floor_info_filename, 'r') as f:
            floor_info = json.load(f)
        self.width = floor_info["map_info"]["width"]
        self.height = floor_info["map_info"]["height"]
