import glob
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import logging

from utils import *


class FloorData(object):
  """Class represents one floor.
  """
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

  def parse_date(self):
    self.logger.info("Parse data from %s" % self.path)
    path_filenames = glob.glob(os.path.join(self.path_data_dir, "*.txt"))
    # Raw data, used to plot positions
    self.raw_data = read_data_files(path_filenames)
    # Calibrated data, used to draw magnetic, wifi, etc.
    self.data = calibrate_magnetic_wifi_ibeacon_to_position(self.raw_data)

  def save_figure(self, fig, name):
    filename = os.path.abspath(os.path.join(
      self.output_path, name))
    # fig.set_size_inches(18.5, 10.5)
    fig.savefig(filename, dpi=500, quality=30)

  def draw_magnetic(self, show=False):
    if not hasattr(self, 'data'):
      self.parse_date()

    magnetic_strength = extract_magnetic_strength(self.data)
    heat_positions = np.array(list(magnetic_strength.keys()))
    heat_values = np.array(list(magnetic_strength.values()))
    fig = visualize_heatmap(heat_positions, heat_values, 
                            self.floor_plan_filename, self.width_meter,
                            self.height_meter, colorbar_title='mu tesla',
                            title='Magnetic Strength', show=show)
    self.save_figure(fig, 'Magnetic.jpg')
    # html_filename = os.path.abspath(os.path.join(
    #     self.output_path, 'magnetic_strength.html'))
    # save_figure_to_html(fig, html_filename)


  def draw_way_points(self, show=False):
    fig, ax = plt.subplots()
    waypoint = []

    for filename, path_data in self.raw_data.items():
      waypoint.append(path_data.waypoint[:, 1:3])
    for wp in waypoint:
      ax.scatter(wp[:, 0], wp[:, 1], linewidths=0.5)

    im = plt.imread(self.floor_plan_filename)
    ax.imshow(im, extent=[0, self.width_meter, 0, self.height_meter])
    if show:
      plt.show()
    self.save_figure(fig, 'WayPoints.jpg')


if __name__ == '__main__':
  floor = FloorData('./output/site1/B1', './data/site1/B1')
  floor.parse_date()
  floor.draw_magnetic()
  floor.draw_way_points()
