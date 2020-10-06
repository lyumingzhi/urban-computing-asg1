import glob
import os
import numpy as np
import json

from utils import *


class FloorData(object):
  """Class represents one floor.
  """
  def __init__(self, output_path, path='./data/site1/B1'):
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

  def parse_date(self):
    path_filenames = glob.glob(os.path.join(self.path_data_dir, "*.txt"))
    mwi_datas = calibrate_magnetic_wifi_ibeacon_to_position(path_filenames)
    self.data = mwi_datas

  def draw_magnetic(self):
    if not hasattr(self, 'data'):
      self.parse_date()

    magnetic_strength = extract_magnetic_strength(self.data)
    heat_positions = np.array(list(magnetic_strength.keys()))
    heat_values = np.array(list(magnetic_strength.values()))
    fig = visualize_heatmap(heat_positions, heat_values, 
                            self.floor_plan_filename, self.width_meter,
                            self.height_meter, colorbar_title='mu tesla',
                            title='Magnetic Strength', show=True)
    html_filename = os.path.abspath(os.path.join(
        self.output_path, 'magnetic_strength.html'))
    save_figure_to_html(fig, html_filename)


if __name__ == '__main__':
  floor = FloorData('./output/site1/B1', './data/site1/B1')
  floor.parse_date()
  floor.draw_magnetic()
