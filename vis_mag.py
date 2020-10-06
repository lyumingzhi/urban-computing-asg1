import glob
import os
import shutil

from utils import *


class FloorData(object):
  """Class represents one floor.
  """
  def __init__(self, output_path, path='./data/site1/B1'):
    if not os.path.isdir(output_path):
      shutil.makedirs(output_path)
    self.path = path
    self.path_data_dir = self.path = path + '/path_data_files'
    self.floor_plan_filename = self.path = path + '/floor_image.png'
    self.floor_info_filename = self.path = path + '/floor_info.json'


  def parse_date(self):
    path_filenames = glob.glob(self.path_data_dir, "*.txt")
    mwi_datas = calibrate_magnetic_wifi_ibeacon_to_position(path_filenames)
    self.data = mwi_datas

  def draw_magnetic(self):
    if not hasattr(self, 'data'):
      self.parse_date()

    magnetic_strength = extract_magnetic_strength(self.data)
    heat_positions = np.array(list(magnetic_strength.keys()))
    heat_values = np.array(list(magnetic_strength.values()))
    fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, colorbar_title='mu tesla', title='Magnetic Strength', show=True)
    html_filename = f'{magn_image_save_dir}/magnetic_strength.html'
    html_filename = str(Path(html_filename).resolve())
    save_figure_to_html(fig, html_filename)


if __name__ == '__main__':
  floor = FloorData('./output/site1/B1', './data/site1/B1')
  floor.parse_date()
  floor.draw_magnetic()
