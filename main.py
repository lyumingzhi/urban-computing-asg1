import json
import os
from pathlib import Path
import get_data
import numpy as np
import matplotlib
import visualize_wp
def main():
	 data=get_data.Floor_data('F1')
	 # print(data.waypoint[list(data.waypoint.keys())[0]])
	 visualize_wp.visualize(data)
main()