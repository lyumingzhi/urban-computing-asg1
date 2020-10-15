import matplotlib.pyplot as plt
import numpy as np

floor_data_dir = './data/site1/F1'
floor_plan_filename = floor_data_dir + '/floor_image.png'


def visualize(data):
    fig, ax = plt.subplots()

    waypoints = []
    for key, trace in data.waypoint.items():
        waypoints.append(trace[:, 1:3])
    print(waypoints)
    # way_points=np.array(way_points)
    width, height = data.width, data.height
    # print('size of way_points',way_points.shape)
    for waypoint in waypoints:
        ax.scatter(waypoint[:, 0], waypoint[:, 1], linewidths=0.5)

    im = plt.imread(floor_plan_filename)
    ax.imshow(im, extent=[0, width, 0, height])
    plt.show()
# visualize([])
