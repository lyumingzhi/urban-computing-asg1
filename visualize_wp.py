import matplotlib.pyplot as plt

floor_data_dir = './data/site1/F1'		
floor_plan_filename = floor_data_dir + '/floor_image.png'
def visualize(data):
	im= plt.imread(floor_plan_filename)
	fig,ax = plt.subplots()
	ax.imshow(im,extent=[0,300,0,400])
	plt.show()

	# way_points=data.way_points
	# for key, point in way_points.items():
	# 	pass
	# 	
visualize([])