class GlacierArea():

	name = ""
	max_velocity = 0.0
	shape_path = ""
	corners = []
	template_dim = 95
	area_sampling = 20
	center = []
	min_delta_time = 12
	max_delta_time = 48

	def __init__(self, data):
		''' Class for handling the glacier area '''

		param_list = data.split("\t")
		self.name = param_list[0]

		try:
			UL = self.read_corner_coord(param_list[1])
			LR = self.read_corner_coord(param_list[2])
			UR = [ LR[0], UL[1] ]
			LL = [ UL[0], LR[1] ]

			self.center = self.read_corner_coord(param_list[3])

			#UR = self.read_corner_coord(param_list[2])
			#LL = self.read_corner_coord(param_list[3])
			
			self.corners = [UL,UR,LL,LR]
			#print self.corners

		except:
			raise ValueError("Invalid corners data ")

		try:
			self.max_velocity = float(param_list[4])
		except:
			#print "Error: Invalid max velocity"
			raise ValueError("Invalid max velocity")


		try:
			self.min_delta_time = float(param_list[5])
		except:
			#print "Error: Invalid max velocity"
			raise ValueError("Invalid min delta time")


		try:
			self.max_delta_time = float(param_list[6])
		except:
			#print "Error: Invalid max velocity"
			raise ValueError("Invalid max delta time")


		try:
			self.template_dim = int(param_list[7])
		except:
			#print "Error: Invalid max velocity"
			raise ValueError("Invalid template dim")

		try:
			self.area_sampling = int(param_list[8])
		except:
			#print "Error: Invalid max velocity"
			raise ValueError("Invalid area sampling")

		self.shape_path = param_list[9]

	def read_corner_coord(self, elem):
		elem = elem.replace("[","").replace("]","").split(",")
		elem = map(float, elem)
		return elem

	def get_corners(self):
		return self.corners

	def get_center(self):
		return self.center

	def get_shape_path(self):
		return self.shape_path

	def get_glacier_name(self):
		return self.name 

	def get_template_dim(self):
		return self.template_dim

	def get_area_sampling(self):
		return self.area_sampling 

	def get_max_velocity(self):
		return self.max_velocity 

	def get_min_delta_time(self):
		return self.min_delta_time 

	def get_max_delta_time(self):
		return self.max_delta_time

	def get_param_string(self):
		return "{}_T{:d}_V{:d}_S{:d}".format(self.name, int(self.template_dim), int(self.max_velocity), int(self.area_sampling))


def Areas_from_txt(path):
	with open(path) as f:
		lines = f.readlines()[1:]  

		Areas = []

		for line in lines:
			if not line[:1] == '#':
				Areas.append(GlacierArea(line.replace("\n","")))

		return Areas
