import numpy as np
from sklearn import cross_validation
import csv

class DataLoader(object):
	"""
	Class for loading data from wjazzd.db
	Data has been loaded with necessary info into csv file
	"""
	def __init__(self):
		
		self.X = []
		self.y = []

	def load(self,csv_file_name):

		raw_XX = [] # 3D list (2nd dim is mutable)
		raw_Y = []  # 2D list (2nd dim is mutable)

		with open(csv_file_name) as csv_file:
			reader = csv.DictReader(csv_file,delimiter=';')
			past_name = None
			X = []
			y = []
			for row in reader:
				# Each row corresponds to a frame (bar)
				# Using 'filename_sv' to determine song boundaries
				if past_name != row['filename_sv']:
					if X:
						raw_XX.append(X)
					if y:
						raw_Y.append(y)
					X = []
					y = []

				key = self._process_key(row['key'])
				X_i = self._process_Xi(row['tpc_hist_counts'],key)
				y_i = self._process_yi(row['chord_raw'],row['chord_types_raw'],key)

				X.append(X_i)
				y.append(y_i)

