import numpy as np
from sklearn import cross_validation
import csv
from collections import Counter

class DataLoader(object):
	"""
	Class for loading data from wjazzd.db
	Data has been loaded with necessary info into csv file
	"""
	def __init__(self):
		
		self.XX = []
		self.Y = []
		self.key = None

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

				past_name = row['filename_sv']

				# Get rid of songs with no key
				if not row['key']:
					continue

				# Note: mode not currently used
				key, mode = self._process_key(row['key'])
				X_i = self._process_Xi(row['tpc_hist_counts'])
				y_i = self._process_yi(row['chords_raw'],row['chord_types_raw'],key)

				# get rid of bars with no chords
				if not y_i:
					continue

				X.append(X_i)
				y.append(y_i)

			if X:
				raw_XX.append(X)
			if y:
				raw_Y.append(y)

		self.XX = self._process_XX(raw_XX)
		self.Y = self._process_Y(raw_Y)

	def _process_key(self,key_raw):
		"""
		Returns absolute pitch class of key_raw and mode

		If key_raw is minor then mode = 'min'
		Everything else treated as major
		Note: mode not currently used
		"""
		try:
			note = self._get_pc(key_raw[0],key_raw[1])
		except IndexError:
			note = self._get_pc(key_raw[0])

		if 'min' in key_raw:
			mode = 'min'
		else:
			mode = 'maj'

		return note, mode


	def _get_pc(self,key,modifier=None):

		if modifier == '#':
			m = 1
		elif modifier == 'b':
			m = -1
		else:
			m = 0

		k = key.capitalize()
		d = ord(k) - 67

		# For A and B
		if d < 0:
			d = 7 + d

		# C,D,E
		if d < 3:
			pc = 2 * d
		# F,G,A,B
		else:
			pc = (2 * d) - 1

		pc = (pc + m) % 12

		return pc
		
		
	def _process_Xi(self,tpc_hist_counts):
		"""
		Convert from string to list
		"""
		return map(int,tpc_hist_counts.split(','))

	def _process_yi(self,chords_raw,chord_types_raw,key):
		"""
		Returns tpc of most occuring chord

		return (3 * tpc) + type where:
			type:
				if maj = 1
				if min = 2
				if dom = 3
		"""
		chord_list = chords_raw.split(',')
		counter = Counter(chord_list)
		mode = counter.most_common(1)
		chord = mode[0][0]

		if chord == 'NA':
			return None

		type_list = chord_types_raw.split(',')
		idx = chord_list.index(chord)
		chord_type_str = type_list[idx]

		if ('j' in chord_type_str) or ('6' in chord_type_str):
			chord_type = 1
		elif ('-' in chord_type_str) or ('m' in chord_type_str):
			chord_type = 2
		else:
			chord_type = 3

		try:
			pc = self._get_pc(chord[0],chord[1])
		except IndexError:
			pc = self._get_pc(chord[0])

		tpc = (pc - key) % 12

		return (tpc * 3) + chord_type
		

	def _process_XX(self,raw_XX):
		"""
		Does nothing at present
		"""
		return raw_XX

	def _process_Y(self,raw_Y):
		"""
		Does nothing at present
		"""
		return raw_Y

	def generate_train_test(self, partition=0.33):
		
		n = len(self.XX))
		j = int(n - (float(n) * partition))

		XX_train = self.XX[0:j]
		Y_train = self.Y[0:j]

		XX_test = self.XX[j:n]
		Y_test = self.Y[j:n]

		data = Data(XX_train, Y_train, XX_test, Y_test)

		return data

class Data(object):
	"""
	Simple data holder
	"""
	def __init__(self, XX_train, Y_train, XX_test, Y_test):
		self.XX_train = XX_train
		self.Y_train = Y_train
		self.XX_test = XX_test
		self.Y_test = Y_test
		











