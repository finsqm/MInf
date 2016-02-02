import numpy as np
from sklearn import cross_validation
import csv
from collections import Counter
from sklearn.cross_validation import KFold
from old_hmm import *
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

class DataLoader(object):
	"""
	Class for loading data from wjazzd.db
	Data has been loaded with necessary info into csv file
	"""
	def __init__(self):
		
		self.XX = []
		self.Y = []
		self.keys = []

	def load(self,csv_file_name):

		raw_XX = [] # 3D list (2nd dim is mutable)
		raw_Y = []  # 2D list (2nd dim is mutable)
		raw_AA = []
		raw_MM = []

		with open(csv_file_name) as csv_file:
			reader = csv.DictReader(csv_file,delimiter=';')
			past_name = None
			X = []
			y = []
			A = []
			M = []
			for row in reader:
				# Each row corresponds to a frame (bar)
				# Using 'filename_sv' to determine song boundaries
				if past_name != row['filename_sv']:
					if X:
						raw_XX.append(X)
					if y:
						raw_Y.append(y)
					if A:
						raw_AA.append(A)
					if M:
						raw_MM.append(M)
					X = []
					y = []
					A = []
					M = []

				past_name = row['filename_sv']

				# Get rid of songs with no key
				if not row['key']:
					continue

				# Note: mode not currently used
				key, mode = self._process_key(row['key'])
				self.keys.append(key)
				X_i = self._process_Xi(row['tpc_hist_counts'])
				y_i = self._process_yi(row['chords_raw'],row['chord_types_raw'],key)
				A_i = self._process_Ai(row['tpc_raw'])
				M_i = self._process_Mi(row['metrical_weight'])

				# get rid of bars with no chords
				if not y_i:
					continue

				X.append(X_i)
				y.append(y_i)
				A.append(A_i)
				M.append(M_i)

			if X:
				raw_XX.append(X)
			if y:
				raw_Y.append(y)
			if A:
				raw_AA.append(A)
			if M:
				raw_MM.append(M)

		self.XX = self._process_XX(raw_XX)
		self.Y = self._process_Y(raw_Y)
		self.AA = self._process_AA(raw_AA)
		self.MM = self._process_MM(raw_MM)

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

		return tpc + 1
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
			chord_type = 0
		if ('-' in chord_type_str) or ('m' in chord_type_str):
			chord_type = 1
		else:
			chord_type = 0

		try:
			pc = self._get_pc(chord[0],chord[1])
		except IndexError:
			pc = self._get_pc(chord[0])

		tpc = (pc - key) % 12

		return tpc * 2 + 1 + chord_type  

	def _process_Ai(self,tpc_raw):
		"""
		Returns sequence of notes
		"""
		return self._process_Xi(tpc_raw)

	def _process_Mi(self,metrical_weight):
		"""
		Returns indexes of first and thirs beat notes
		If none then looks at all notes
		"""
		ms = self._process_Xi(metrical_weight)

		indexes = []
		for i, m in enumerate(ms):
			if m > 1:
				indexes.append(i)

		if not indexes:
			for i, m in enumerate(ms):
				indexes.append(i)

		return indexes

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

	def _process_AA(self,raw_AA):
		"""
		Does nothing at present
		"""
		return raw_AA

	def _process_MM(self,raw_MM):
		"""
		Does nothing at present
		"""
		return raw_MM

	def generate_train_test(self, partition=0.33):
		
		n = len(self.XX)
		j = int(n - (float(n) * partition))

		XX_train = self.XX[0:j]
		Y_train = self.Y[0:j]
		keys_train = self.keys[0:j]
		AA_train = self.AA[0:j]
		MM_train = self.MM[0:j]

		XX_test = self.XX[j:n]
		Y_test = self.Y[j:n]
		keys_test = self.keys[j:n]
		AA_test = self.AA[j:n]
		MM_test = self.MM[j:n]

		data = Data(XX_train, Y_train, XX_test, Y_test, \
			keys_train, keys_test, AA_train, AA_test, MM_train, MM_test)

		return data

class Data(object):
	"""
	Simple data holder
	"""
	def __init__(self, XX_train, Y_train, XX_test, \
			Y_test, keys_train, keys_test, AA_train, AA_test, MM_train, MM_test):
		self.XX_train = XX_train
		self.Y_train = Y_train
		self.XX_test = XX_test
		self.Y_test = Y_test
		self.keys_train = keys_train
		self.keys_test = keys_test
		self.AA_train = AA_train
		self.AA_test = AA_test
		self.MM_train = MM_train
		self.MM_test = MM_test

	def cross_val_A(self,n=10):

		AA_train = self.AA_train
		Y_train = self.Y_train
		MM_train = self.MM_train

		L = len(self.AA_train)
		kf = KFold(L,n_folds=n)

		models = []
		scores = []

		c = 0

		for c, (train_indexes, val_indexes) in enumerate(kf):

			logger.debug("On Fold " + str(c))

			aa_train = []
			y_train = []
			mm_train = []
			aa_val = []
			y_val = []
			mm_val = []
			for i in train_indexes:
				aa_train.append(AA_train[i][:])
				y_train.append(Y_train[i][:])
				mm_train.append(MM_train[i][:])
			for j in val_indexes:
				aa_val.append(AA_train[j][:])
				y_val.append(Y_train[j][:])
				mm_val.append(MM_train[j][:])

			model = HMM_A()

			logger.debug(str(len(aa_train)) + "," + str(len(y_train)))
			model.train(aa_train,y_train,mm_train)

			logger.debug("Testing ...")
			count, correct = model.test(aa_val,y_val,mm_val)

			score = float(correct) / float(count)
			logger.debug("Fold " + str(c) + " scored " + str(score))

			models.append(model)
			scores.append(score)

		max_score = max(scores)

		max_index = 0
		for idx, score in enumerate(scores):
			if score == max_score:
				max_index = idx
				break

		return models[max_index]


	def cross_val(self,n=10):
		"""
		n : n-crossvalidation
		"""

		XX_train = self.XX_train
		Y_train = self.Y_train

		L = len(self.XX_train)
		kf = KFold(L,n_folds=n)

		models = []
		scores = []

		c = 0

		for c, (train_indexes, val_indexes) in enumerate(kf):

			logger.debug("On Fold " + str(c))

			xx_train = []
			y_train = []
			xx_val = []
			y_val = []
			for i in train_indexes:
				xx_train.append(XX_train[i][:])
				y_train.append(Y_train[i][:])
			for j in val_indexes:
				xx_val.append(XX_train[j][:])
				y_val.append(Y_train[j][:])

			model = HMM()

			logger.debug(str(len(xx_train)) + "," + str(len(y_train)))
			model.train(xx_train,y_train)

			logger.debug("Testing ...")
			count, correct = model.test(xx_val,y_val)

			score = float(correct) / float(count)
			logger.debug("Fold " + str(c) + " scored " + str(score))

			models.append(model)
			scores.append(score)

		max_score = max(scores)

		max_index = 0
		for idx, score in enumerate(scores):
			if score == max_score:
				max_index = idx
				break

		return models[max_index]









		















