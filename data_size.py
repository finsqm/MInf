from hmm import *
from data_loader import *
from utils import *
import matplotlib.pyplot as plt
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.tree import *
import logging

import cPickle

logging.basicConfig(filename='chord_tones_test.log',level=logging.DEBUG)
logging.info('Initialising data ...')

with open('all_chords_data.pkl','r') as f:
	data = cPickle.load(f)

XX_train = data.XX_train
Y_train = data.Y_train

print len(XX_train)
print len(XX_train[0])
print len(XX_train[0][0])

chord_tones = get_chord_tones(XX_train, Y_train)

XX_train_ct = []
Y_train_ct = []

for i, song in enumerate(XX_train):
	for j, frame in enumerate(song):
		x_ij = np.zeros(12)
		for k, note in enumerate(frame):
			if chord_tones[i][j][k] == 1:
				x_ij[int(note[0])] = 1
		XX_train_ct.append(x_ij)
		Y_train_ct.append(Y_train[i][j])
		print i, j

XX_train_ct = np.asarray(XX_train_ct)
Y_train_ct = np.asarray(Y_train_ct)

print XX_train_ct.shape
print Y_train_ct.shape