from hmm import *
from data_loader import *
from utils import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data_loader = DataLoader()
data_loader.load('all_chords.csv')
hmm = HMM()
data = data_loader.generate_train_test()

XX_train = data.XX_train
Y_train = data.Y_train

chord_tones = get_chord_tones(XX_train, Y_train)

ct_np = []
XX_np = []
for i, song in enumerate(chord_tones):
	for j, frame in enumerate(song):
		ct_np += frame
		for note in XX_train[i][j]:
			XX_np.append(note)

XX_np = np.asarray(XX_np)
ct_np = np.asarray(ct_np)

XX_test = data.XX_test
Y_test = data.Y_test

chord_tones_test = get_chord_tones(XX_test, Y_test)

ct_test_np = []
XX_test_np = []
for i, song in enumerate(chord_tones_test):
	for j, frame in enumerate(song):
		ct_test_np += frame
		for note in XX_test[i][j]:
			XX_test_np.append(note)

XX_test_np = np.asarray(XX_test_np)
ct_test_np = np.asarray(ct_test_np)

ct_notes = np.zeros((79502,12))

for i, note in enumerate(XX_local):
	if ct_np[i] == 1:
		ct_notes[i,int(note)] = 1

