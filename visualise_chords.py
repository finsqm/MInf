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
chords_np = []
for i, song in enumerate(chord_tones):
	ct_song_np = []
	chords_np += Y_train[i]
	for j, frame in enumerate(song):
		ct_song_np += frame
	chord_tones_np += ct_song_np

chord_tones_np = np.asarray(chord_tones_np)

