from hmm import *
from data_loader import *
from utils import *
import matplotlib.pyplot as plt
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.tree import *
import logging

import cPickle
import time

start_time_t = time.time()
start_time_c = time.clock()

logging.basicConfig(filename='experiments1.log',level=logging.DEBUG)
logging.info('Initialising data ...')

with open('all_chords_data.pkl','r') as f:
	data = cPickle.load(f)

XX_train = data.XX_train
Y_train = data.Y_train

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

XX_train_ct = np.asarray(XX_train_ct)
Y_train_ct = np.asarray(Y_train_ct)
 
XX_test = data.XX_test
Y_test = data.Y_test

chord_tones_test = get_chord_tones(XX_test, Y_test)

XX_test_ct = []
Y_test_ct = []

for i, song in enumerate(XX_test):
	for j, frame in enumerate(song):
		x_ij = np.zeros(12)
		for k, note in enumerate(frame):
			if chord_tones_test[i][j][k] == 1:
				x_ij[int(note[0])] = 1
		XX_test_ct.append(x_ij)
		Y_test_ct.append(Y_test[i][j])

XX_test_ct = np.asarray(XX_test_ct)
Y_test_ct = np.asarray(Y_test_ct)

logging.info('Predicting chord from true chord tones')

logging.info('Training SVC model ...')
svc_ch = SVC(decision_function_shape='ovo',probability=True)
svc_ch.fit(XX_train_ct,Y_train_ct)
logging.info('Testing SVC model ...')
svc_ch_result = svc_ch.score(XX_test_ct,Y_test_ct)
logging.info('SVC model scored {0}'.format(svc_ch_result))

#logging.info('Training SVC Linear model ...')
#svc_lin = LinearSVC()
#svc_lin.fit(XX_train_ct,Y_train_ct)
#logging.info('Testing SVC Linear model ...')
#svc_lin_result = svc_lin.score(XX_test_ct,Y_test_ct)
#logging.info('SVC Linear model scored {0}'.format(svc_lin_result))

#logging.info('Training Decision Tree model ...')
#dt = DecisionTreeClassifier()
#dt.fit(XX_train_ct,Y_train_ct)
#logging.info('Testing Decision Tree model ...')
#dt_result = dt.score(XX_test_ct,Y_test_ct)
#logging.info('Decision Tree model scored {0}'.format(dt_result))


logging.info('Predicting chord tones from data')

a = [1,2,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

ct_train_np = []
XX_train_np = []
for i, song in enumerate(chord_tones):
	for j, frame in enumerate(song):
		ct_train_np += frame
		for note in XX_train[i][j]:
			XX_train_np.append(np.delete(note, a))

XX_train_np = np.asarray(XX_train_np)
ct_train_np = np.asarray(ct_train_np)

ct_test_np = []
XX_test_np = []
for i, song in enumerate(chord_tones_test):
	for j, frame in enumerate(song):
		ct_test_np += frame
		for note in XX_test[i][j]:
			XX_test_np.append(np.delete(note, a))

XX_test_np = np.asarray(XX_test_np)
ct_test_np = np.asarray(ct_test_np)

logging.info('Training SVC model ...')
svc = SVC(probability=True)
svc.fit(XX_train_np,ct_train_np)
logging.info('Testing SVC model ...')
svc_result = svc.score(XX_test_np,ct_test_np)
logging.info('SVC model scored {0}'.format(svc_result))

#logging.info('Training SVC Linear model ...')
#svc_lin = LinearSVC()
#svc_lin.fit(XX_train_np,ct_train_np)
#logging.info('Testing SVC Linear model ...')
#svc_lin_result = svc_lin.score(XX_test_np,ct_test_np)
#logging.info('SVC Linear model scored {0}'.format(svc_lin_result))

#logging.info('Training Decision Tree model ...')
#dt = DecisionTreeClassifier()
#dt.fit(XX_train_np,ct_train_np)
#logging.info('Testing Decision Tree model ...')
#dt_result = dt.score(XX_test_np,ct_test_np)
#logging.info('Decision Tree model scored {0}'.format(dt_result))


######################################################


logging.info('Predicting chords from data via chord tones')

XX_train_ct_full = []

chord_tones_predicted = svc.predict(XX_train_np)

c = 0

for i, song in enumerate(XX_train):
	for j, frame in enumerate(song):
		x_ij = np.zeros(12)
		for k, note in enumerate(frame):
			if chord_tones_predicted[c] == 1:
				x_ij[int(note[0])] = 1
			c += 1
		XX_train_ct_full.append(x_ij)

XX_train_ct_full = np.asarray(XX_train_ct_full)

######################################################

XX_test_ct_full = []

chord_tones_predicted_test = svc.predict(XX_test_np)

c = 0

for i, song in enumerate(XX_test):
	for j, frame in enumerate(song):
		x_ij = np.zeros(12)
		for k, note in enumerate(frame):
			if chord_tones_predicted_test[c] == 1:
				x_ij[int(note[0])] = 1
			c += 1
		XX_test_ct_full.append(x_ij)

XX_test_ct_full = np.asarray(XX_test_ct_full)

######################################################

logging.info('Testing trained SVC model on predicting chords from data ...')
svc_ch_result_full = svc_ch.score(XX_test_ct_full,Y_test_ct)
logging.info('SVC model scored {0}'.format(svc_ch_result_full))

######################################################

logging.info('Predicting chords from data via chord tones - with model trained on predicted data')

logging.info('Training SVC model ...')
svc_ch_2 = SVC(decision_function_shape='ovo')
svc_ch_2.fit(XX_train_ct_full,Y_train_ct)
logging.info('Testing SVC model ...')
svc_ch_2_result = svc_ch_2.score(XX_test_ct_full,Y_test_ct)
logging.info('SVC model scored {0}'.format(svc_ch_2_result))

end_time_t = time.time()
end_time_c = time.clock()

logging.info("Entire process took {0} seconds".format(end_time_t - start_time_t))
logging.info("Or maybe it was {0} seconds".format(end_time_c - start_time_c))






