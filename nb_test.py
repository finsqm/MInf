from old_data_loader import *
import matplotlib.pyplot as plt
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.tree import *
import logging
from sklearn.preprocessing import normalize
import numpy as np

import cPickle
import time

start_time_t = time.time()
start_time_c = time.clock()

logging.basicConfig(filename='experiments1.log',level=logging.DEBUG)
logging.info('Initialising data ...')

data_loader = DataLoader()
data_loader.load('all_chords.csv')
data = data_loader.generate_train_test()

XX_train = data.XX_train
Y_train = data.Y_train

XX_test = data.XX_test
Y_test = data.Y_test

#logging.info('Predicting chord from true chord tones')

#logging.info('Training SVC model ...')
#svc_ch = SVC(decision_function_shape='ovo',probability=True)
#svc_ch.fit(XX_train_ct,Y_train_ct)
#logging.info('Testing SVC model ...')
#svc_ch_result = svc_ch.score(XX_test_ct,Y_test_ct)
#logging.info('SVC model scored {0}'.format(svc_ch_result))

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

logging.info('Training NB model ...')

model = dict()

for state in range(1,25):
	model[state] = np.zeros(12)
for i, song in enumerate(XX_train):
	for j, frame in enumerate(song):
		state = Y_train[i][j]
		model[state] += frame

# Smooth and Normalise
for state in range(1,25):
	model[state] += 1
	model[state] = normalize(model[state][:,np.newaxis], axis=0).ravel()

def get_state(X, m):

	best = 0
	return_state = None

	for state in m:
		score = m[state] * X
		score = np.sum(score)
		if score >= best:
			best = score
			return_state = state

	return return_state


logging.info('Testing NB model ...')

Y_test_predicted = []

for i, song in enumerate(XX_test):
	song_chords = []
	for j, frame in enumerate(song):
		song_chords.append(get_state(frame, model))
	Y_test_predicted.append(song_chords)

correct = 0
count = 0

for i, song in enumerate(Y_test_predicted):
	for j, frame in enumerate(song):
		if frame == Y_test[i][j]:
			correct += 1
		count += 1

nb_result = float(correct) / float(count)


logging.info('NB model scored {0}'.format(nb_result))

end_time_t = time.time()
end_time_c = time.clock()

logging.info("Entire process took {0} seconds".format(end_time_t - start_time_t))
logging.info("Or maybe it was {0} seconds".format(end_time_c - start_time_c))

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

