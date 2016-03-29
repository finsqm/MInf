import numpy as np
import scipy
import math
from sklearn.preprocessing import normalize
from nltk.probability import (ConditionalFreqDist, ConditionalProbDist, MLEProbDist)
from scipy.stats import multivariate_normal
from numpy.linalg import LinAlgError
from random import randint
from emission import *
from utils import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.svm import *
import copy
from mini_hmm import *


ZER0_VECTOR = [0,0,0,0,0,0,0,0,0,0,0,0]

class HMM(object):
	"""
	Baseline Hidden Markov Model
	"""
	def __init__(self,number_of_states=2,dim=12):
		self.number_of_states = number_of_states
		self.transition_model = TransitionModel(number_of_states)
		self.emission_model = EmissionModel(number_of_states,dim)
		self.trained = False
		self.states = range(1,number_of_states+1)

	def train(self,X,y):
		"""
		Method used to train HMM

		X :	2D Matrix
			X[:]		= songs
			X[:][:]		= notes

		y :	Labels
			y[:]	= song
			y[:]	= Labels
		"""

		self.emission_model.train(X,y)
		self.transition_model.train(y)

		self.trained = True

	def test(self, X, y):

		return self._test_vit(X,y)

	def _test_max_em(self, X, y):

		if not self.trained:
			raise Exception('Model not trained')

		y_pred = []
		for song in X:
			y_pred_i = self.predict_max(song)
			y_pred.append(y_pred_i)

		print y_pred

		# Compare
		count = 0
		correct = 0
		for i, song in enumerate(y):
			for j, frame in enumerate(song):
				count += 1
				if frame % 2 == 0:
					other = -1
				else:
					other = 1
				if (frame == y_pred[i][j]) or (frame + other == y_pred[i][j]):
					correct += 1

		return count, correct

	def predict_max(self,song):

		chord_sequence = []

		for frame in song:
			maximum = -10000000000000
			max_state = 0
			scores = self.emission_model.score_templates(frame)
			for state in self.states:
				lp = scores[state]
				if lp > maximum:
					maximum = lp
					max_state = state
			chord_sequence.append(max_state)

		return chord_sequence

	def _test_vit(self, X, y):
		"""
		Method for testing whether the predictions for XX match Y.

		X :	2D Matrix
			X[:]		= songs
			X[:][:]		= notes

		y :	Labels
			y[:]	= song
			y[:]	= Labels
		"""

		if not self.trained:
			raise Exception('Model not trained')

		y_pred = []
		for song in X:
			y_pred_i = self.viterbi(song)
			y_pred.append(y_pred_i)

		# Compare
		count = 0
		correct = 0
		for i, song in enumerate(y):
			for j, frame in enumerate(song):
				count += 1
				if frame % 2 == 0:
					other = -1
				else:
					other = 1
				if (frame == y_pred[i][j]) or (frame + other == y_pred[i][j]):
					correct += 1

		return count, correct

	def viterbi(self, X):
		"""
		Viterbi forward pass algorithm
		determines most likely state (chord) sequence from observations

		X :	3D Matrix

			X[:] 		= frames (varying size)
			X[:][:]		= notes
			X[:][:][:]	= components

		Returns state (chord) sequence

		Notes:
		State 0 	= starting state
		State N+1	= finish state
		X here is different from X in self.train(X,y), here it is 2D
		"""

		T = len(X)
		N = self.number_of_states

		# Create path prob matrix
		vit = np.zeros((N+2, T))
		# Create backpointers matrix
		backpointers = np.empty((N+2, T))

		# Note: t here is 0 indexed, 1 indexed in Jurafsky et al (2014)

		# Initialisation Step
		for s in range(1,N+1):
			vit[s,0] = self.transition_model.logprob(0,s) + self.emission_model.logprob(s,X[0][:])
			backpointers[s,0] = 0

		# Main Step
		for t in range(1,T):
			for s in range(1,N+1):
				vit[s,t] = self._find_max_vit(s,t,vit,X)
				backpointers[s,t] = self._find_max_back(s,t,vit,X)

		# Termination Step
		vit[N+1,T-1] = self._find_max_vit(N+1,T-1,vit,X,termination=True)
		backpointers[N+1,T-1] = self._find_max_back(N+1,T-1,vit,X,termination=True)

		return self._find_sequence(vit,backpointers,N,T)

	def _find_max_vit(self,s,t,vit,X,termination=False):

		N = self.number_of_states

		if termination:
			v_st_list = [vit[i,t] + self.transition_model.logprob(i,s) \
						for i in range(1,N+1)]
		else:
			v_st_list = [vit[i,t-1] + self.transition_model.logprob(i,s) \
						* self.emission_model.logprob(s,X[t][:]) for i in range(1,N+1)]
		
		return max(v_st_list)

	def _find_max_back(self,s,t,vit,X,termination=False):

		N = self.number_of_states
		
		if termination:
			b_st_list = np.array([vit[i,t] + self.transition_model.logprob(i,s) \
						for i in range(1,N+1)])
		else:
			b_st_list = np.array([vit[i,t-1] + self.transition_model.logprob(i,s) \
						for i in range(1,N+1)])
		
		return np.argmax(b_st_list) + 1

	def _find_sequence(self,vit,backpointers,N,T):
		
		seq = [None for i in range(T)]

		state = backpointers[N+1,T-1]
		seq[-1] = state
		
		for i in range(1,T):
			state = backpointers[state,T-i]
			seq[-(i+1)] = state

		return seq

class TransitionModel(object):
	""" 
	Transition Model 
	n :	Numer of states
	model[i][j] = probability of transitioning to j in i
	"""
	
	def __init__(self, n):
		"""
		Note for transition model states include start and end (0 and n+1)
		"""
		self.number_of_states = n
		self._model = None
		self.states = range(self.number_of_states+2)

	def train(self, y):
		"""
		Supervised training of transition model

		Y :	sequences of chords
			rows	= songs
			columns = chord at each time step

		TODO: augmented sequences with start and end state
		"""

		# Augment data with start and end states for training
		Y = copy.copy(y)
		for i in range(len(y)):
			Y[i].insert(0,0)
			Y[i].append(self.number_of_states + 1)

		self._model = self._get_normalised_bigram_counts(Y)

	def _get_normalised_bigram_counts(self,y):

		model = dict()

		for state in self.states:
			model[state] = np.zeros(self.number_of_states + 2)

		for sequence in y:
			lasts = None
			for state in sequence:
				if lasts is not None:
					model[lasts][state] += 1
				lasts = state

		# Smooth and Normalise
		for state in self.states:
			model[state] += 1
			model[state] = normalize(model[state][:,np.newaxis], axis=0).ravel()

		return model

	def doesnt_work(self,y):
		"""
		Code adapted from NLTK implementation of supervised training in HMMs
		"""
		
		estimator = lambda fdist, bins: MLEProbDist(fdist)

		transitions = ConditionalFreqDist()
		outputs = ConditionalFreqDist()
		for sequence in y:
			lasts = None
			for state in sequence:
				if lasts is not None:
					transitions[lasts][state] += 1
				lasts = state

		N = self.number_of_states + 2
		model = ConditionalProbDist(transitions, estimator, N)

		return model

	def logprob(self, state, next_state):
		prob = self._model[state][next_state]
		return math.log(prob,2)

class EmissionModel(object):
	"""
	Gaussian Emission Model
	Different Gaussian parameters for each state
	"""
	def __init__(self,number_of_states,dim=12):
		self.number_of_states = number_of_states 	# can change
		self.dim = dim 				# should be 12
		self.states = range(1,number_of_states+1)
		self._model = None

	def train(self,X,y):
		"""
		Supervised training of emission model

		X :	2D Matrix
			X[:]		= songs
			X[:][:]		= notes

		y :	sequences of chords
			rows	= songs
			columns = chord at each time step
		"""

		self._train_chord_tones_hmm(X,y)

	def _train_chord_tones_hmm(self, X, y):

		self.chord_hmms = dict()
		for state in self.states:
			chord_tones_states = get_chord_tones_states_majmin(X, y)
			hmm_i = MINIHMM(state)
			hmm_i.train(X,y,chord_tones_states)
			self.chord_hmms[state] = hmm_i

	def _train_chord_tones_dt(self, X, y):

		chord_tones = get_chord_tones(X, y)

		######################################################

		#logger.info('Predicting chord tones from data')

		X_np, ct_np = get_concat_ct_X(X, chord_tones)

		#logger.info('Training Decision Tree model ...')
		self.dt_part1 = DecisionTreeClassifier()
		print X_np.shape
		print ct_np.shape
		self.dt_part1.fit(X_np,ct_np)

		######################################################

		X_ct , y_ct = get_ct_features(X, y, chord_tones)

		#logger.info('Predicting chords from true chord tones')

		#logger.info('Training Decision Tree model ...')
		self.dt_part2 = DecisionTreeClassifier()
		self.dt_part2.fit(X_ct, y_ct)

	def _train_chord_tones_svm(self, X, y):

		chord_tones = get_chord_tones(X, y)

		######################################################

		#logger.info('Predicting chord tones from data')

		X_np, ct_np = get_concat_ct_X(X, chord_tones)

		#logger.info('Training Decision Tree model ...')
		self.dt_part1 = SVC(decision_function_shape='ovo')
		print X_np.shape
		print ct_np.shape
		self.dt_part1.fit(X_np,ct_np)

		######################################################

		X_ct , y_ct = get_ct_features(X, y, chord_tones)

		#logger.info('Predicting chords from true chord tones')

		#logger.info('Training Decision Tree model ...')
		self.dt_part2 = SVC(decision_function_shape='ovo',probability=True)
		self.dt_part2.fit(X_ct, y_ct)

	def logprob(self, state, obv):

		return self.logprob_chord_tones_hmm(state, obv)

	def logprob_chord_tones_hmm(self,state,obv):

		return self.chord_hmms[state].get_viterbi_logprob(obv)

	def logprob_templates(self, state, obv):
		"""
		Tenplate matching algorithm, weighted on duratum
		"""

		score = dict()
		score_total = 0

		for chord in self.states:
			if chord % 2 == 0:
				# Minor
				chord_tpc = (chord / 2) - 1
				mode = 0
			else:
				# Major
				chord_tpc = ((chord + 1) / 2) - 1
				mode = 1
			template = (chord_tpc,(chord_tpc + 3 + mode) % 12,(chord_tpc + 7) % 12)
			score[chord] = 0.5
			#missing = [1,1,1]
			dur_total = 1
			for note, dur in obv:
				dur_total += dur
			dur_total = float(dur_total)
			for note, dur in obv:
				if note in template:
					score[chord] += (dur / (2 * dur_total))
					idx = template.index(note)
					#missing[idx] = 0
				else:
					score[chord] -= (dur / (2 * dur_total))
			#score[chord] -= sum(missing)
			score_total += score[chord]

		for chord in self.states:
			if score[chord] == 0:
				score[chord] = 0.000000001
				score_total += 0.000000001
			score[chord] = float(score[chord]) / float(score_total)

		try:
			lp = math.log(score[state],2)
		except ValueError:
			lp = -500

		return lp

	def score_templates(self, obv):

		score = dict()

		for chord in self.states:
			if chord % 2 == 0:
				# Minor
				chord_tpc = (chord / 2) - 1
				mode = 0
			else:
				# Major
				chord_tpc = ((chord + 1) / 2) - 1
				mode = 1
			template = (chord_tpc,(chord_tpc + 3 + mode) % 12,(chord_tpc + 7) % 12)
			missing = [1,1,1]
			score[chord] = 0
			for note, dur in obv:
				if note in template:
					score[chord] += (dur + 1)
					idx = template.index(note)
					missing[idx] = 0
				else:
					score[chord] -= (dur + 1)
			score[chord] -= sum(missing)

		return score

	def logprob_freq(self, state, obv):

		prob = self._model[state][obv]
		return math.log(prob,2)

	def logprob_dt(self, state, obv):

		a = [1,2,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

		X = []
		for note in obv:
			X.append(np.delete(note, a))

		X = np.asarray(X)
		
		predicted_ct = self.dt_part1.predict(X)

		x = np.zeros(12)

		for i, note in enumerate(X):
			if predicted_ct[i] == 1:
				x[int(note[0])] = 1

		x = np.asarray(x)

		[logprobs] = self.dt_part2.predict_log_proba([x])

		return logprobs[state - 1]

	def _get_nb_estimates(self, X, y):

		model = dict()

		for state in self.states:
			model[state] = np.zeros(self.dim)
		for i, song in enumerate(X):
			for j, frame in enumerate(song):
				state = y[i][j]
				model[state] += frame

		# Smooth and Normalise
		for state in self.states:
			model[state] += 1
			model[state] = normalize(model[state][:,np.newaxis], axis=0).ravel()

		return model


	def _get_mle_estimates(self, X, y):

		model = dict()
		
		# align data to states
		lists = dict()
		for state in self.states:
			lists[state] = []
		for i, song in enumerate(y):
			for j, state in enumerate(song):
				lists[state].append(X[i][j][:])

		# create numpy version cos numpy's great
		xs = dict()
		for state in self.states:
			xs[state] = np.asarray(lists[state])
		del lists

		# Calculate means and covs
		for state in self.states:
			if len(xs[state]) > 0:
				mean = np.mean(xs[state],axis=0)
				cov = np.cov(xs[state].T)
				try:
					model[state] = multivariate_normal(mean,cov)
				except LinAlgError:
					model[state] = multivariate_normal(mean,cov=1.0)
			else:
				model[state] = multivariate_normal(mean=ZER0_VECTOR,cov=1.0)

		return model

	def _train_templates(self, X, y):
		"""
		Template matching method
		"""

		pass

	def _train_smooth_freq(self, X, y):
		"""
		Smoothed frequency count model
		"""

		model = dict()

		# Initialise and smooth
		for chord in self.states:
			model[chord] = dict()
			# Smoothed

			model[chord]['total'] = 1
			for note in range(self.dim):
				# Smoothed
				model[chord][note] = 1

		# Count frequencies
		for i, song in enumerate(X):
			for j, note in enumerate(song):
				chord = y[i][j]
				model[chord][note] += 1
				model[chord]['total'] += 1

		# Normalise
		for chord in self.states:
			N = model[chord]['total']
			for note in range(self.dim):
				model[chord][note] = float(model[chord][note]) / N

		self._model = model




