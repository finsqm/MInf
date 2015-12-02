import numpy as np
import scipy
import sklearn
from nltk.probability import (ConditionalFreqDist, ConditionalProbDist, MLEProbDist)
from scipy.stats import multivariate_normal

class HMM(object):
	"""
	Hidden Markov Model
	"""
	def __init__(self,number_of_states=60,dim=12):
		self.number_of_states = number_of_states
		self.transition_model = TransitionModel(number_of_states)
		self.emission_model = EmissionModel(number_of_states,dim)
		self.trained = False

	def train(self,X,y):
		"""
		Method used to train HMM

		X :	3D Matrix
			X[:]		= songs
			X[:][:] 	= frames (varying size)
			X[:][:][:]	= components

		y :	Labels
			y[:]	= song
			y[:]	= Labels
		"""

		self.emission_model.train(X,y)
		self.transition_model.train(y)

		self.trained = True

	def viterbi(self, X):
		"""
		Viterbi forward pass algorithm
		determines most likely state (chord) sequence from observations

		X : Observation Matrix (song)
			rows 	= each observation at time t 
			columns	= component of observation (prob dist over 12 notes)

		Returns state (chord) sequence

		Notes:
		State 0 	= starting state
		State N+1	= finish state
		X here is different from X in self.train(X,y), here it is 2D
		"""

		if not self.trained:
			raise Exception('Model not trained')

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

		if termination:
			v_st_list = [vit[i,t] + self.transition_model.logprob(i,s) \
						for i in range(1,N+1)]
		else:
			v_st_list = [vit[i,t-1] + self.transition_model.logprob(i,s) \
						* self.emission_model.logprob(s,X[t][:]) for i in range(1,N+1)]
		
		return max(v_st_list)

	def _find_max_back(self,s,t,vit,X,termination=False):
		
		if termination:
			b_st_list = np.array([vit[i,t] + self.transition_model.logprob(i,s) \
						for i in range(1,N+1)])
		else:
			b_st_list = np.array([vit[i,t-1] + self.transition_model.logprob(i,s) \
						for i in range(1,N+1)])
		
		return np.argmax(b_st_list) + 1

	def _find_sequence(vit,backpointers,N,T):
		
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
		self.number_of_states = n
		

	def train(self, y):
		"""
		Supervised training of transition model

		Y :	sequences of chords
			rows	= songs
			columns = chord at each time step

		TODO: augmented sequences with start and end state
		"""

		# Augment data with start and end states for training
		Y = y[:]
		for i in range(len(y)):
			Y[i].insert(0,0)
			Y[i].append(self.number_of_states + 1)

		self._model = self._get_normalised_bigram_counts(Y)

	def _get_normalised_bigram_counts(self,y):
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

	def logprob(self, i, j):

		return self._model[i].logprob(i)

class EmissionModel(object):
	"""
	Gaussian Emission Model
	Different Gaussian parameters for each state
	"""
	def __init__(self,number_of_states,dim):
		self.number_of_states = number_of_states 	# can change
		self.dim = dim 				# should be 12
		self.states = range(1,number_of_states+1)

	def train(self,X,y):
		"""
		Supervised training of emission model

		X :	3D list
			X[:]		= song
			X[:][:] 	= frame
			X[:][:][:]	= component

		y :	sequences of chords
			rows	= songs
			columns = chord at each time step
		"""

		self._model = self._get_mle_estimates(X,y)

	def logprob(self, state, obv):
		return self._model[state].logpdf(obv)

	def _get_mle_estimates(self, X, y):

		model = dict()
		
		# ALIGN DATA
		lists = dict()
		for state in self.states:
			lists[state] = []
		for i, song in enumerate(y):
			for j, state in enumerate(song):
				lists[state].append(X[i][j][:])

		# make create numpy version cos numpy's great
		xs = dict()
		for state in self.states:
			xs[state] = np.asarray(lists[state])
		del lists

		# Calculate means and covs
		for state in self.states:
			mean = np.mean(xs[state],axis=0)
			cov = np.cov(xs[state].T)
			model[state] = multivariate_normal(mean,cov)

		return model






		
		
		
		