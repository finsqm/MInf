import numpy as np
from hmm import *
from data_loader import DataLoader

NUMBER_OF_STATES = 36
DIM = 12

def crossval(hmm,XX_train,Y_train):
	# TODO
	hmm.train(hmm,XX_train,Y_train)

def test(hmm,XX_test,Y_test):
	# TODO 
	pass

def run(hmm,data):
	
	XX_train = data.XX_train
	Y_train = data.Y_train
	XX_test = data.XX_test
	Y_test = data.Y_test

	crossval(hmm,XX_train,Y_train)

	acc = test(hmm,XX_test,Y_test)


if __name__ == "__main__":

	# Load Data
	dl = DataLoader()
	dl.load('results.csv')
	data = dl.generate_train_test()

	hmm = HMM(NUMBER_OF_STATES,DIM)

	run(hmm,data)









