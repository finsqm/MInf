import numpy as np
from hmm import *
from data_loader import DataLoader

NUMBER_OF_STATES = 12
DIM = 12

if __name__ == "__main__":

	# Load Data
	dl = DataLoader()
	dl.load('results.csv')
	data = dl.generate_train_test()

	model = data.cross_val()


	









