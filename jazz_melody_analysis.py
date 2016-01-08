import numpy as np
from hmm import *
from data_loader import DataLoader

NUMBER_OF_STATES = 12
DIM = 12

if __name__ == "__main__":

	# Load Data
	dl = DataLoader()
	dl.load('all_chords.csv')
	data = dl.generate_train_test()

	model = data.cross_val_A()

	count, correct = model.test(data.AA_test,data.Y_test,data.MM_test)

	score = float(correct) / float(count)

	print score


	









