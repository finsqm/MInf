import numpy as np

def get_chord_tones(X, y):
		"""
		X :	4D List
			X[:]			= songs
			X[:][:] 		= frames (varying size)
			X[:][:][:]		= notes
			X[:][:][:][:]	= components

		y :	sequences of chords
			rows	= songs
			columns = chord at each time step

		Return

		chord_tones : 3D
			t[:] 		= songs
			t[:][:] 	= frames
			t[:][:][:] 	= 1 if note is chord tone, 0 otherwise
		"""

		chord_tones = []

		for i, song in enumerate(y):
			ith_song_tones = []
			for j, chord in enumerate(song):
				# Mode = 1 if maj, 0 if min
				if chord % 2 == 0:
					# Minor
					chord_tpc = (chord / 2) - 1
					mode = 0
				else:
					# Major
					chord_tpc = ((chord + 1) / 2) - 1
					mode = 1
				jth_chord_tones = []
				for k, note in enumerate(X[i][j]):
					# TPC (Tonal Pitch Class, [0:11]) of note stored as first component (already normalised by key)
					tpc = note[0]
					# root
					if tpc == chord_tpc:
						jth_chord_tones.append(1)
					# 5th
					elif tpc == (chord_tpc + 7) % 12:
						jth_chord_tones.append(1)
					# 3rd (maj or min)
					elif tpc == (chord_tpc + 3 + mode) % 12:
						jth_chord_tones.append(1)
					else:
						jth_chord_tones.append(0)
				ith_song_tones.append(jth_chord_tones)
			chord_tones.append(ith_song_tones)

		return chord_tones

def get_ct_features(X, y, chord_tones):
		
		X_ct = []
		y_ct = []


		for i, song in enumerate(X):
			for j, frame in enumerate(song):
				x_ij = np.zeros(12)
				for k, note in enumerate(frame):
					if chord_tones[i][j][k] == 1:
						x_ij[int(note[0])] = 1
				X_ct.append(x_ij)
				y_ct.append(y[i][j])

		X_ct = np.asarray(X_ct)
		y_ct = np.asarray(y_ct)

		return X_ct, y_ct

def get_concat_ct_X(X, ct):

	a = [1,2,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

	ct_np = []
	X_np = []
	for i, song in enumerate(ct):
		for j, frame in enumerate(song):
			ct_np += frame
			for note in X[i][j]:
				X_np.append(np.delete(note, a))

	X_np = np.asarray(X_np)
	ct_np = np.asarray(ct_np)

	return X_np, ct_np

