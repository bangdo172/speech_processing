import librosa
import numpy as np
import os
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm

NUM_CLASS = 3

init_data = {}
init_data['start']
init_data['trans']
init_data['start']['nguoi'] = [1, 0, 0]
init_data['start']['khong'] = [1, 0, 0]
init_data['start']['nhieu'] = [1, 0, 0, 0]
init_data['start']['tien'] = [1, 0, 0, 0]
init_data['start']['thoigian'] = [1, 0, 0, 0, 0, 0]

init_data['trans']['nguoi'] = [[0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0],
[0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0],
[0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0],
[0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0],
[0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0],
[0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0],
[0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1],
[0, 0, 0, 0, 0, 0, 0, 0.5, 0.5],
[0, 0, 0, 0, 0, 0, 0, 0, 1]]
init_data['trans']['khong'] = init_data['trans']['nguoi']
init_data['trans']['nhieu'] = [[0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
init_data['trans']['tien'] = init_data['nhieu']
init_data['trans']['thoigian'] = [[0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.4, 0.1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

def get_mfcc(file_path):
	y, sr = librosa.load(file_path) # read .wav file
	hop_length = math.floor(sr*0.010) # 10ms hop
	win_length = math.floor(sr*0.025) # 25ms frame
	# mfcc is 12 x T matrix
	mfcc = librosa.feature.mfcc(
		y, sr, n_mfcc=12, n_fft=1024,
		hop_length=hop_length, win_length=win_length)
	# substract mean from mfcc --> normalize mfcc
	mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
	# delta feature 1st order and 2nd order
	delta1 = librosa.feature.delta(mfcc, order=1)
	delta2 = librosa.feature.delta(mfcc, order=2)
	# X is 36 x T
	X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
	# return T x 36 (transpose of X)
	return X.T # hmmlearn use T x N matrix

def get_class_data(data_dir):
	files = os.listdir(data_dir)
	mfcc = [get_mfcc(os.path.join(data_dir,f)) for f in files if f.endswith(".wav")]
	return mfcc

def clustering(X, n_clusters=10):
	kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
	kmeans.fit(X)
	print("centers", kmeans.cluster_centers_.shape)
	return kmeans  

def get_init(class_name):
	file = open('init/{}.txt'.format(class_name), 'r')
	file.readline()
	num_phonemic = int(file.readline())
	file.readline()
	file.readline()
	start_prob = file.readline().split('\t')
	start_prob_ = []
	for s in start_prob:
		start_prob_.append(float(s.replace('\n', '')))
	print(class_name)
	print(start_prob_)
	file.readline()
	file.readline()
	transmat = []
	for i in range(num_phonemic * 3):
		print(i)
		line = file.readline().split('\t')
		line_ = []
		for l in line:
			ll = l.replace('\n', '')
			print(ll)
			line_.append(float(ll))
		print(line_)
		transmat.append(line_)
	return start_prob, transmat

if __name__ == "__main__":
	class_names = ["nguoi", "khong", "nhieu", "tien", "thoigian", "test_nguoi", "test_khong", "test_nhieu", "test_tien", "test_thoigian"]
	dataset = {}
	for cname in class_names:
		print(f"Load {cname} dataset")
		dataset[cname] = get_class_data(os.path.join("data", cname))

	# Get all vectors in the datasets
	all_vectors = np.concatenate([np.concatenate(v, axis=0) for k, v in dataset.items()], axis=0)
	print("vectors", all_vectors.shape)
	# Run K-Means algorithm to get clusters
	kmeans = clustering(all_vectors)
	print("centers", kmeans.cluster_centers_.shape)

	models = {}
	for cname in class_names:
		class_vectors = dataset[cname]
		# convert all vectors to the cluster index
		# dataset['one'] = [O^1, ... O^R]
		# O^r = (c1, c2, ... ct, ... cT)
		# O^r size T x 1
		dataset[cname] = list([kmeans.predict(v).reshape(-1,1) for v in dataset[cname]])
		
		transmat = np.array()

		if cname[:4] != 'test':
			hmm = hmmlearn.hmm.MultinomialHMM(
				n_components=6, random_state=0, n_iter=1000, verbose=True,
				startprob_prior=np.array([0.7,0.2,0.1,0.0,0.0,0.0]),
				transmat_prior=np.array([
					[0.1,0.5,0.1,0.1,0.1,0.1,],
					[0.1,0.1,0.5,0.1,0.1,0.1,],
					[0.1,0.1,0.1,0.5,0.1,0.1,],
					[0.1,0.1,0.1,0.1,0.5,0.1,],
					[0.1,0.1,0.1,0.1,0.1,0.5,],
					[0.1,0.1,0.1,0.1,0.1,0.5,],
				]),
			)

			X = np.concatenate(dataset[cname])
			lengths = list([len(x) for x in dataset[cname]])
			print("training class", cname)
			print(X.shape, lengths, len(lengths))
			hmm.fit(X, lengths=lengths)
			models[cname] = hmm
	print("Training done")

	print("Testing")
	for true_cname in class_names:
		for O in dataset[true_cname]:
			score = {cname : model.score(O, [len(O)]) for cname, model in models.items() if cname[:4] != 'test' }
			print(true_cname, score)