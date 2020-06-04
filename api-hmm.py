import os, numpy as np
import pickle
import hmmlearn.hmm as hmm
from sklearn.cluster import KMeans
import librosa
from sklearn.model_selection import train_test_split
# np.random.seed(13)
import math
# import pickle, os
import numpy as np
from sklearn.metrics import classification_report

class_names = ['khong', 'nhieu', 'thoigian', 'nguoi', 'tien']
n_states = [9, 12, 19, 12, 12]
# build data
path = 'data/'

def get_mfcc(filename):
	y, sr = librosa.load(filename) # read .wav file
	hop_length = math.floor(sr*0.010) # 10ms hop
	win_length = math.floor(sr*0.025) # 25ms frame
	# mfcc is 13 x T matrix
	mfcc = librosa.feature.mfcc(
		y, sr, n_mfcc=13, n_fft=1024,
		hop_length=hop_length, win_length=win_length)
	# substract mean from mfcc --> normalize mfcc
	mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
	# delta feature 1st order and 2nd order
	delta1 = librosa.feature.delta(mfcc, order=1)
	delta2 = librosa.feature.delta(mfcc, order=2)
	# X is 39 x T
	X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
	# return T x 39 (transpose of X)
	return X.T # hmmlearn use T x N matrix

def read_data():
	X, y = {}, {}
	for idx, cln in enumerate(class_names):
		files = [os.path.join(path, cln, f) for f in os.listdir(os.path.join(path, cln))]
		mfcc = [get_mfcc(file) for file in files]
		label = [idx for i in range(len(mfcc))]
		X.update( {cln: mfcc} )
		y.update( {cln: label} )
	return X, y

def split_data(X, y):
	X_train, y_train = {}, {}
	X_test, y_test = {}, {}
	for key in class_names:
		data_train, data_test, label_train, label_test = train_test_split(X[key], y[key], test_size = 0.2, random_state=13, shuffle=None)
		X_train.update( {key: data_train} )
		y_train.update( {key: label_train} )
		X_test.update( {key: data_test} )
		y_test.update( {key: label_test} )
	return X_train, y_train, X_test, y_test

def train(X_train):
	model = {}
	for idx, key in enumerate(class_names):
		print('________', key)
		start_prob = np.full(n_states[idx], 0.0)
		start_prob[0] = 1.0
		trans_matrix = np.full((n_states[idx], n_states[idx]), 0.0)
		np.fill_diagonal(trans_matrix, 0.5)
		np.fill_diagonal(trans_matrix[0:, 1:], 0.5)
		np.fill_diagonal(trans_matrix[0:, 2:], 0.5)
		trans_matrix[-1,-1] = 1.0
		
		model[key] = hmm.GaussianHMM(
			n_components=n_states[idx], 
			verbose=True, 
			n_iter=300, 
			startprob_prior=start_prob, 
			transmat_prior=trans_matrix,
			params='mc',
			init_params='stmc',
			random_state=0
		)

		model[key].fit(X=np.vstack(X_train[key]), lengths=[x.shape[0] for x in X_train[key]])
		save_model(model[key], key)
	return model

def predict(mfcc, model):
	scores = []
	for key in class_names:
		# print(key)
		# print(model[key])
		logit = model[key].score(mfcc)
		scores.append(logit)

	pred = np.argmax(scores)
	return pred

def validate(X_test, y_test, model):

	y_true = []
	y_pred = []
	for key in class_names:
		for data, target in zip(X_test[key], y_test[key]):
			pred = predict(data, model)
			y_pred.append(pred)
			y_true.append(target)
	# print(y_true)
	# print(y_pred)
	report = classification_report(y_true, y_pred, target_names=class_names)
	return report

def save_model(model, name):
	model_path = 'models/' + name
	pickle.dump(model, open(model_path, 'wb'))

def load_model():
	model = {}
	model_path = 'models/'
	for x in class_names:
		model[x] = pickle.load(open(model_path + x, 'rb'))
	return model

def main():
	X, y = read_data()
	X_train, y_train, X_test, y_test = split_data(X, y)
	model = train(X_train)
	model = load_model()
	val = validate(X_test, y_test, model)
	print(val)

def test_one_file(file_):
	record_mfcc = get_mfcc(file_)
	model = load_model()
	print(model)
	scores = [model[cname].score(record_mfcc) for cname in class_names]
	pred = np.argmax(scores)
	print(class_names[pred])

if __name__ == '__main__':
	file = 'test/nguoi/nguoi_005.wav'
	# test_one_file(file)
	main()


