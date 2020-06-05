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
import tkinter as tk
from tkinter import messagebox
import pygame
from pydub import AudioSegment, silence
import ffmpeg
import pyaudio
import wave
from base64 import b64decode

class_names = ['khong', 'nhieu', 'thoigian', 'nguoi', 'tien']
n_states = [9, 12, 19, 12, 12]
# build data
path = 'datax/'
# path = '/Users/bangdo/code/school/speech_processing/speech_processing/test/trimmed'
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

def read_data(path_ = path):
	X, y = {}, {}
	for idx, cln in enumerate(class_names):
		files = [os.path.join(path_, cln, f) for f in os.listdir(os.path.join(path_, cln))]
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


def test_one_file(file_):
	record_mfcc = get_mfcc(file_)
	model = load_model()
	print(model)
	scores = [model[cname].score(record_mfcc) for cname in class_names]
	pred = np.argmax(scores)
	print(class_names[pred])


def detect_leading_silence(sound, silence_threshold=-30.0, chunk_size=10):
	'''
	sound is a pydub.AudioSegment
	silence_threshold in dB
	chunk_size in ms

	iterate over chunks until you find the first one with sound
	'''
	trim_ms = 0 # ms

	assert chunk_size > 0 # to avoid infinite loop
	while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
		trim_ms += chunk_size

	return trim_ms

def record():
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 22050
	RECORD_SECONDS = 2
	WAVE_OUTPUT_FILENAME = "record_data/record.wav"

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
					channels=CHANNELS,
					rate=RATE,
					input=True,
					frames_per_buffer=CHUNK)

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

def play():    
	filename = 'record_data/record.wav'
	pygame.init()
	pygame.mixer.init()
	sounda = pygame.mixer.Sound(filename)
	sounda.play()
	#winsound.PlaySound(filename, winsound.SND_FILENAME)

def trim(ori_path = 'record_data/record.wav', fpath = 'record_data/trimmed.wav'):
    myaudio = AudioSegment.from_wav(ori_path)
    audios = silence.split_on_silence(myaudio, min_silence_len=600, silence_thresh=-40, keep_silence=100)            
    for audio in audios:
        audio.export(fpath, format = "wav")            
        break
	
def playtrimmed():    
	trim()
	filename = 'record_data/trimmed.wav'
	pygame.init()
	pygame.mixer.init()
	sounda = pygame.mixer.Sound(filename)
	sounda.play()
#     winsound.PlaySound(filename, winsound.SND_FILENAME)

# def trim(record_path='record_data/', name = 'trimmed.wav'):
# 	input_path = record_path + 'record.wav'
# 	output_path = record_path + name
# 	sound = AudioSegment.from_file(input_path, format="wav")
# 	start_trim = detect_leading_silence(sound)
# 	end_trim = detect_leading_silence(sound.reverse())
# 	duration = len(sound)
# 	print(start_trim, end_trim)
# 	trimmed_sound = sound[start_trim:duration-end_trim]    
# 	trimmed_sound.export(output_path, format="wav")

def predict_new():
	trim()
	#Predict
	record_mfcc = get_mfcc("record_data/trimmed.wav")
	model = load_model()
	scores = [model[cname].score(record_mfcc) for cname in class_names]
	pred = np.argmax(scores)
	messagebox.showinfo("result", class_names[pred])

def retrain(label_):
	model = load_model()
	X_train = {}
	y_train = {}
	mfcc = []
	label = []
	mfcc.append(get_mfcc('record_data/trimmed.wav'))
	label.append(label_)
	count = 0
	for idx, cln in enumerate(class_names):
		if cln == label_:
			X_train.update( {cln: mfcc} )
			y_train.update( {cln: [idx]} )
			model[cln].fit(X=np.vstack(X_train[cln]), lengths=[x.shape[0] for x in X_train[cln]])
			save_model(model[cln], cln)
	messagebox.showinfo('Notification', 'Retrain with new data!')
	return model


def gui():
	window = tk.Tk()
	window.geometry("600x500")
	window.title("Speech recognition")

	frame0 = tk.Frame(master=window)
	frame0.pack()

	frame4 = tk.Frame(master=window)
	frame4.pack()

	frame1 = tk.Frame(master=window)
	frame1.pack()

	frame2 = tk.Frame(master=window)
	frame2.pack()

	frame3 = tk.Frame(master=window)
	frame3.pack()





	label = tk.Label(master=frame0, text="Speech recognition")
	label.pack(padx=5, pady=10)

	btn_playback = tk.Button(master=frame4, width=13, height=2, text="Train", command = gui_train)
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)

	btn_playback = tk.Button(master=frame4, width=13, height=2, text="Validate", command = gui_validate)
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)

	btn_record = tk.Button(master=frame1, width=13, height=2, text="Record", command=record)
	btn_record.pack(side=tk.LEFT, padx=5, pady=5)

	btn_playback = tk.Button(master=frame2, width=13, height=2, text="Playback", command=play)
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)

	btn_playback = tk.Button(master=frame2, width=13, height=2, text="Play trim", command=playtrimmed)
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)

	btn_predict = tk.Button(master=frame3, width=13, height=2, text="Predict", command=predict_new)
	btn_predict.pack(side=tk.LEFT, padx=5, pady=5)



	lb = tk.Frame(master = window)
	lb.pack()

	btn_playback = tk.Button(master=lb, width=5, height=2, text="Người", command= lambda: retrain('nguoi'))
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)
	btn_playback = tk.Button(master=lb, width=5, height=2, text="Không", command= lambda: retrain('khong'))
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)
	btn_playback = tk.Button(master=lb, width=5, height=2, text="Nhiều", command= lambda: retrain('nhieu'))
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)
	btn_playback = tk.Button(master=lb, width=5, height=2, text="Tiền", command= lambda: retrain('tien'))
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)
	btn_playback = tk.Button(master=lb, width=5, height=2, text="Thời gian", command= lambda: retrain('thoigian'))
	btn_playback.pack(side=tk.LEFT, padx=5, pady=5)

	window.mainloop()

def gui_train():
	X, y = read_data()
	X_train, y_train, X_test, y_test = split_data(X, y)
	model = train(X_train)
	messagebox.showinfo('Notification', 'Trained!')

def gui_validate():
	X, y = read_data()
	X_train, y_train, X_test, y_test = split_data(X, y)
	model = load_model()
	val = validate(X_test, y_test, model)
	print(val)
	messagebox.showinfo('Notification', val)



def trimxxx(pathx):

	for cln in class_names:
		files = [f for f in os.listdir(os.path.join(pathx, cln))]
		print(files)
		for f in files:
			input_path = pathx + cln + '/' + f
			output_path = pathx + 'trimmed/' + cln + '/' + f
			sound = AudioSegment.from_file(input_path, format="wav")
			start_trim = detect_leading_silence(sound)
			end_trim = detect_leading_silence(sound.reverse())
			duration = len(sound)
			trimmed_sound = sound[start_trim:duration-end_trim]    
			trimmed_sound.export(output_path, format="wav")
def main():
	X, y = read_data()
	X_train, y_train, X_test, y_test = split_data(X, y)
	model = train(X_train)
	model = load_model()
	val = validate(X_test, y_test, model)
	print(val)
	gui()


if __name__ == '__main__':
	# pathx = '/Users/bangdo/code/school/speech_processing/speech_processing/datax/'
	# trimxxx(pathx)
	main()


