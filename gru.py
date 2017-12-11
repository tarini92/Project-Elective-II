import pickle
import tflearn
import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical

# Code for first GRU
def create_network(historyLength, input_feat_dim, num_classes):

	### Create a one layer GRU
	net = tflearn.input_data(shape=[None, historyLength , input_feat_dim])
	print("Input Layer created")
	net = tflearn.gru(net, 256, dropout = 0.2)
	print("gru layer created")
	net = tflearn.fully_connected(net, num_classes, activation = 'softmax')
	print("FC created")
	net = tflearn.regression(net, optimizer = 'adam', loss = 'mean_square', name = 'output1')
	print("regression Layer created")
	return net

# Retrieve stored feature sequences per video as a list of sequence arrays
def get_features_per_video(output_file_path, historyLength, k):
	
	with open(output_file_path, 'rb') as f:
		feat = pickle.load(f)


	X = []
	Y = []
	for seq in feat:

		# extracting data point corresponding to 1 label
		nrow = seq.shape[0]
		datapoint = seq[nrow - historyLength : nrow, 0:4096]
		#datapoint = datapoint.flatten().reshape(1,(datapoint.shape[0]*datapoint.shape[1]))
		X.append(datapoint)
		Y.append(1)


		# extracting data point corresponding to 0 label
		for _ in range(k-1):
			random_index = random.randrange(0, nrow - historyLength)
			datapoint = seq[random_index:random_index + historyLength, 0:4096]
			#datapoint = datapoint.flatten().reshape(1,(datapoint.shape[0]*datapoint.shape[1]))
			X.append(datapoint)
			Y.append(0)

	#X = np.vstack(X)
	

	return X, Y
# Returns features and labels from entire training list of videos
def consolidated_features_and_labels():
	
	hist_length=7
	k=4
	X=[]
	Y=[]
	features=[]
	labels=[]

	train_videos = ['TUD-Campus','TUD-Stadtmitte','ETH-Sunnyday','KITTI-13','Venice-2']
	outFile = '/Users/tarinichandra/Desktop/PE/MOT Dataset/2DMOT2015/train/'
	for name in train_videos:
		vid_path = os.path.join(outFile,name)
		vid_path = str(vid_path)+"/"
		feat_path = str(vid_path)+"outFile.npz"
		features, labels = features_and_labels_per_video(feat_path,hist_length,k)
		X.append(features)
		Y.append(labels)
	X = np.asarray(X)
	Y = np.asarray(Y)	

	return X,Y	


# Defining the parameters
X,Y = consolidated_features_labels()


X = np.array(X)
Y = np.array(Y)

hist_length = 7

#Converting the target to one-hot vector
Y = to_categorical(Y, num_classes)

#Creating a Tensorflow model
net = create_model(hist_length, input_size, num_classes)
model = tflearn.DNN(net, tensorboard_verbose = 0)

#Cross validation
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

model.fit(X_train, y_train ,show_metric = True, snapshot_step = 100, n_epoch = 10)


# Evaluate. Note that although we're passing in "train" data,
# this is actually our holdout dataset, so we never actually
# used it to train the model. Bad variable semantics.
print(model.evaluate(X_test, y_test))


# Save the model
model.save('GRU1.tfl')
