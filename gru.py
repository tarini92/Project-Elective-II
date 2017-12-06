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
def get_features(output_file_path, historyLength, k):
	
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
	X = np.asarray(X)
	Y = np.asarray(Y)

	return X, Y

features_path = 'C:\\Users\\Akanksha\\Desktop\\Sem III\\ML PE\\MOT\\MOT Dataset\\2DMOT2015\\2DMOT2015\\train\\TUD-Campus\\outFile.npz'

# Defining the parameters
historyLength = 7
k = 4
input_feat_dim = 4096
num_classes = 2

X, Y = get_features(features_path, historyLength, k)
print("Input Features Shape: %d, %d, %d" % (X.shape[0], X.shape[1], X.shape[2]))

# Converting the target variable to one hot vector
Y = to_categorical(Y, num_classes)
print("Target Shape: %d, %d" % (Y.shape[0], Y.shape[1]))

net = create_network(historyLength, input_feat_dim, num_classes)
model = tflearn.DNN(net, tensorboard_verbose = 0)
print("Network created")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

model.fit(X_train, y_train , show_metric = True	, validation_set= (X_test, y_test),  snapshot_step = 100, n_epoch = 10)

print(model.evaluate(X_train, y_train))

# Save the model
model.save('GRU1.tfl')
