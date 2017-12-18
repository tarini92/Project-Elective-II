import pickle
import tflearn
import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
import matplotlib.pyplot as plt

# defining the recurrent network with 1 input layer(with 3d tensor), 2 layers of gru and one final estimator layer to perform classification
def get_network_wide(historyLength, input_feat_dim, num_classes):

	net = tflearn.input_data(shape=[None, historyLength , input_feat_dim])
	print("Input Layer created")
	net = tflearn.gru(net, 128, return_seq = True)
	print("Recurrent layer 1 created")
	net = tflearn.gru(net, 64)
	print("Recurrent layer 2 created")
	net = tflearn.fully_connected(net, num_classes, activation = 'softmax', regularizer = 'L2')
	print("FC layer 1 created")
	net = tflearn.regression(net, optimizer = 'adam', loss = 'mean_square', name = 'output1', learning_rate = 0.001)
	print("Regression Layer created")
	return net

# Retrieve stored feature sequences per video as a list of sequence of arrays
def get_features(feat_seqs_list, historyLength, k, input_feat_dim, filter):

	X = []
	Y = []
	for seq in feat_seqs_list:
		#print(seq.shape)
		# extracting data point corresponding to 1 label
		nrow = seq.shape[0]
		if nrow >= filter:
			datapoint = seq[nrow - historyLength : nrow, 0:input_feat_dim]
			X.append(datapoint)
			Y.append(1)


			# extracting data point corresponding to 0 label
			for _ in range(k-1):
				random_index = random.randrange(0, nrow - historyLength)
				datapoint = seq[random_index:random_index + historyLength, 0:input_feat_dim]
				X.append(datapoint)
				Y.append(0)
		
	#X = np.vstack(X)
	X = np.asarray(X)
	Y = np.asarray(Y)

	return X, Y

# Takes the path for all videos in trining data and extracts the sequences from the videos and append them into a single list
# Returns the lsit of all iid(independent and identically distributed) sequences in the training set of videos
def read_features(train_folder_path, train_name_list, feat_out_file):
	feature_sequences_list = []
	for name in train_name_list:
		path = os.path.join(train_folder_path, name)
		path = os.path.join(path, feat_out_file)
		with open(path, 'rb') as f:
			feat = pickle.load(f)
		for seq in feat:
			feature_sequences_list.append(seq)

	return feature_sequences_list


if __name__ == '__main__':
	# Declaring the paths
	train_folder_path = 'C:\\Users\\Akanksha\\Desktop\\Sem III\\ML PE\\MOT\\MOT Dataset\\2DMOT2015\\2DMOT2015\\train'
	train_name_list = ['ADL-Rundle-6', 'ADL-Rundle-8','ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2']
	feat_out_file = 'outFile.npz'


	# Defining the variables
	historyLength = 10      ### defines the number of previous number of frames to be considered for history
	k = 3					### defines the ratio of frames with 0 labels to frames with 1 labels (k:1)
	input_feat_dim = 4104   ### dimensions of the features
	num_classes = 2
	filter = 15

	feat_seqs_list = read_features(train_folder_path, train_name_list, feat_out_file)

	X, Y = get_features(feat_seqs_list, historyLength, k, input_feat_dim, filter)
	print("Input Features Shape: %d, %d, %d" % (X.shape[0], X.shape[1], X.shape[2]))

	# Converting the target variable to one hot vector
	Y = to_categorical(Y, num_classes)
	print("Target Shape: %d, %d" % (Y.shape[0], Y.shape[1]))

	net = get_network_wide(historyLength, input_feat_dim, num_classes)
	model = tflearn.DNN(net, tensorboard_verbose = 0)
	print("Network created")
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

	model.fit(X_train, y_train , show_metric = True	, validation_set= 0.1, n_epoch = 5)

	# Evaluate the model
	print(model.evaluate(X_test, y_test))

	### Predicting through the model
	y_pred = model.predict(X_test)
	print(y_pred[1:10, ])
	y_pred = np.argmax(y_pred, axis = 1)
	y_pred = y_pred.reshape(y_pred.shape[0], 1)
	y_test = np.argmax(y_test, axis = 1)
	y_test = y_test.reshape(y_test.shape[0], 1)
	print(y_test[1:10, 0])
	print(y_pred[1:10, 0])

	### Plotting the predictions
	plt.figure(1, figsize = (20, 6))
	plt.suptitle('Prediction')
	plt.title('History='+str(historyLength)+', Future='+str(1))
	plt.plot(y_test, 'r--', label = 'Actual')
	plt.plot(y_pred, 'gx', label = 'Predicted')
	plt.savefig('pred.jpg')


	# Save the model
	model.save('GRU1.tfl')

