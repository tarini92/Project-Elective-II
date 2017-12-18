import os
import itertools
import operator
import numpy as np
from PIL import Image
import inception
import pickle


### Extracts and saves the bounding boxes of objects from the video frames in the bounding_boxes_img1 folder in the video sequence folder
def bounding_box_extract(image_folder, gt_path, save_folder):
	with open(gt_path, 'r') as gt_file:
		for line in gt_file:
			line_list = line.split(",")
			frame_num = int(line_list[0])     ### frame number 
			target_id = int(line_list[1])	  ### target number
			x = float(line_list[2])			  ### bounding box coordinates
			y = float(line_list[3])
			w = float(line_list[4])
			h = float(line_list[5])
			img_name = str('{:06d}'.format(frame_num)) + '.jpg'
			img_path = os.path.join(image_folder, img_name)
			img = Image.open(img_path)
			bb_dim = (x, y, x+w, y+h)
			bb = img.crop(bb_dim)         ### cropping the bounding box from the images 
			save_path = str(frame_num) + '_' + str(target_id) + '.jpg'
			bb.save(os.path.join(save_folder, save_path))                 # saving bounding boxes
			print('Bounding Box extracted for target %d in frame %d'% (target_id, frame_num))

def get_frame_sequences(gt_file_path):
	### Initialize the set to get the number of objects in the video sequence
	target_ids = set()
	### Opening ground truth file
	with open(gt_file_path, 'r') as gt_file:
		print("Reading ground truth file...")
		for line in gt_file:
			line_list = line.split(",")
			### Get the set of all objects in the video
			target_ids.add(int(line_list[1]))

		target_ids = sorted(target_ids)
		length = target_ids[-1]          ## length of the target sequences list.
										# last element is chosen as the length of the list as some of the targets may be missing
		
		### Read the file again
		gt_file.seek(0)

		### Get the ordered list of frames for every object
		### initialize the list for every target
		target_frames_list = [[] for _ in range(length)]
		for line in gt_file:
			line_list = line.split(",")
			target_frames_list[int(line_list[1])-1].append(int(line_list[0]))


		frame_sequences = [[] for _ in range(length)]

		### Splitting the contiguous frames for a target in the respective sequence
		for i, target_list in enumerate(target_frames_list):
			for key, group in itertools.groupby(enumerate(target_list), lambda x: x[0] - x[1]):  
				frame_sequences[i].append(list(map(operator.itemgetter(1), group)))

		return frame_sequences


### Extract features for each image(frame and bounding box) from inception network
### and generate sequences of arrays of feature vectors to be fed into RNN
def get_feature_sequence(bb_images_path ,main_frames_path , frame_sequences, gt_file_path):
	with open(gt_file_path, 'r') as gt:
		lines = gt.readlines()
	feat_seq = []  # list to hold the array sequences of features
	model = inception.Inception()
	count= 1
	for i, seq_list in enumerate(frame_sequences):
		for seq in seq_list:
			target_id = i+1
			bb_array = []
			for j, frame_num in enumerate(seq):
				features = []
				bb_name = str(frame_num) + '_' + str((i+1)) + '.jpg'
				bb_path = os.path.join(bb_images_path, bb_name)
				global_img_name = str('{:06d}'.format(frame_num)) + '.jpg'
				global_img_path = os.path.join(main_frames_path, global_img_name)
				features.extend(model.transfer_values(global_img_path))
				features.extend(model.transfer_values(bb_path))
				
				with Image.open(global_img_path) as global_img:
					width, height = global_img.size
				
				features.extend((0, 0, width, height))
				
				x = 0
				y = 0
				w = 0
				h = 0
				for line in lines:
					line_list = line.split(",")
					if int(line_list[0]) == frame_num and int(line_list[1]) == (i+1):
						x = float(line_list[2])
						y = float(line_list[3])
						w = float(line_list[4])
						h = float(line_list[5])
						break
				features.extend((x, y, (x+w), (y+h)))
				features = np.array(features)
				features = features.reshape(1, features.shape[0])

				### Appending the label
				if j != len(seq) - 1:
					features = np.insert(features, 4104, 0).reshape(1, 4105)
				else:
					features = np.insert(features, 4104, 1).reshape(1, 4105)
				
				bb_array.append(features)

			bb_array = np.vstack(bb_array)
			feat_seq.append(bb_array)
			print(bb_array.shape)
			print("Sequence %d generated" % count)
			count += 1

	return feat_seq

if __name__ == '__main__':
	# list of name of all training videos
	train_vid_list = ['ADL-Rundle-6', 'ADL-Rundle-8','ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2']

	gt_file_name = 'gt\\gt.txt'
	bb_images_path_name = 'bounding_boxes_img1'
	main_frames_path_name = 'img1'
	train_folder = 'C:\\Users\\Akanksha\\Desktop\\Sem III\\ML PE\\MOT\\MOT Dataset\\2DMOT2015\\2DMOT2015\\train'

	for name in train_vid_list:
		parent_folder = os.path.join(train_folder, name)

		gt_file_path = os.path.join(parent_folder, gt_file_name)
		bb_images_path = os.path.join(parent_folder, bb_images_path_name)
		main_frames_path = os.path.join(parent_folder, main_frames_path_name)
		print(gt_file_path)
		print(bb_images_path)
		print(main_frames_path)

		# to be called only when bounding boxes need to be formed adn saving in already existing folder naming bounding_boxes_img1
		#bounding_box_extract(main_frames_path, gt_file_path, bb_images_path)

		# to be called to extract frame sequences and features
		frame_sequences = get_frame_sequences(gt_file_path)
		features_seq = get_feature_sequence(bb_images_path, main_frames_path, frame_sequences, gt_file_path)

		# to save the extracted features in the parent folder
		outFile = os.path.join(parent_folder, 'outfile.npz')
		with open(outFile, 'wb') as f:
			pickle.dump(features_seq, f)

		print("Sequences extracted and saved for " + name)



