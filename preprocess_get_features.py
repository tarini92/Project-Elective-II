import os
import itertools
import operator
import numpy as np
from PIL import Image
import inception
import pickle


### Extracts and saves the bounding boxes of objects from the video frames
def bounding_box_extract(image_folder, gt_path, save_folder):
	with open(gt_path, 'r') as gt_file:
		for line in gt_file:
			line_list = line.split(",")
			frame_num = int(line_list[0])
			target_id = int(line_list[1])
			x = float(line_list[2])
			y = float(line_list[3])
			w = float(line_list[4])
			h = float(line_list[5])
			img_name = str('{:06d}'.format(frame_num)) + '.jpg'
			img_path = os.path.join(image_folder, img_name)
			img = Image.open(img_path)
			bb_dim = (x, y, x+w, y+h)
			bb = img.crop(bb_dim)
			save_path = str(frame_num) + '_' + str(target_id) + '.jpg'
			bb.save(os.path.join(save_folder, save_path))
			print('Bounding Box extracted for target %d in frame %d'% (target_id, frame_num))

### Get contiguous frame sequences for all targets, returns list of lists
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

		#print(len(target_ids))
		### Read the file again
		gt_file.seek(0)

		### Get the ordered list of frames for every object
		### initialize the list for every target
		target_frames_list = [[] for _ in range(len(target_ids))]
		for line in gt_file:
			line_list = line.split(",")
			target_frames_list[int(line_list[1])-1].append(int(line_list[0]))


		#print(len(target_frames_list))
		#print(target_frames_list)
		frame_sequences = [[] for _ in range(len(target_ids))]

		### Splitting the contiguous frames for a target in the respective sequence
		for i, target_list in enumerate(target_frames_list):
			for key, group in itertools.groupby(enumerate(target_list), lambda x: x[0] - x[1]):  
				frame_sequences[i].append(list(map(operator.itemgetter(1), group)))

		#print(len(frame_sequences))
		#print(frame_sequences)
		return frame_sequences


### Extract features for each image(frame and bounding box) from inception
### and generate sequences of arrays of feature vectors to be fed into RNN
def get_feature_sequence(bb_images_path ,main_frames_path , frame_sequences):
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
				features.append(model.transfer_values(global_img_path))
				features.append(model.transfer_values(bb_path))
				features = np.array(features)
				features = features.reshape(1, features.shape[0]*features.shape[1])
				### Appending the label
				if j != len(seq) - 1:
					features = np.insert(features, 4096, 0).reshape(1, 4097)
					#print("label inserted")
				else:
					features = np.insert(features, 4096, 1).reshape(1, 4097)
					#print("label inserted")
				
				#print(features.shape)
				bb_array.append(features)

			bb_array = np.vstack(bb_array)
			feat_seq.append(bb_array)
			print(bb_array.shape)
			print("Sequence %d generated" % count)
			count += 1

	return feat_seq


gt_file_path = 'C:\\Users\\Akanksha\\Desktop\\Sem III\\ML PE\\MOT\\MOT Dataset\\2DMOT2015\\2DMOT2015\\train\\TUD-Campus\\gt\\gt.txt'
bb_images_path = 'C:\\Users\\Akanksha\\Desktop\\Sem III\\ML PE\\MOT\\MOT Dataset\\2DMOT2015\\2DMOT2015\\train\\TUD-Campus\\bounding_boxes_img1'
main_frames_path = 'C:\\Users\\Akanksha\\Desktop\\Sem III\\ML PE\\MOT\\MOT Dataset\\2DMOT2015\\2DMOT2015\\train\\TUD-Campus\\img1'
save_folder = 'C:\\Users\\Akanksha\\Desktop\\Sem III\\ML PE\\MOT\\MOT Dataset\\2DMOT2015\\2DMOT2015\\train\\TUD-Campus'

### to be run only once to extract bounding boxes
#bounding_box_extract(main_frames_path, gt_file_path, bb_images_path)

frame_sequences = get_frame_sequences(gt_file_path)

features_seq = get_feature_sequence(bb_images_path, main_frames_path, frame_sequences)

# to store the feature array sequences
outFile = os.path.join(save_folder, 'outFile.npz') 
with open(outFile, 'wb') as f:
	pickle.dump(features_seq, f)


#gt_file_path = 'C:\\Users\\Akanksha\\Desktop\\Sem III\\ML PE\\MOT\\MOT Dataset\\2DMOT2015\\2DMOT2015\\train\\TUD-Stadtmitte\\gt\\gt.txt'
#bb_images_path = 'C:\\Users\\Akanksha\\Desktop\\Sem III\\ML PE\\MOT\\MOT Dataset\\2DMOT2015\\2DMOT2015\\train\\TUD-Stadtmitte\\bounding_boxes_img1'
#main_frames_path = 'C:\\Users\\Akanksha\\Desktop\\Sem III\\ML PE\\MOT\\MOT Dataset\\2DMOT2015\\2DMOT2015\\train\\TUD-Stadtmitte\\img1'
#save_folder = 'C:\\Users\\Akanksha\\Desktop\\Sem III\\ML PE\\MOT\\MOT Dataset\\2DMOT2015\\2DMOT2015\\train\\TUD-Stadtmitte'

#bounding_box_extract(main_frames_path, gt_file_path, bb_images_path)

#frame_sequences = get_frame_sequences(gt_file_path)

#features_seq = get_feature_sequence(bb_images_path, main_frames_path, frame_sequences)
#for arr in features_seq:
#	print(arr.shape)