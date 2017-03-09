import os, sys
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image
from skimage.transform import resize
import numpy as np
import h5py
from visualize import visualize

'''
Given the resized data (64x64), this method builts a 
minibatch of images with square black holes in the center.
A list of corrupted images is returned in inputs, respective 
targets in a list named targets. 
Also, the list captions returns a set of captions for each
image.

For train = 1, images are from the training data;
for train = 0, images are from the validation data.

all = True if batch_size = data set length
In this case, batch_idx = 0

'''

def build_minibatch(batch_idx, batch_size, train, all, split="complete", caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"):
	if train == 1:
		mscoco = "inpainting/useful/train/"
	else:
		mscoco = "inpainting/useful/valid/"

	data_path = os.path.join(mscoco, split)


	caption_path = os.path.join(mscoco, caption_path)
	with open(caption_path) as fd:
		caption_dict = pkl.load(fd)


	print(data_path + "/*.jpg")
	imgs = glob.glob(data_path + "/*.jpg")

	total_amount = len(imgs)
	
	if all == True:
		batch_imgs = imgs
	else:		
		batch_imgs = imgs[batch_idx*batch_size:(batch_idx+1)*batch_size]

	inputs = []
	targets = []
	captions = []

	for i, img_path in enumerate(batch_imgs):

		img = Image.open(img_path)
		img_array = np.array(img)

		cap_id = os.path.basename(img_path)[:-4]

        ### Get input/target from the images
		center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
		#print img_array.shape
		
		if len(img_array.shape) == 3:
			input = np.copy(img_array)
			input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
			target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
			inputs.append(input)
			targets.append(target)
			captions.append(caption_dict[cap_id])
		else:
			None
	print('Finished preparing data!!')			
	return np.asarray(inputs), np.asarray(targets), np.asarray(captions)

def build_hdf5_all_data(train_or_valid, normalization = True, save_captions = False, split='complete', caption_path="dict_key_imgID_value_caps_train_and_valid.pkl", hdf5_filename='data.hdf'):
	if train_or_valid == 1:
		mscoco = "inpainting/useful/train/"
		hdf5_filename = 'train_' + hdf5_filename
	else:
		mscoco = "inpainting/useful/valid/"
		hdf5_filename = 'valid_' + hdf5_filename

	data_path = os.path.join(mscoco, split)
	print(data_path + "/*.jpg")
	imgs = glob.glob(data_path + "/*.jpg")

	caption_path = os.path.join(mscoco, caption_path)
	with open(caption_path) as fd:
		caption_dict = pkl.load(fd)
	
	inputs = []
	targets = []
	captions = []

	for i, img_path in enumerate(imgs):

		img = Image.open(img_path)
		img_array = np.array(img)
		center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
		cap_id = os.path.basename(img_path)[:-4]		
		
		if len(img_array.shape) == 3:
			input = np.copy(img_array)
			input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
			target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
			inputs.append(input)
			targets.append(target)
			captions.append(caption_dict[cap_id])
		else:
			None

	inputs = np.asarray(inputs) 
	targets = np.asarray(targets)
	captions = np.asarray(captions)

	if normalization:
		inputs = (inputs/255.0) 
		targets = (targets/255.0)


	hdf5 = h5py.File(hdf5_filename, 'w')
	dataset = hdf5.create_dataset('inputs', data=inputs)
	dataset = hdf5.create_dataset('targets', data=targets)
	
	if save_captions:
		dataset = hdf5.create_dataset('captions', data=captions)

	hdf5.close()


if __name__ == '__main__':

	build_hdf5_all_data(0)
