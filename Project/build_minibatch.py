import os, sys
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image
from skimage.transform import resize
import numpy as np

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

	print data_path + "/*.jpg"
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
			#Image.fromarray(input).show()
			#Image.fromarray(target).show()
			#print i, caption_dict[cap_id]
		else:
			None
	print 'Finished preparing data!!'			
	return np.asarray(inputs)/255, np.asarray(targets)/255, captions		
        	
if __name__ == '__main__':
	inp, targ, cap = build_minibatch(5, 2, 1, False)
	#print len(inp)
	#print len(targ)
	#print len(cap) 
	#for i,img in enumerate(targ):
	#	img = img.flatten(1)
	#	targ[i] = img
		
