import h5py
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

def batch_generator(train_or_valid, hdf5_name, batch_size=32, train_size = 82611, valid_size = 40438):
	
	if train_or_valid == 1:
		number_of_batches = int(round(train_size/batch_size))
	else:
		number_of_batches = int(round(valid_size/batch_size))	
	
	while True:
		open_file = h5py.File(hdf5_name, 'r')

		for i in xrange(0, number_of_batches):
			inputs_batch = open_file['inputs'][i*batch_size:(i+1)*batch_size]
			targets_batch = open_file['targets'][i*batch_size:(i+1)*batch_size]
        		yield (inputs_batch, targets_batch)	

		open_file.close()
