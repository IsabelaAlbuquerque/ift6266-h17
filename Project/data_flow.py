import h5py
import numpy as np
import cPickle as pkl
import string
from tempfile import mkdtemp
import os
import gensim


def batch_generator(hdf5_name, data_size, batch_size=32):
	
	number_of_slices = int(np.ceil(data_size/batch_size))
	
	while True:
		open_file = h5py.File(hdf5_name, 'r')

		for i in xrange(0, number_of_slices):
			inputs_batch = open_file['inputs'][i*batch_size:min((i+1)*batch_size, data_size)]
			targets_batch = open_file['targets'][i*batch_size:min((i+1)*batch_size, data_size)]
       			
       			yield (inputs_batch, targets_batch)

		open_file.close()			


def batch_generator_with_embeddings(hdf5_name, hdf5_name_captions_emb, data_size, load_captions_emb=True, batch_size=32):
	
	number_of_slices = int(np.ceil(data_size/batch_size))
	
	if (load_captions_emb):
		while True:
			open_file_cap_emb = h5py.File(hdf5_name_captions_emb, 'r')
			open_file = h5py.File(hdf5_name, 'r')

			for i in xrange(0, number_of_slices):
				caps_emb_batch = open_file_cap_emb['emb'][i*batch_size:min((i+1)*batch_size, data_size)]
				caps_emb_batch_out = np.zeros([batch_size, 300])

				for j in xrange(0, batch_size):
					rand_num = np.random.randint(0, 5)
					caps_emb_batch_out[j, :] = caps_emb_batch[i, rand_num, :]
        			
				inputs_batch = open_file['inputs'][i*batch_size:min((i+1)*batch_size, data_size)]
				targets_batch = open_file['targets'][i*batch_size:min((i+1)*batch_size, data_size)]	
        			
        			yield (inputs_batch, targets_batch, caps_emb_batch_out)

			open_file.close()
			open_file_cap_emb.close()

	else:
		while True:
			open_file = h5py.File(hdf5_name, 'r')

			for i in xrange(0, number_of_slices):
				inputs_batch = open_file['inputs'][i*batch_size:min((i+1)*batch_size, data_size)]
				targets_batch = open_file['targets'][i*batch_size:min((i+1)*batch_size, data_size)]
        			
        			yield (inputs_batch, targets_batch)

			open_file.close

def caps_processing():

	model = gensim.models.KeyedVectors.load_word2vec_format(
	    '/home/isabela/Desktop/Project/GoogleNews-vectors-negative300.bin',
	    binary=True)

	with open('/home/isabela/Desktop/Project/inpainting/dict_key_imgID_value_caps_train_and_valid.pkl') as f:
	    dict = pkl.load(f)

	files = sorted(os.listdir('/home/isabela/Desktop/Project/inpainting/train2014/'))

	fname = os.path.join(mkdtemp(), 'newfile.dat')
	train = np.memmap(fname, dtype='float32', mode='w+', shape=(len(files), 5, 300))

	for i in range(0, len(files)):
	    key = files[i][:-4]
	    for j in range(0, len(dict[key])):
	        if (j > 4):
	            continue
	        l = dict[key][j].split()
	        m = 0
	        for k in range(0, len(l)):
	            vector = model[l[k]]
	            dset_train[i, j, :] += vector
	            m += 1
	        dset_train[i, j, :] /= m
	        dset_train[i, j, :] = (dset_train[i, j, :] + 1) / 2    

	np.save('/home/isabela/Desktop/Project/inpainting/useful/all_words_2_vectors_train_norm.npy', train)

	files = sorted(os.listdir('/home/isabela/Desktop/Project/inpainting/val2014/'))

	fname = os.path.join(mkdtemp(), 'newfile.dat')
	val = np.memmap(fname, dtype='float32', mode='w+', shape=(len(files), 5, 300))

	for i in range(0, len(files)):
	    key = files[i][:-4]
	    for j in range(0, len(dict[key])):
	        if (j > 4):
	            continue
	        l = dict[key][j].split()
	        m = 0
	        for k in range(0, len(l)):
	            vector = model[l[k]]
	            dset_val[i, j, :] += vector
	            m += 1
	        dset_val[i, j, :] /= m
	        dset_val[i, j, :] = (dset_val[i, j, :] + 1) / 2    

	np.save('/home/isabela/Desktop/Project/inpainting/useful/all_words_2_vectors_val_norm.npy', val)			

		