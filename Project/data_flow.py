import h5py
import numpy as np

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
			open_file = h5py.File(hdf5_name, 'r')
			open_file_cap_emb = h5py.File(hdf5_name_captions_emb, 'r')

			for i in xrange(0, number_of_slices):
				caps_emb_batch = open_file_cap_emb['emb'][i*batch_size:min((i+1)*batch_size, data_size)]
				caps_emb_batch_out = np.zeros([batch_size, 300])

				for i in xrange(0, batch_size):
					rand_num = np.random.randint(0, 5)
					caps_emb_batch_out[i, :] = caps_emb_batch[i, rand_num, :]
        			
				inputs_batch = open_file['inputs'][i*batch_size:min((i+1)*batch_size, data_size)]
				targets_batch = open_file['targets'][i*batch_size:min((i+1)*batch_size, data_size)]	
				captions_batch = open_file['captions'][i*batch_size:min((i+1)*batch_size, data_size)]	
        			
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


if __name__ == '__main__':
	