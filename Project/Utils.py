import numpy as np
import matplotlib.pyplot as plt
import h5py

from keras.models import load_model


def visualize(img_input, img_target, img_pred, save_file = 'file_name.jpg', save = True):
	for idx in range(0, img_pred.shape[0]):

		center = (int(np.floor(img_input[idx].shape[0] / 2.)), int(np.floor(img_input[idx].shape[1] / 2.)))
	
		true = np.copy(img_input[idx])
		true[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = img_target[idx]

		pred = np.copy(img_input[idx])
		pred[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = img_pred[idx] 


		plt.subplot(2, img_pred.shape[0], idx+1)
		plt.imshow(true)
		plt.axis('off')

		plt.subplot(2, img_pred.shape[0], idx+img_pred.shape[0]+1)
		plt.imshow(pred)
		plt.axis('off')

	if save:	
		plt.savefig(save_file, bbox_inches='tight')	
	plt.show()

def just_save(img_input, img_target, img_pred, save_file):
	
	for idx in range(0, img_pred.shape[0]):

		center = (int(np.floor(img_input[idx].shape[0] / 2.)), int(np.floor(img_input[idx].shape[1] / 2.)))
	
		true = np.copy(img_input[idx])
		true[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = img_target[idx]
	
		pred = np.copy(img_input[idx])
		pred[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = img_pred[idx] 

		plt.subplot(2, img_pred.shape[0], idx+1)
		plt.imshow(true)
		plt.axis('off')

		plt.subplot(2, img_pred.shape[0], idx+img_pred.shape[0]+1)
		plt.imshow(pred)
		plt.axis('off')

	plt.savefig(save_file, bbox_inches='tight')	
 

def load_model_visualize(model, image = 10000, save = False, save_file = 'file_name.jpg', number_of_images = 5, valid_data = 'train_data.hdf'):
	
	model_to_predict = load_model(model)

	data = h5py.File(valid_data, 'r')
	inputs = data['inputs'][image:image+number_of_images]
	targets = data['targets'][image:image+number_of_images]
	data.close() 

	
	enc = load_model('model_encoder.h5')
	noise = np.random.normal(0., 0.2, (number_of_images, 4, 4, 128))
	noisy_inputs = enc.predict(inputs)
	noisy_inputs += noise
	model_samples = model_to_predict.predict(noisy_inputs)

	visualize(inputs, targets, model_samples, save, save_file)


def load_model_visualize_with_emb(model, image = 10000, save = False, save_file = 'file_name.jpg', number_of_images = 5, valid_data = 'valid_data.hdf', captions_data = 'embeddings_valid_norm.hdf'):
	
	model_to_predict = load_model(model)

	data = h5py.File(valid_data, 'r')
	data_cap = h5py.File(captions_data, 'r')
	inputs = data['inputs'][image:image+number_of_images]
	targets = data['targets'][image:image+number_of_images]
	embedd = data_cap['emb'][image:image+number_of_images]

	captions_selected = np.zeros([number_of_images, 300])

	for i in xrange(0, number_of_images):
		rand_num = np.random.randint(0, 5)
		captions_selected[i, :] = embedd[i, rand_num, :]

	data.close() 
	data_cap.close()

	enc = load_model('model_encoder.h5')
	noise = np.random.normal(0., 0.2, (number_of_images, 4, 4, 128))
	noisy_inputs = enc.predict(inputs_samples)
	inputs += noise
	model_samples = model_to_predict.predict([noisy_inputs, captions_selected])

	visualize(inputs, targets, model_samples, save, save_file)	


def load_model_epochs(base_model, number_of_epochs, image = 10000, save_file = 'file_name.jpg', number_of_images = 8, valid_data = 'valid_data.hdf'):
	
	data = h5py.File(valid_data, 'r')
	inputs = data['inputs'][image:image+number_of_images]
	targets = data['targets'][image:image+number_of_images]
	data.close()

	for epc in xrange(0, number_of_epochs): 
		if epc < 10:
			model = base_model + '-0-0'+ str(epc) + '.h5'
		else:
			model = base_model + '-0-'+ str(epc) + '.h5'
		
		model_to_predict = load_model(model)

		model_samples = model_to_predict.predict(inputs)
		save_file_epc = save_file + '_epoch_' + str(epc) + '.jpg'

		just_save(epc, inputs, targets, model_samples, save_file_epc)	


def plot_loss(pkl = 'losses.p'):
	loss_data = pickle.load(file(pkl))
	for key in loss_data:
		to_plot = loss_data[key]
		plt.plot(to_plot)
		
	plt.legend(loss_data.keys())	
	plt.show()			

