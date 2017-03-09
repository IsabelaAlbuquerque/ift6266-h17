import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py


def visualize(img_input, img_target, img_pred, save = True, save_file = 'test.jpg'):
	for idx in range(0, img_pred.shape[0]):

		center = (int(np.floor(img_input[idx].shape[0] / 2.)), int(np.floor(img_input[idx].shape[1] / 2.)))
	
		# True full image
		true_full = np.copy(img_input[idx])
		true_full = true_full
		true_full[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = img_target[idx]
	
		# Predicted full image
		pred_full = np.copy(img_input[idx])
		pred_full = pred_full
		pred_full[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = img_pred[idx] 

		plt.subplot(2, img_pred.shape[0], idx+1)
		plt.imshow(true_full)
		plt.title('True')
		plt.axis('off')

		plt.subplot(2, img_pred.shape[0], idx+img_pred.shape[0]+1)
		plt.imshow(pred_full)
		plt.title('Pred')
		plt.axis('off')

	if save:	
		plt.savefig(save_file, bbox_inches='tight')	
	plt.show()

def just_save(title_keyword, img_input, img_target, img_pred, save_file):
	
	title_keyword = 'Epoch' + str(title_keyword)

	for idx in range(0, img_pred.shape[0]):

		center = (int(np.floor(img_input[idx].shape[0] / 2.)), int(np.floor(img_input[idx].shape[1] / 2.)))
	
		# True full image
		true_full = np.copy(img_input[idx])
		true_full = true_full
		true_full[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = img_target[idx]
	
		# Predicted full image
		pred_full = np.copy(img_input[idx])
		pred_full = pred_full
		pred_full[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = img_pred[idx] 

		plt.subplot(2, img_pred.shape[0], idx+1)
		plt.imshow(true_full)
		#plt.title('True')
		plt.axis('off')

		plt.subplot(2, img_pred.shape[0], idx+img_pred.shape[0]+1)
		plt.imshow(pred_full)
		#plt.title('Pred')
		plt.axis('off')

	plt.suptitle(title_keyword)
	plt.savefig(save_file, bbox_inches='tight')	
	#plt.show()	 	 

def load_model_visualize(model, save = True, save_file = 'test.jpg', number_of_images = 10, valid_data = 'valid_data.hdf'):
	
	# Load the model
	model = load_model(model)

	# Read samples from the validation data
	open_file = h5py.File(valid_data, 'r')
	inputs_samples = open_file['inputs'][105:105+number_of_images]
	targets_samples = open_file['targets'][105:105+number_of_images]
	open_file.close()

	# Predict outputs
	predict_samples = model.predict(inputs_samples)

	visualize(inputs_samples, targets_samples, predict_samples, save, save_file)

def load_model_save_results_per_epoch(base_model, number_of_epochs, save_file = 'test.jpg', number_of_images = 8, valid_data = 'valid_data.hdf'):
	
	# Read samples from the validation data
	open_file = h5py.File(valid_data, 'r')
	inputs_samples = open_file['inputs'][10000:10000+number_of_images]
	targets_samples = open_file['targets'][10000:10000+number_of_images]
	open_file.close()

	# Load the model
	for epc in xrange(0, number_of_epochs): 
		if epc < 10:
			model = base_model + '-0-0'+ str(epc) + '.h5'
		else:
			model = base_model + '-0-'+ str(epc) + '.h5'
		
		model = load_model(model)

		# Predict outputs
		predict_samples = model.predict(inputs_samples)
		save_file_epc = save_file + '_epoch_' + str(epc) + '.jpg'

		just_save(epc, inputs_samples, targets_samples, predict_samples, save_file_epc)	

def save_image_many_epochs(base_model, epochs, image = 10000, save_file = 'test.jpg', valid_data = 'valid_data.hdf'):
	
	# Read samples from the validation data
	open_file = h5py.File(valid_data, 'r')
	x = open_file['inputs'][image:image+1]
	open_file.close()


	count = 1
	plt.subplot(1, len(epochs)+1, count)
	plt.imshow(x[0])
	plt.axis('off')

	for i in epochs: 
		if i < 10:
			model = base_model + '-0-0'+ str(i) + '.h5'
		else:
			model = base_model + '-0-'+ str(i) + '.h5'
		
		model = load_model(model)
		#x.reshape(1, 64, 64, 3)
		generated = model.predict(x)


		center = (int(np.floor(x[0].shape[0] / 2.)), int(np.floor(x[0].shape[1] / 2.)))
		# Predicted full image
		x_gen = np.copy(x[0])
		x_gen[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = generated 

		count += 1
		plt.subplot(1, len(epochs)+1, count)
		plt.imshow(x_gen)
		plt.axis('off')

	plt.savefig(save_file, bbox_inches='tight')


def true_generated_batch(img_center_true, img_center_generated):
	batch_size = img_center_true.shape[0]
	inputs_discriminator = np.vstack([img_center_true, img_center_generated])
	targets_discriminator = np.zeros([2*batch_size, 2])
	targets_discriminator[0:batch_size, 0] = 1	# true image: target = (1,0)
	targets_discriminator[batch_size:, 1] = 1	# generated image: target = (0,1)

	return inputs_discriminator, targets_discriminator

if __name__ == '__main__':
	# Results 1
	#load_model_visualize('cnnL2AE.h5', number_of_images=2, save_file = 'cnnL2AE_2imgs.jpg')
	#load_model_visualize('cnnL1AE.h5', number_of_images=2, save_file = 'cnnL1AE_2imgs.jpg')

	# Results2
	#load_model_save_results_per_epoch('cnnL2AE', 14, save_file = 'cnnL2AE')
	
	#save_image_many_epochs('cnnL2AE', [0, 5, 9, 13], image = 15000, save_file = 'cnnL2AE_img1.jpg')
	#save_image_many_epochs('cnnL2AE', [0, 5, 9, 13], image = 3, save_file = 'cnnL2AE_img2.jpg')
	#save_image_many_epochs('cnnL2AE', [0, 5, 9, 13], image = 1000, save_file = 'cnnL2AE_img3.jpg')
	#save_image_many_epochs('cnnL2AE', [0, 5, 9, 13], image = 37, save_file = 'cnnL2AE_img4.jpg')
	#save_image_many_epochs('cnnL2AE', [0, 5, 9, 13], image = 21032, save_file = 'cnnL2AE_img5.jpg')
	#save_image_many_epochs('cnnL2AE', [0, 5, 9, 13], image = 19421, save_file = 'cnnL2AE_img6.jpg')
	save_image_many_epochs('cnnL2AE', [0, 5, 9, 13], image = 14444, save_file = 'cnnL2AE_img7.jpg')
	#save_image_many_epochs('cnnL2AE', [0, 5, 9, 13], image = 1989, save_file = 'cnnL2AE_img8.jpg')
	#save_image_many_epochs('cnnL2AE', [0, 5, 9, 13], image = 30237, save_file = 'cnnL2AE_img9.jpg')
