import numpy as np
import matplotlib.pyplot as plt

'''
Plot the true image and the image filled with the model prediction
'''


def visualize(img_input, img_target, img_pred):
	center = (int(np.floor(img_input.shape[0] / 2.)), int(np.floor(img_input.shape[1] / 2.)))
	
	# True full image
	true_full = np.copy(img_input)
	true_full = true_full
	true_full[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = img_target
	
	# Predicted full image
	pred_full = np.copy(img_input)
	pred_full = pred_full
	pred_full[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = img_pred 

	plt.subplot(2, 1, 1)
	plt.imshow(true_full)
	plt.title('True image')

	plt.subplot(2, 1, 2)
	plt.imshow(pred_full)
	plt.title('Predicted image')

	plt.show()


