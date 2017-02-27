import numpy as np
import matplotlib.pyplot as plt

'''
Plot the true image and the image filled with the model prediction
'''


def visualize(img_input, img_target, img_pred):
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
	plt.savefig('test2.jpg', bbox_inches='tight')	
	plt.show()
	


