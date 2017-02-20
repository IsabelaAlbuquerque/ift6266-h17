#Assignment 1 - IFT6266 winter 2017
#MNIST data classification using a feed forward neural net with 1 hidden layer

# Shuffle 
# Validar em cada epoch 
# Calcular gradiente do bias ok
# Incluir regularization 
# Testar import dos dados ok


import numpy as np

import cPickle
import gzip
 
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)


def decoding_softmax(f_Softmax):

	decod_output = f_Softmax.argmax(axis=0)
	 
	return decod_output

def accuracy_calc(output, target):

	comparison = 1.0*np.equal(output, target)
	accuracy = sum(comparison)/target.shape[0]

	return accuracy 

def conv_to_softmax(inpt):
		maximum_out = np.max(inpt)
		aux = inpt - maximum_out
		expon = np.exp(aux)
		sum_expon = sum(expon)
		f_softmax = expon/sum_expon

		return f_softmax

# PREPARING THE DATA
x_train_set = train_set[0].T 					# [784, 50000]
y_train_set = train_set[1]


y_train_set_modif = np.zeros([10, y_train_set.shape[0]])
for i in range(0, y_train_set.shape[0]):
	y_train_set_modif[y_train_set[i], i] = 1.0

	
x_valid_set = valid_set[0].T
y_valid_set = valid_set[1]

#y_valid_set_modif = np.zeros([10, y_valid_set.shape[0]])
#for i in range(0, y_valid_set.shape[0]):
#	y_valid_set_modif[y_valid_set[i], i] = 1.0

x_test_set = test_set[0].T
y_test_set = test_set[1]

#y_test_set_modif = np.zeros([10, y_test_set.shape[0]])
#for i in range(0, y_test_set.shape[0]):
#	y_test_set_modif[y_test_set[i], i] = 1.0


#PARAMETERS AND HYPERPARAMETERS 
batchSize = 1000
epochs = 10
learningRate = 0.5

n_inputs = 784
n_outputs = 10
n_hidden = 150

train_setSize = y_train_set.shape[0]


numberOfBatches = np.ceil((1.0*train_setSize)/(1.0*batchSize)).astype('int64')


#W_ih = np.random.random((n_hidden+1, n_inputs+1))		#[51, 785]
#W_ho = np.random.random((n_outputs, n_hidden+1))		#[10, 51]

W_ih = np.random.normal(0, 0.1, (n_hidden, n_inputs))		#[51, 785]
W_ho = np.random.normal(0, 0.1, (n_outputs, n_hidden))		#[10, 51]

#x_train_set = np.vstack((1.0*np.ones((1, x_train_set.shape[1])), x_train_set))
#x_valid_set = np.vstack((1.0*np.ones((1, x_valid_set.shape[1])), x_valid_set))
#x_test_set = np.vstack((1.0*np.ones((1, x_test_set.shape[1])), x_test_set))


#TRAIN STEP
for i in range(epochs):

	x_T = x_train_set.T
 	y_T = y_train_set_modif.T

 	aux = np.c_[x_T.reshape(len(x_T), -1), y_T.reshape(len(y_T), -1)]

 	x_Taux = aux[:, :x_T.size//len(x_T)].reshape(x_T.shape) 

	y_Taux = aux[:, x_T.size//len(y_T):].reshape(y_T.shape)

	np.random.shuffle(aux)

	x_train_set = x_Taux.T
	y_train_set_modif = y_Taux.T

 	for j in range(numberOfBatches):

 	 	input_batch = x_train_set[:, j*batchSize:min((j+1)*batchSize, x_train_set.shape[1])]
 	 	label_batch = y_train_set_modif[:, j*batchSize: min((j+1)*batchSize, y_train_set_modif.shape[1])]
 	 	 	 	
 	 	#FORWARD PASS
		a_ih = W_ih.dot(input_batch)			
		h_ih = 1/(1 + np.exp(-1*a_ih))				#[51, batchSize]

		#h_ih[0, :] = 1
		
		a_ho = W_ho.dot(h_ih)							# [10, batchSize]
		
		f_softmax = conv_to_softmax(a_ho)

		output_batch_decod = decoding_softmax(f_softmax)

		accuracy_batch = accuracy_calc(output_batch_decod, decoding_softmax(label_batch))

		print('Accuracy in epoch', i, 'for batch', j, 'was', accuracy_batch)

		#BACKWARD PASS
		delta_oh = f_softmax - label_batch
		grad_oh = delta_oh.dot(h_ih.T) / input_batch.shape[1]
		
		grad_hi = W_ho.T.dot(delta_oh) * (h_ih * (1 - h_ih))
		grad_hi = grad_hi.dot(input_batch.T) / input_batch.shape[1]

		W_ho = W_ho - learningRate * grad_oh
		W_ih = W_ih - learningRate * grad_hi



	#VALIDATION 
	a_ih = W_ih.dot(x_valid_set)				
	h_ih = 1/(1 + np.exp(-1*a_ih))				
	#h_ih[0, :] = 1
		
	a_ho = W_ho.dot(h_ih)

	f_softmax_valid = conv_to_softmax(a_ho)

	output_valid = decoding_softmax(f_softmax_valid)
	accuracy_valid = accuracy_calc(output_valid, y_valid_set)
	print('Accuracy for validation in epoch', i, 'was', accuracy_valid)
	

#TESTING

a_ih = W_ih.dot(x_test_set)				
h_ih = 1/(1 + np.exp(-1*a_ih))				
#h_ih[0, :] = 1
		
a_ho = W_ho.dot(h_ih)						
		
f_softmax_test = conv_to_softmax(a_ho)

output_test = decoding_softmax(f_softmax_test)

accuracy_test = accuracy_calc(output_test, y_test_set)

print('Accuracy for the testing was', accuracy_test)

