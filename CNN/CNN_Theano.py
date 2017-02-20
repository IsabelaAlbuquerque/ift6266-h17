import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import cPickle
import gzip


with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)


class FullyConnectedOutputLayer(object):
	
	def __init__(self, rng, input, n_inputs, n_outputs):
		self.input = input
		self.W = theano.shared(np.asarray(rng.normal(0, 0.1, (n_inputs, n_outputs)), dtype=theano.config.floatX), borrow=True)
		self.b = theano.shared(np.asarray(rng.normal(0, 0.1, (n_outputs,)), dtype=theano.config.floatX), borrow=True)

		a = (T.dot(input, self.W) + self.b)

		self.output = T.nnet.softmax(a)
		self.params = [self.W, self.b]


class FullyConnectedLayer(object):

	def __init__(self, rng, input, n_inputs, n_outputs):

		self.input = input
		self.W = theano.shared(np.asarray(rng.normal(0, 0.1, (n_inputs, n_outputs)), dtype=theano.config.floatX), borrow=True)
		self.b = theano.shared(np.asarray(rng.normal(0, 0.1, (n_outputs,)), dtype=theano.config.floatX), borrow=True)

		a = (T.dot(input, self.W) + self.b)
		h = T.nnet.sigmoid(a)

		self.output = h
		self.params = [self.W, self.b]


class ConvLayer(object):
	"""Convolutional + ReLU + Max Pooling

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols) """
	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), W=None):
		#assert image_shape[1] == filter_shape[1] # Testa se o numero de input feature maps e igual para as duas tuplas
		self.input = input
		# Inicializacao do tensor dos pesos
		if W is None:
			W = theano.shared(np.asarray(rng.normal(0, 0.1, size=filter_shape), dtype=theano.config.floatX), name = 'W', borrow=True)
		self.W = W

		# Bias is a 1D tensor -- one bias per output feature map
		b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)
		
		conv_out = conv2d(input=input,filters=self.W,filter_shape=filter_shape,input_shape=image_shape)
		pooled_out = pool.pool_2d(input=conv_out, ds=poolsize, ignore_border=True)
        
		self.output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		
		#self.output = T.nnet.relu(pooled_out)

		self.params = [self.W, self.b]
		#self.params = [self.W]

		self.input = input

def accuracy_calc(output, target):

	accuracy = T.mean(T.eq(output, target), dtype=theano.config.floatX)
	return accuracy        


def buildModelAndEvaluate(learningRate, batchSize, epochs, n_hidden, nkernels, train_set, valid_set, test_set):
	x = T.tensor4('x')
	y_onehot = T.matrix('y_onehot')
	y_integers = T.ivector('y_integers')

	rng = np.random.RandomState(4325)


	# Forward pass
	#firstConvInput = x.reshape((batchSize, 1, 28, 28)) #4D tensor. 1 is the number of channels

	firstConvInput = x

	# First conv layer. 
	# Filter shape is 5x5 -> image is reduced to (28-5+1, 28-5+1) = (24, 24)
	# After maxpooling: (24, 24) -> (12, 12) 
 	convLayer1 = ConvLayer(rng, input=firstConvInput, image_shape=(None, 1, 28, 28), filter_shape=(nkernels[0], 1, 5, 5), poolsize=(2, 2))
 
 	# Second conv layer.
 	# (12, 12) -> (12-5+1, 12-5+1) = (8, 8)
 	# After maxpooling: (8, 8) -> (4, 4)
 	convLayer2 = ConvLayer(rng, input=convLayer1.output, image_shape=(None, nkernels[0], 12, 12), filter_shape=(nkernels[1], nkernels[0], 5, 5), poolsize=(2, 2))

	# First fully connected layer.
 	# Its is necessary to transform a 4D tensor into a 2D matrix
 	# The shape of the input matrix is (batchSize, nkernels[1]*4*4)
	firstFullyConInput = convLayer2.output.flatten(2)

	fullyConnectedLayer1 = FullyConnectedLayer(rng, firstFullyConInput, nkernels[1]*4*4, n_hidden)

	fullyConnectedLayer2 = FullyConnectedOutputLayer(rng, fullyConnectedLayer1.output, n_hidden, 10)

	cost = T.nnet.categorical_crossentropy(fullyConnectedLayer2.output, y_onehot).sum()

	accuracy_train = accuracy_calc(T.argmax(fullyConnectedLayer2.output, axis=1), T.argmax(y_onehot, axis=1))
	accuracy_validtest = accuracy_calc(T.argmax(fullyConnectedLayer2.output, axis=1), y_integers)


	# Backward pass
	all_parameters = fullyConnectedLayer2.params + fullyConnectedLayer1.params + convLayer2.params + convLayer1.params
	grads = T.grad(cost, all_parameters)
	updates = [(param_i, param_i - learningRate * grad_i/batchSize) for param_i, grad_i in zip(all_parameters, grads)]


	# Theano functions for training and validation/testing
	training = theano.function([x, y_onehot], accuracy_train, updates = updates)





	valid_test = theano.function([x, y_integers], accuracy_validtest)

	

	# PREPARING THE DATA
	x_train_set = train_set[0] 					# [50000, 784]
	y_train_set = train_set[1]

	y_train_set_onehot = np.zeros([y_train_set.shape[0], 10])
	for i in range(0, y_train_set.shape[0]):
		y_train_set_onehot[i, y_train_set[i]] = 1.0

	x_valid_set = valid_set[0]
	y_valid_set = valid_set[1]
	y_valid_set = y_valid_set.astype('int32')

	x_test_set = test_set[0]
	y_test_set = test_set[1].astype('int32')

	train_setSize = y_train_set.shape[0]
	numberOfBatches = np.ceil((1.0*train_setSize)/(1.0*batchSize)).astype('int64')	


	# TRAIN STEP

	for i in xrange(epochs):

 		aux = np.c_[x_train_set.reshape(len(x_train_set), -1), y_train_set_onehot.reshape(len(y_train_set_onehot), -1)]

 		x_shuffle = aux[:, :x_train_set.size//len(x_train_set)].reshape(x_train_set.shape) 

		y_shuffle = aux[:, x_train_set.size//len(y_train_set_onehot):].reshape(y_train_set_onehot.shape)

		np.random.shuffle(aux)

		x_train_set = x_shuffle
		y_train_set_onehot = y_shuffle
 	
 		for j in xrange(numberOfBatches):
 		
 	 		input_batch = x_train_set[j*batchSize:min((j+1)*batchSize, x_train_set.shape[0]), :]
 	 		label_batch = y_train_set_onehot[j*batchSize: min((j+1)*batchSize, y_train_set_onehot.shape[0]), :]
 	 		
 	 		input_batch = input_batch.reshape((-1, 1, 28, 28))
 	 		accuracy_batch = training(input_batch, label_batch) 	 	
 	 	
			print('Accuracy in epoch', i, 'for batch', j, 'was', accuracy_batch)

		# Validation
		x_valid_set = x_valid_set.reshape((-1, 1, 28, 28))
		accuracy_valid = valid_test(x_valid_set, y_valid_set)
		print('Accuracy for validation in epoch', i, 'was', accuracy_valid)

	# Testing
	x_test_set = x_test_set.reshape((-1, 1, 28, 28))
	accuracy_test = valid_test(x_test_set, y_test_set)
	print('Accuracy for the testing was', accuracy_test)

if __name__ == '__main__':
    buildModelAndEvaluate(0.5, 2500, 5, 50, [20, 50], train_set, valid_set, test_set)