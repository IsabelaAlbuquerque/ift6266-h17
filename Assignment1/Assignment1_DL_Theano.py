import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import cPickle
import gzip
 
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)

# PREPARING THE DATA
x_train_set = train_set[0].T 					# [784, 50000]
y_train_set = train_set[1]

y_train_set_onehot = np.zeros([10, y_train_set.shape[0]])
for i in range(0, y_train_set.shape[0]):
	y_train_set_onehot[y_train_set[i], i] = 1.0

x_valid_set = valid_set[0].T
y_valid_set = valid_set[1]

x_test_set = test_set[0].T
y_test_set = test_set[1]

# PARAMETERS AND HYPERPARAMETERS
batchSize = 1000
epochs = 10
learningRate = 10

n_inputs = 784
n_outputs = 10
n_hidden = 50

train_setSize = y_train_set.shape[0]
numberOfBatches = np.ceil((1.0*train_setSize)/(1.0*batchSize)).astype('int64')

def accuracy_calc(output, target):

	accuracy = T.mean(T.eq(output, target), dtype=theano.config.floatX)
	return accuracy

# INPUTS
inputdata = T.dmatrix('inputdata')
target = T.dmatrix('target')

target_valid_test = T.dvector('target_valid_test')

# SHARED VARIABLES 
#w_ih = theano.shared(np.array(np.random.rand(n_hidden+1, n_inputs+1), dtype = theano.config.floatX))
#w_ho = theano.shared(np.array(np.random.rand(n_outputs, n_hidden+1), dtype = theano.config.floatX))

w_ih = theano.shared(np.array(np.random.normal(0, 0.1, (n_hidden, n_inputs)), dtype = theano.config.floatX))
w_ho = theano.shared(np.array(np.random.normal(0, 0.1, (n_outputs, n_hidden)), dtype = theano.config.floatX))
b_ih = theano.shared(np.array(np.random.normal(0, 0.1, (n_hidden, 1)), dtype = theano.config.floatX), broadcastable=(False, True))
b_ho = theano.shared(np.array(np.random.normal(0, 0.1, (n_outputs, 1)), dtype = theano.config.floatX), broadcastable=(False, True))

# Forward pass
h_hidden = nnet.sigmoid(T.dot(w_ih, inputdata) + b_ih)
h_output = (T.dot(w_ho, h_hidden) + b_ho)
out_softmax = nnet.softmax(h_output.T).T
cost_expression = nnet.categorical_crossentropy(out_softmax.T, target.T).sum()


accuracy_train = accuracy_calc(T.argmax(out_softmax, axis=0), T.argmax(target, axis=0))

# Backward pass
deriv_cost_w_ho = T.grad(cost_expression, w_ho)/batchSize
deriv_cost_w_ih = T.grad(cost_expression, w_ih)/batchSize
deriv_cost_b_ho = T.grad(cost_expression, b_ho)/batchSize
deriv_cost_b_ih = T.grad(cost_expression, b_ih)/batchSize

updates = [(w_ho, w_ho - learningRate*deriv_cost_w_ho), (w_ih, w_ih - learningRate*deriv_cost_w_ih), (b_ho, b_ho - learningRate*deriv_cost_b_ho), (b_ih, b_ih - learningRate*deriv_cost_b_ih)]

return_accuracy_train = theano.function([inputdata, target], accuracy_train, updates = updates)

accuracy_valid_test = accuracy_calc(T.argmax(out_softmax, axis=0), target_valid_test)

return_accuracy_valid_test = theano.function([inputdata, target_valid_test], accuracy_valid_test)
# TRAIN STEP
for i in xrange(epochs):

	x_T = x_train_set.T
 	y_T = y_train_set_onehot.T

 	aux = np.c_[x_T.reshape(len(x_T), -1), y_T.reshape(len(y_T), -1)]

 	x_Taux = aux[:, :x_T.size//len(x_T)].reshape(x_T.shape) 

	y_Taux = aux[:, x_T.size//len(y_T):].reshape(y_T.shape)

	np.random.shuffle(aux)

	x_train_set = x_Taux.T
	y_train_set_onehot = y_Taux.T

	#b = np.ones([1, x_train_set.shape[1]])
	#x_train_set = np.vstack([x_train_set, b])
 	
 	for j in xrange(numberOfBatches):
 		
 	 	input_batch = x_train_set[:, j*batchSize:min((j+1)*batchSize, x_train_set.shape[1])]
 	 	label_batch = y_train_set_onehot[:, j*batchSize: min((j+1)*batchSize, y_train_set_onehot.shape[1])]

 	 	accuracy_batch = return_accuracy_train(input_batch, label_batch) 	 	
 	 	
		print('Accuracy in epoch', i, 'for batch', j, 'was', accuracy_batch)

	# Validation
	#b = np.ones([1, x_valid_set.shape[1]])
	#x_valid_set = np.vstack([x_valid_set, b])	
	accuracy_valid = return_accuracy_valid_test(x_valid_set, y_valid_set)
	print('Accuracy for validation in epoch', i, 'was', accuracy_valid)

# Testing
#b = np.ones([1, x_test_set.shape[1]])
#x_test_set = np.vstack([x_test_set, b])
accuracy_test = return_accuracy_valid_test(x_test_set, y_test_set)
print('Accuracy for the testing was', accuracy_test)		


