import numpy as np
np.random.seed(42)  # for reproducibility

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from build_minibatch import build_minibatch
from tqdm import tqdm # O exemplo do Keras usa essa funcao so para mostrar o progresso do treino
import os.path

def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable=val


batch_size = 500
nb_epoch = 500
k_hyperparam = 10

# input image dimensions
img_rows, img_cols = 64, 64

# Preparing the training data
inputs, targets, captions = build_minibatch(0, None, 1, True)


# Preparing the training data
inputs_valid, targets_valid, captions_valid = build_minibatch(0, None, 0, True)


# number of convolutional filters to use on the generator
nb_filters1 = 32
nb_filters2 = 32
nb_filters3 = 32
nb_filters4 = 64
nb_filters5 = 64
nb_filters6 = 64

# number of deconvolutional filters to use on the generator
nb_dfilters1 = 32
nb_dfilters2 = 16
nb_dfilters3 = 3

# convolution kernel sizes on the generator
nb_conv1 = 4
nb_conv2 = 4
nb_conv3 = 4
nb_conv4 = 4
nb_conv5 = 4
nb_conv6 = 4

# deconvolution kernel sizes on the generator
nb_dconv1 = 3
nb_dconv2 = 3
nb_dconv3 = 5

# Strides on the generator
str_conv1 = 3

# number of convolutional filters to use on the discriminator
nb_discFilters1 = 64
nb_discFilters2 = 128
nb_discFilters3 = 256

# convolution kernel sizes on the discriminator
nb_discConv1 = 4
nb_discConv2 = 4
nb_discConv3 = 4

# Strides on the discriminator
str_discConv1 = 2
str_discConv2 = 2
str_discConv3 = 2

runMax = 300
epoch=0
runG=0
runG=D
CPNameG = None
CPNameD = None

for i in range(runMax,-1,-1):
	for j in range(nb_epoch,-1,-1):	
		epochStr = str(j)
		runStr = str(i)

		if len(epochStr)==1:
			epochStr=str(0)+epochStr

		if os.path.exists('CP/cnnL2AE/cnnL2AE-'+runStr+'-'+epochStr+'.h5'):	
			CPName='CP/cnnL2AE/cnnL2AE-'+runStr+'-'+epochStr+'.h5'
			epoch=j+1
			run=i+1
			break

if CPName is None:

	input_img = Input(shape=(64,64,3))

	g = Convolution2D(nb_filters1, nb_conv1, nb_conv1, activation='relu', border_mode='valid', subsample=(str_conv1, str_conv1) )(input_img)
	g = Convolution2D(nb_filters2, nb_conv2, nb_conv2, activation='relu', border_mode='valid')(g)
	g = Convolution2D(nb_filters3, nb_conv3, nb_conv3, activation='relu', border_mode='valid')(g)
	g = Convolution2D(nb_filters4, nb_conv4, nb_conv4, activation='relu', border_mode='valid')(g)
	g = Convolution2D(nb_filters5, nb_conv5, nb_conv5, activation='relu', border_mode='valid')(g)
	g = Convolution2D(nb_filters6, nb_conv6, nb_conv6, activation='relu', border_mode='valid')(g)
	g = Flatten()(g)
	encoded = Dense(2304, activation='relu')(g)
	
	g=Reshape([6,6,64])(encoded)
	g = UpSampling2D((2,2) )(g)
	g = Convolution2D(nb_dfilters1, nb_dconv1, nb_dconv1, activation='relu', border_mode='valid')(g)
	g = UpSampling2D((2,2) )(g)
	g = Convolution2D(nb_dfilters2, nb_dconv2, nb_dconv2, activation='relu', border_mode='valid')(g)
	g = UpSampling2D((2,2) )(g)
	decoded = Convolution2D(nb_dfilters3, nb_dconv3, nb_dconv3, activation='sigmoid', border_mode='valid')(g)
	generator = Model(input_img, decoded)
	generator.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

else:

	autoencoder = load_model(CPName)
	nb_epoch-=epoch
	nb_epoch=max(nb_epoch,1)

d_input = Input(shape=(64,64,3))
d = ZeroPadding2D(padding=(1,1))(d_input)
d = Convolution2D(nb_discFilters1, nb_discConv1, nb_discConv1, activation='relu', border_mode='valid', subsample=(str_discConv1, str_discConv1) )(d)
d = ZeroPadding2D(padding=(1,1))(d)
d = Convolution2D(nb_discFilters2, nb_discConv2, nb_discConv2, activation='relu', border_mode='valid', subsample=(str_discConv2, str_discConv2) )(d)
d = ZeroPadding2D(padding=(1,1))(d)
d = Convolution2D(nb_discFilters3, nb_discConv3, nb_discConv3, activation='relu', border_mode='valid', subsample=(str_discConv3, str_discConv3) )(d)
d = Flatten()(d)
d = Dense(1500, activation='relu')(d)
d = Dense(600, activation='relu')(d)
d_v = d = Dense(2, activation='softmax')(d)
discriminator = model(d_input, d_v)
discriminator.compile(loss='categorical_crossentropy', optimizer='adam')

#build GAN

gan_input = Input(shape=(64,64,3))
H = generator(gan_input)
gan_v = discriminator(H)
GAN = model(gan_input, gan_v)
GAN.compile(loss='categorical_crossentropy', optimizer='adam')


# Training loop
losses = {"d":[], "g":[]} #loss storage vector

for i in tqdm(range(nb_epoch)):
	# Training the discriminator k_hyperparam times
	for k in range(k_hyperparam)

		# 1st step: create the "training set" (true centers + generated centers)
		random_idx = np.random.randint(0, inputs.shape[0], size=batch_size)
		true_centers_batch = targets[random_idx,:,:,:]
		inputs_to_generate = inputs[random_idx,:,:,:]
		generated_centers_batch = generator.predict(inputs_to_generate) 
		inputs_discriminator = np.concatenate(true_centers_batch, generated_centers_batch)
		targets_discriminator = np.zeros([2*batch_size, 2])	#NAO ENTENDI!!!!!
		targets_discriminator[0:batch_size, 1] = 1
		targets_discriminator[batch_size:, 0] = 1
		
		# 2nd step: training the discriminator
		d_loss = discriminator.train_on_batch(inputs_discriminator, targets_discriminator)
		losses["d"].append(d_loss)

	# Training JUST the generative model
	random_idx = np.random.randint(0, inputs.shape[0], size=batch_size)
	inputs_batch = inputs[random_idx, :, :, :]
	targets_batch = targets[random_idx, :, :, :]

	make_trainable(discriminator, False)
	g_loss = GAN.train_on_batch(inputs_batch, targets_batch)
	losses["g"].append(g_loss)



early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkPointer = ModelCheckpoint(filepath='CP/cnnL2AE/cnnL2AE-'+str(run)+'-{epoch:02d}.h5', verbose=1, save_best_only=False, save_weights_only=False)

history=autoencoder.fit(inputs, targets, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(inputs_valid, targets_valid), callbacks=[early_stopping, checkPointer])

score = autoencoder.evaluate(inputs_valid, targets_valid, verbose=0)

print('Validation Loss:', history.history['val_loss'][-1])

autoencoder.save('cnnL2AE.h5')
