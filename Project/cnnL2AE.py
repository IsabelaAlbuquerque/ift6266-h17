import numpy as np
np.random.seed(42)  # for reproducibility

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D, UpSampling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from build_minibatch import build_minibatch
import os.path

batch_size = 128
nb_outputs = 32*32*3
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 64, 64

# Preparing the training data
inputs, targets, captions = build_minibatch(0, 128, 1, False)

# Preparing the training data
inputs_valid, targets_valid, captions_valid = build_minibatch(0, 128, 0, False)


# number of convolutional filters to use
nb_filters1 = 32
nb_filters2 = 32
nb_filters3 = 32
nb_filters4 = 64
nb_filters5 = 64
nb_filters6 = 64

# number of deconvolutional filters to use
nb_dfilters1 = 32
nb_dfilters2 = 32
nb_dfilters3 = 3

# convolution kernel size
nb_conv1 = 4
nb_conv2 = 4
nb_conv3 = 4
nb_conv4 = 4
nb_conv5 = 4
nb_conv6 = 4

# deconvolution kernel size
nb_dconv1 = 5
nb_dconv2 = 5
nb_dconv3 = 5

# Stride
str_conv1 = 3

runMax = 50
epoch=0
run=0
CPName = None

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

	x = Convolution2D(nb_filters1, nb_conv1, nb_conv1, activation='relu', border_mode='same', subsample=(str_conv1, str_conv1) )(input_img)
	x = Convolution2D(nb_filters2, nb_conv2, nb_conv2, activation='relu', border_mode='same')(x)
	x = Convolution2D(nb_filters3, nb_conv3, nb_conv3, activation='relu', border_mode='same')(x)
	x = Convolution2D(nb_filters4, nb_conv4, nb_conv4, activation='relu', border_mode='same')(x)
	x = Convolution2D(nb_filters5, nb_conv5, nb_conv5, activation='relu', border_mode='same')(x)
	x = Convolution2D(nb_filters6, nb_conv6, nb_conv6, activation='relu', border_mode='same')(x)
	x = Flatten()(x)
	encoded = Dense(2000, activation='relu')(x)
	
	x=Reshape([80,5,5])(encoded)
	x = Convolution2D(nb_dfilters1, nb_dconv1, nb_dconv1, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2,2) )(x)
	x = Convolution2D(nb_dfilters2, nb_dconv2, nb_dconv2, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2,2) )(x)
	x = Convolution2D(nb_dfilters3, nb_dconv3, nb_dconv3, activation='relu', border_mode='same')(x)
	decoded = UpSampling2D((2,2) )(x)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

else:

	autoencoder = load_model(CPName)
	nb_epoch-=epoch
	nb_epoch=max(nb_epoch,1)

early_stopping = EarlyStopping(monitor='val_loss', patience=35)
checkPointer = ModelCheckpoint(filepath='CP/cnnL2AE/cnnL2AE-'+str(run)+'-{epoch:02d}.h5', verbose=1, save_best_only=False, save_weights_only=False)

history=autoencoder.fit(inputs, targets, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(inputs_valid, targets_valid), callbacks=[early_stopping, checkPointer])

score = autoencoder.evaluate(inputs_valid, targets_valid, verbose=0)

print('Validation Loss:', history.history['val_loss'][-1])

model.save('cnnL2AE.h5')

autoencoder.save('cnnL2AE.h5')
