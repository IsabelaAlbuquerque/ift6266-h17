import numpy as np
np.random.seed(42)  # for reproducibility

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D, UpSampling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os.path
from data_flow import batch_generator

batch_size = 100
nb_epoch = 50

samples_per_epoch = 82611
nb_val_samples = 40438

# input image dimensions
img_rows, img_cols = 64, 64

# number of convolutional filters to use
nb_filters1 = 32
nb_filters2 = 32
nb_filters3 = 32
nb_filters4 = 64
nb_filters5 = 64
nb_filters6 = 64

# number of deconvolutional filters to use
nb_dfilters1 = 32
nb_dfilters2 = 16
nb_dfilters3 = 3

# convolution kernel size
nb_conv1 = 4
nb_conv2 = 4
nb_conv3 = 4
nb_conv4 = 4
nb_conv5 = 4
nb_conv6 = 4

# deconvolution kernel size
nb_dconv1 = 3
nb_dconv2 = 3
nb_dconv3 = 5

# Stride
str_conv1 = 3

runMax = 300
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

	input_img = Input(shape=(img_rows,img_cols,3))

	x = Convolution2D(nb_filters1, nb_conv1, nb_conv1, activation='relu', border_mode='valid', subsample=(str_conv1, str_conv1) )(input_img)
	x = Convolution2D(nb_filters2, nb_conv2, nb_conv2, activation='relu', border_mode='valid')(x)
	x = Convolution2D(nb_filters3, nb_conv3, nb_conv3, activation='relu', border_mode='valid')(x)
	x = Convolution2D(nb_filters4, nb_conv4, nb_conv4, activation='relu', border_mode='valid')(x)
	x = Convolution2D(nb_filters5, nb_conv5, nb_conv5, activation='relu', border_mode='valid')(x)
	x = Convolution2D(nb_filters6, nb_conv6, nb_conv6, activation='relu', border_mode='valid')(x)
	x = Flatten()(x)
	encoded = Dense(2304, activation='relu')(x)
	
	x=Reshape([6,6,64])(encoded)
	x = UpSampling2D((2,2) )(x)
	x = Convolution2D(nb_dfilters1, nb_dconv1, nb_dconv1, activation='relu', border_mode='valid')(x)
	x = UpSampling2D((2,2) )(x)
	x = Convolution2D(nb_dfilters2, nb_dconv2, nb_dconv2, activation='relu', border_mode='valid')(x)
	x = UpSampling2D((2,2) )(x)
	decoded = Convolution2D(nb_dfilters3, nb_dconv3, nb_dconv3, activation='sigmoid', border_mode='valid')(x)
	autoencoder = Model(input_img, decoded)
	autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

else:

	autoencoder = load_model(CPName)
	nb_epoch-=epoch
	nb_epoch=max(nb_epoch,1)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
checkPointer = ModelCheckpoint(filepath='CP/cnnL2AE/cnnL2AE-'+str(run)+'-{epoch:02d}.h5', verbose=1, save_best_only=False, save_weights_only=False)

generatorTrain = batch_generator(1,'train_data.hdf',batch_size)
generatorValid = batch_generator(0,'valid_data.hdf',batch_size)

history=autoencoder.fit_generator(generatorTrain, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, verbose=1, validation_data=generatorValid, nb_val_samples=nb_val_samples, callbacks=[early_stopping, checkPointer])

autoencoder.save('cnnL2AE.h5')
