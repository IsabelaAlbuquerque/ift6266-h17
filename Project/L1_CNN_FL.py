import numpy as np
np.random.seed(42)  # for reproducibility

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os.path
from data_flow import batch_generator

batch_size = 100
nb_outputs = 32*32*3
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

# convolution kernel size
nb_conv1 = 4
nb_conv2 = 4
nb_conv3 = 4
nb_conv4 = 4
nb_conv5 = 4
nb_conv6 = 4

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

		if os.path.exists('CP/cnnL1/cnnL1-'+runStr+'-'+epochStr+'.h5'):	
			CPName='CP/cnnL1/cnnL1-'+runStr+'-'+epochStr+'.h5'
			epoch=j+1
			run=i+1
			break

if CPName is None:
	
	model = Sequential()

	model.add(Convolution2D(nb_filters1, nb_conv1, nb_conv1, border_mode='valid', subsample=(str_conv1, str_conv1), input_shape=(img_rows, img_cols,3)))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters2, nb_conv2, nb_conv2))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters3, nb_conv3, nb_conv3))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters4, nb_conv4, nb_conv4))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters5, nb_conv5, nb_conv5))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters6, nb_conv6, nb_conv6))
	model.add(Activation('relu'))

	model.add(Flatten())
	model.add(Dense(4800))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_outputs))
	model.add(Activation('softmax'))
	model.add(Reshape((32,32,3)))

	model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

else:
	model = load_model(CPName)
	nb_epoch-=epoch
	nb_epoch=max(nb_epoch,1)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
checkPointer = ModelCheckpoint(filepath='CP/cnnL1/cnnL1-'+str(run)+'-{epoch:02d}.h5', verbose=1, save_best_only=False, save_weights_only=False)

generatorTrain = batch_generator(1,'train_data.hdf',batch_size)
generatorValid = batch_generator(0,'valid_data.hdf',batch_size)

history=model.fit_generator(generatorTrain, samples_per_epoch=samples_per_epoch , nb_epoch=nb_epoch, verbose=0, validation_data=generatorValid, nb_val_samples=nb_val_samples,callbacks=[early_stopping, checkPointer])

model.save('cnnL1.h5')
