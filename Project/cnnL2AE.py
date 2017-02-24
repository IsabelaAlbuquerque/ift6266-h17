import numpy as np
np.random.seed(42)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D, UpSampling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from build_minibatch import build_minibatch

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
nb_deconv1 = 5
nb_deconv2 = 5
nb_deconv3 = 5

# Stride
str_conv1 = 3

input_img = input(shape=(3,64,64)(x)

x = Convolution2D(nb_filters1, nb_conv1, nb_conv1, activation='relu', border_mode='same', subsample=(str_conv1, str_conv1) )(input_img)
x = Convolution2D(nb_filters2, nb_conv2, nb_conv2, activation='relu', border_mode='same')(x)
x = Convolution2D(nb_filters3, nb_conv3, nb_conv3, activation='relu', border_mode='same')(x)
x = Convolution2D(nb_filters4, nb_conv4, nb_conv4, activation='relu', border_mode='same')(x)
x = Convolution2D(nb_filters5, nb_conv5, nb_conv5, activation='relu', border_mode='same')(x)
x = Convolution2D(nb_filters6, nb_conv6, nb_conv6, activation='relu', border_mode='same')(x)
encoded = Dense(2000, activation='relu')(x)


x = Deconvolution2d(nb_dfilters1, nb_dconv1, nb_dconv1, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2,2) )(x)
x = Deconvolution2d(nb_dfilters2, nb_dconv2, nb_dconv2, activation='relu', border_mode='same')(x)
x = UpSampling2D((2,2) )(x)
x = Deconvolution2d(nb_dfilters3, nb_dconv3, nb_dconv3, activation='relu', border_mode='same')(x)
decoded = UpSampling2D((2,2) )(x)

autoencoder = model(input_img, decoded)

autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

autoencoder.fit(inputs, targets, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(inputs_valid, targets_valid), callbacks=[early_stopping])

score = model.evaluate(inputs_valid, targets_valid, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
