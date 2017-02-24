import numpy as np
np.random.seed(42)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from build_minibatch import build_minibatch

batch_size = 128
nb_outputs = 32*32*3
nb_epoch = 10

# Preparing the training data
inputs, targets, captions = build_minibatch(0, 128, 1, False)


# Preparing the training data
inputs_valid, targets_valid, captions_valid = build_minibatch(0, 128, 0, False)
		
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

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.fit(inputs, targets, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(inputs_valid, targets_valid), callbacks=[early_stopping])

score = model.evaluate(inputs_valid, targets_valid, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
