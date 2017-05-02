import numpy as np
np.random.seed(12345) 

import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from data_flow import batch_generator_with_embeddings
import os.path


def make_trainable(model, value):
	for layer in model.layers:
		layer.trainable = value

def create_discriminator_mini_batch(true_mini_batch_center, generated_mini_batch_center):
	mini_batch_size = true_mini_batch_center.shape[0]
	input_mini_batch = np.vstack([true_mini_batch_center, generated_mini_batch_center])
	target_mini_batch = np.zeros([2*mini_batch_size, 2])

	target_mini_batch[0:mini_batch_size, 0] = np.random.uniform(0.8, 0.99, (mini_batch_size)) #true image: target = (1,0)
	target_mini_batch[0:mini_batch_size, 1] = 1.0 - target_mini_batch[0:mini_batch_size, 0]  
	target_mini_batch[mini_batch_size:, 0] = np.random.uniform(0.01, 0.2, (mini_batch_size)) #fake image: target = (0,1)
	target_mini_batch[mini_batch_size:, 1] = 1.0 - target_mini_batch[mini_batch_size:, 0]
		
	return input_mini_batch, target_mini_batch		


# Creating the models

# Merging original context to generated/true center
context = layers.Input(shape=(64, 64, 3))
center = layers.Input(shape=(32, 32, 3))
center_padded = ZeroPadding2D(padding=(16, 16)) (center)
merged = layers.add([center_padded, context])
merger = models.Model([center, context], merged)

# Horizontal concatenation of encoded context+noise and caption embedding
encoded_input = layers.Input(shape=(4, 4, 128))
embedding_input = layers.Input(shape=(300, ))
embedding_input_dense = Dense(4*4*128)(embedding_input)
embedding_reshaped = Reshape(target_shape=(4, 4, 128))(embedding_input_dense)
concat = layers.concatenate([encoded_input, embedding_reshaped])
concatenated = models.Model([encoded_input, embedding_input], concat)

# Loading pre-trained encoder
enc = load_model('encoder.h5')
make_trainable(enc, False)	


# Generator model
gen = Sequential()

gen.add(concatenated)

gen.add(Conv2DTranspose(512, 5, padding='same', strides=(2,2), input_shape=(4, 4, 256)))
gen.add(BatchNormalization())
gen.add(Activation('relu'))
gen.add(GaussianNoise(0.01))

gen.add(Conv2D(256, 5, padding='same', strides=(1,1)))
gen.add(BatchNormalization())
gen.add(Activation('relu'))

gen.add(Conv2DTranspose(256, 5, padding='same', strides=(2,2))) 
gen.add(BatchNormalization())
gen.add(Activation('relu'))
gen.add(GaussianNoise(0.01))

gen.add(Conv2D(128, 5, padding='same', strides=(1,1)))
gen.add(BatchNormalization())
gen.add(Activation('relu'))

gen.add(Conv2DTranspose(128, 5, padding='same', strides=(2,2))) 
gen.add(BatchNormalization())
gen.add(Activation('relu'))
gen.add(GaussianNoise(0.01))

gen.add(Conv2DTranspose(3, 5, padding='same', strides=(1,1)))
gen.add(Activation('sigmoid'))



# Discriminator model
disc = Sequential()

disc.add(merger)

disc.add(Conv2D(64, 5, padding='same', strides=(2, 2), input_shape=(64, 64, 3)))
disc.add(Dropout(0.5))
disc.add(BatchNormalization())
disc.add(LeakyReLU(alpha=0.2))

disc.add(Conv2D(128, 5, padding='same', strides=(2, 2)))
disc.add(Dropout(0.5))
disc.add(LeakyReLU(alpha=0.2))

disc.add(Conv2D(256, 5, padding='same', strides=(2, 2)))
disc.add(Dropout(0.5))
disc.add(BatchNormalization())
disc.add(LeakyReLU(alpha=0.2))

disc.add(Conv2D(512, 5, padding='same', strides=(2, 2)))
disc.add(Dropout(0.5))
disc.add(LeakyReLU(alpha=0.2))

disc.add(Conv2D(512, 5, padding='same'))
disc.add(Dropout(0.5))
disc.add(BatchNormalization())
disc.add(LeakyReLU(alpha=0.2))

disc.add(Flatten())
disc.add(Dense(512))
disc.add(LeakyReLU(alpha=0.2))
disc.add(Dropout(0.5))
disc.add(Dense(2))
disc.add(Activation('softmax'))

# GAN

input_to_encode = layers.Input(shape=(64, 64, 3))
true_input = layers.Input(shape=(64, 64, 3))
caption_embedding = layers.Input(shape=(300,))
encoded_input = enc(input_to_encode)
generated = gen([encoded_input, caption_embedding])

make_trainable(disc, False)
disc_output = disc([generated, true_input])

GAN = models.Model([input_to_encode, true_input, caption_embedding], disc_output)

# Compiling models

gen.compile(loss='mean_absolute_error', optimizer='rmsprop')
GAN.compile(loss='categorical_crossentropy', optimizer='rmsprop')
make_trainable(disc, True)
disc.compile(loss='categorical_crossentropy', optimizer='sgd')


#Training loop

chunk_size = 10000
mini_batch_size = 64 
true_train_size = 82611

epochs = 100
chunks = int(np.ceil(true_train_size/chunk_size))

for i in xrange(0, epochs):

	data_generator = batch_generator_with_embeddings(hdf5_name='train_data.hdf', hdf5_name_captions_emb='embeddings_train_norm.hdf', data_size=true_train_size, load_captions_emb=True, batch_size=chunk_size)

	for k in range(chunks):

		context_batch, centers_batch, captions_batch = next(data_generator)

		current_chunk_size = context_batch.shape[0]
		number_of_mini_batches = int(np.ceil(current_chunk_size/mini_batch_size))

		for j in range(number_of_mini_batches):

			context_mini_batch = context_batch[j*mini_batch_size:min((j+1)*mini_batch_size, current_chunk_size)]
			true_center_mini_batch = centers_batch[j*mini_batch_size:min((j+1)*mini_batch_size, current_chunk_size)]
			captions_mini_batch = captions_batch[j*mini_batch_size:min((j+1)*mini_batch_size, current_chunk_size)]

			current_mini_batch_size = context_mini_batch.shape[0]

			# Training the discriminator
			noise_mini_batch = np.random.normal(0., 0.2, (current_mini_batch_size, 4, 4, 128))
			encoded_mini_batch = enc.predict(context_mini_batch)
			gen_input_mini_batch = encoded_mini_batch + noise_mini_batch
			gen_center_mini_batch = gen.predict([gen_input_mini_batch, captions_mini_batch])
			disc_input_mini_batch, disc_center_mini_batch = create_discriminator_mini_batch(true_center_mini_batch, gen_center_mini_batch)
			make_trainable(disc, True)
			disc.train_on_batch([disc_input_mini_batch, np.vstack([context_mini_batch, context_mini_batch])], disc_center_mini_batch)
			make_trainable(disc, False)

			#Training the generator with L1 loss
			gen.train_on_batch([context_mini_batch, captions_mini_batch], true_center_mini_batch)

			#Training the generator with adversarial loss
			GAN_target_mini_batch = np.zeros([actual_mini_batch_size,2])
			GAN_target_mini_batch[:,0]=1.0
			GAN.train_on_batch([context_mini_batch, context_mini_batch, captions_mini_batch], GAN_target_mini_batch)



# Saving the models
gen.save('gen_model.h5')
disc.save('disc_model.h5')
GAN.save('GAN_model.h5')
