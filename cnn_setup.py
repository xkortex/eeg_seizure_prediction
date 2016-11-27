import numpy as np
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint

nb_epoch = 500

model = Sequential()

model.add(Convolution1D(128, 3, border_mode='same', input_dim=6, input_length=256))
model.add(Convolution1D(128, 3, border_mode='same'))
model.add(MaxPooling1D(pool_length=2))

model.add(Convolution1D(256, 3, border_mode='same'))
model.add(Convolution1D(256, 3, border_mode='same'))
model.add(MaxPooling1D(pool_length=2))

model.add(Convolution1D(512, 3, border_mode='same'))
model.add(Convolution1D(512, 3, border_mode='same'))
model.add(MaxPooling1D(pool_length=2))

model.add(Flatten())

model.add(Dense(64, init='he_normal', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, init='he_normal', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2, init='he_normal', activation = 'softmax'))
print model.summary()