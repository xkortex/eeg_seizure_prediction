import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, BatchNormalization, Dropout, GaussianNoise, GaussianDropout, Activation
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras import regularizers

import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
# fix random seed for reproducibility
np.random.seed(7)


def make_convo_lstm(n_input_len=256):
    model = Sequential()
    model.add(Convolution1D(nb_filter=64, filter_length=8, input_dim=1,
                            input_length=n_input_len, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(LSTM(100))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def autoencoder2(ds, compression_factor=20):
    # compression_factor=20
    print('DS shape: {}'.format(ds.shape))
    in_dims = np.prod(ds.shape[1:])
    encoding_dim = int(in_dims // compression_factor)
    in_shape = ds[0].shape
    print('Input Dims: {}, input shape: {}, encoding dims: {}'.format(in_dims, in_shape, encoding_dim))

    # this is our input placeholder
    input_img = Input(shape=(in_dims,))

    #     encoded = Dense(encoding_dim*4, activation='relu', activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
    encoded = Dense(encoding_dim * 4, activation='sigmoid')(input_img)
    encoded = Dense(encoding_dim * 2, activation='sigmoid')(encoded)
    encoded = Dense(encoding_dim, activation='sigmoid')(encoded)

    # DECODED LAYER
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(encoding_dim * 2, activation='sigmoid')(encoded)
    decoded = Dense(encoding_dim * 4, activation='sigmoid')(decoded)
    decoded = Dense(in_dims, activation='sigmoid')(decoded)

    # MODEL
    autoencoder = Model(input=input_img, output=decoded)

    # SEPERATE ENCODER MODEL
    encoder = Model(input=input_img, output=encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))

    # retrieve the last layer of the autoencoder model
    decoder_layer1 = autoencoder.layers[-3]
    decoder_layer2 = autoencoder.layers[-2]
    decoder_layer3 = autoencoder.layers[-1]

    # create the decoder model - unrolling the model as we go
    decoder = Model(input=encoded_input, output=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

    #     model.add(GaussianNoise(0.1), input_shape=(n_input_len,))
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.model_name = 'Autoencoder 1'
    return autoencoder, encoder, decoder


def autoencoder3(ds, compression_factor=16, input_noise=0.2, dropout_p=0.1, activ='tanh', final_activ='tanh'):
    """
    This one works really well!
    :param ds: Data set, just used to get dimension of input (need to refactor this)
    :param compression_factor: Compression ratio
    :param input_noise: Gaussian sigma to apply to input vector
    :param dropout_p: Dropout rate in the 3 input dropouts and one output dropout
    :param activ: activation function used in all but the last layer
    :param final_activ: activation function of the last layer
    :return:
    """
    # compression_factor=20
    print('DS shape: {}'.format(ds.shape))
    in_dims = np.prod(ds.shape[1:])
    encoding_dim = int(in_dims // compression_factor)
    in_shape = ds[0].shape
    print('Input Dims: {}, input shape: {}, encoding dims: {}'.format(in_dims, in_shape, encoding_dim))

    # this is our input placeholder
    input_img = Input(shape=(in_dims,))
    encoded = GaussianNoise(input_noise)(input_img)

    encoded = Dense(encoding_dim * 4, activation=activ, activity_regularizer=regularizers.activity_l1(10e-5))(encoded)
    #     encoded = Dense(encoding_dim*4, activation='sigmoid')(input_img)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(dropout_p)(encoded)  # batch norm before dropout
    #     encoded = Dense(encoding_dim*3, activation=activ)(encoded)
    #     encoded = Dropout(dropout_p)(encoded)
    encoded = Dense(encoding_dim * 2, activation=activ)(encoded)
    encoded = Dropout(dropout_p)(encoded)

    encoded = Dense(encoding_dim, activation=activ)(encoded)
    # Middle Noise
    encoded = GaussianNoise(0.02)(encoded)

    # DECODED LAYER
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(encoding_dim * 2, activation=activ)(encoded)
    #     decoded = Dropout(dropout_p)(decoded)
    decoded = Dense(encoding_dim * 4, activation=activ)(decoded)
    decoded = Dropout(dropout_p)(decoded)
    decoded = Dense(in_dims, activation=final_activ)(decoded)

    # MODEL
    autoencoder = Model(input=input_img, output=decoded)

    # SEPERATE ENCODER MODEL
    encoder = Model(input=input_img, output=encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))

    # retrieve the last layer of the autoencoder model
    decoder_layer0 = autoencoder.layers[-4]
    decoder_layer1 = autoencoder.layers[-3]
    decoder_layer2 = autoencoder.layers[-2]
    decoder_layer3 = autoencoder.layers[-1]
    # todo: make this into a dedicated unrolling function

    # create the decoder model - unrolling the model as we go
    decoder = Model(input=encoded_input, output=decoder_layer3(decoder_layer2(
        decoder_layer1(decoder_layer0(encoded_input)))))

    #     model.add(GaussianNoise(0.1), input_shape=(n_input_len,))
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.model_name = 'Autoencoder 1'
    return autoencoder, encoder, decoder


class SimpleAutoClasser:
    """
    This one works really well!
    :param ds: Data set, just used to get dimension of input (need to refactor this)
    :param compression_factor: Compression ratio
    :param input_noise: Gaussian sigma to apply to input vector
    :param dropout_p: Dropout rate in the 3 input dropouts and one output dropout
    :param activ: activation function used in all but the last layer
    :param final_activ: activation function of the last layer
    :return:
    """
    def __init__(self, ds_shape, latent_dim=2, input_noise=0.2, dropout_p=0.5, activ='tanh', final_activ='tanh',
                      nb_classes=2, batch_size=100, compression_factor=None):
        # compression_factor=20
        print('DS shape: {}'.format(ds_shape))
        in_dims = np.prod(ds_shape[1:])
        # latent_dim = int(in_dims // compression_factor)
        in_shape = ds_shape[1:]
        sizes = [4*int(in_dims**0.5), 4*int(in_dims**0.25)]
        # this is our input placeholder
        batch_shape = (batch_size,) + in_shape
        # input_img = Input(batch_shape=batch_shape, name='main_input')
        print('Batch Shape: ', batch_shape )
        x_in = Input(shape=(in_dims,))
        # encoded = Dense(in_dims, activation='linear')(x_in)
        print('Input Dims: {}, input shape: {}, encoding dims: {}'.format(in_dims, in_shape, latent_dim))
        print('Sizes', sizes)

        encoded = Dense(sizes[0], activation=activ, activity_regularizer=regularizers.activity_l1(10e-5))(x_in)
        encoded = GaussianNoise(input_noise)(encoded)

        #     encoded = Dense(encoding_dim*4, activation='sigmoid')(input_img)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(dropout_p)(encoded)  # batch norm before dropout
        #     encoded = Dense(encoding_dim*3, activation=activ)(encoded)
        #     encoded = Dropout(dropout_p)(encoded)
        encoded = Dense(sizes[1], activation=activ)(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(dropout_p)(encoded)

        latent = Dense(latent_dim, activation=activ)(encoded)
        # Middle Noise
        encoded = GaussianNoise(0.02)(encoded)

        # DECODED LAYER
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(sizes[1], activation=activ)(latent)
        #     decoded = Dropout(dropout_p)(decoded)
        decoded = Dense(sizes[0], activation=activ)(decoded)
        decoded = Dropout(dropout_p/2)(decoded)
        decoded = Dense(in_dims, activation=final_activ)(decoded)

        # MODELs
        self.autoencoder = Model(input=x_in, output=decoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(latent_dim,))

        # retrieve the last layer of the autoencoder model
        decoder_layer0 = self.autoencoder.layers[-4]
        decoder_layer1 = self.autoencoder.layers[-3]
        decoder_layer2 = self.autoencoder.layers[-2]
        decoder_layer3 = self.autoencoder.layers[-1]
        # todo: make this into a dedicated unrolling function

        class_out = Dense(nb_classes, activation='sigmoid')(latent)


        # Moar Models
        self.encoder = Model(input=x_in, output=latent)

        # create the decoder model - unrolling the model as we go
        self.decoder = Model(input=encoded_input, output=decoder_layer3(decoder_layer2(
            decoder_layer1(decoder_layer0(encoded_input)))))

        self.classer = Model(input=x_in, output=class_out)
        #     model.add(GaussianNoise(0.1), input_shape=(n_input_len,))
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.classer.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.autoencoder.model_name = 'Autoencoder 1'

    def get_models(self):
        return self.autoencoder, self.encoder, self.decoder, self.classer


def Simple_Convo(train, nb_classes):
    batch_size = 128
    img_rows, img_cols = 56, 56

    nb_filters_1 = 32  # 64
    nb_filters_2 = 64  # 128
    nb_filters_3 = 128  # 256
    nb_conv = 3

    # train = np.concatenate([train, train], axis=1)
    trainX = train[:, 1:].reshape(train.shape[0], 28, 28, 1)
    trainX = trainX.astype(float)

    trainX /= 255.0
    trainX = np.concatenate([trainX, np.roll(trainX, 14, axis=1)], axis=1)
    trainX = np.concatenate([trainX, np.fliplr(np.roll(trainX, 7, axis=2))], axis=2)
    print(trainX.shape)

    cnn = models.Sequential()

    cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", input_shape=(img_rows, img_cols, 1),
                               border_mode='same'))
    cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.MaxPooling2D(strides=(2, 2)))

    cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.MaxPooling2D(strides=(2, 2)))

    # cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
    # cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
    # cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
    # cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
    # cnn.add(conv.MaxPooling2D(strides=(2,2)))

    cnn.add(core.Flatten())
    cnn.add(core.Dropout(0.2))
    cnn.add(core.Dense(128, activation="relu"))  # 4096
    cnn.add(core.Dense(nb_classes, activation="softmax"))

    cnn.summary()
    return cnn

class Simple_Convo_Classer(object):
    def __init__(self, img_size, nb_classes):
        batch_size = 128
        img_rows, img_cols = img_size

        nb_filters_1 = 32  # 64
        nb_filters_2 = 64  # 128
        nb_filters_3 = 128  # 256
        nb_conv = 3


        cnn = models.Sequential()

        cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", input_shape=(img_rows, img_cols, 1),
                                   border_mode='same'))
        cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", border_mode='same'))
        cnn.add(conv.MaxPooling2D(strides=(2, 2)))

        cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
        cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
        cnn.add(conv.MaxPooling2D(strides=(2, 2)))

        # cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
        # cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
        # cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
        # cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
        # cnn.add(conv.MaxPooling2D(strides=(2,2)))

        cnn.add(core.Flatten())
        cnn.add(core.Dropout(0.2))
        cnn.add(core.Dense(128, activation="relu"))  # 4096
        cnn.add(core.Dense(nb_classes, activation="softmax"))

        cnn.summary()
        cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.cnn = cnn

