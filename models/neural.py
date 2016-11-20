import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, BatchNormalization, Dropout, GaussianNoise, GaussianDropout
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
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


