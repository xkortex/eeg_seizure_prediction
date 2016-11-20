import numpy
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
# fix random seed for reproducibility
numpy.random.seed(7)


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
    in_dims = numpy.prod(ds.shape[1:])
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
