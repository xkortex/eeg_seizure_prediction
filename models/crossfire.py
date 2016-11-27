
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, BatchNormalization, Dropout, GaussianNoise
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers
from keras import backend as K_backend
from keras import objectives
from keras.datasets import mnist



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



class SimpleCrossfire(object):
    """
    Covolutional VAE with a "crossfire" component - a classifier is bolted onto the end of the encoder and loss function
    can be parameterized to function from classifier loss instead of autoencoder loss. Ideally, the model starts in
    pure autoencoder mode to learn features, then as loss flattens out the network starts weighing classifier loss
    more heavily.
    Note to self: things to try:
    * Add burst /batch / epoch noise to input
    * Modularize out the activation from dropout ordering"""

    def __init__(self, input_shape=(28, 28, 1), latent_dim=2, intermediate_dim=256, batch_size=100, epsilon_std=1.0,
                 dropout_p=0.1, n_stax=0, n_classes=10):
        # input image dimensions
        self.input_shape = input_shape
        if len(input_shape) == 3:
            self.img_rows, self.img_cols, self.img_chns = input_shape
        elif len(input_shape) == 2:
            self.img_rows, self.img_cols = input_shape
            self.img_chns = 1
        else:
            raise IndexError("Invalid shape: {}".format(input_shape))
        self.batch_size = batch_size
        self.original_dim = np.prod(input_shape)
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.epsilon_std = epsilon_std
        self.epsilon_ce = 1.0e-9

        # number of convolutional filters to use
        nb_filters = 64
        # convolution kernel size
        nb_conv = 3

        input_img = Input(shape=(1, input_shape[0], input_shape[1]))

        # Same as the code above, but with some params changed
        # Now let's define the model.

        # (no.of, colour, channels, height, width) = (3, 30, 30)
        # Set input dimensions:
        input_img = Input(shape=(3, 30, 30))

        # Encoder: define a chain of Conv2D and MaxPooling2D layers
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
        encoded = MaxPooling2D((2, 2), border_mode='same')(x)

        # at this point, the representation is (8, 4, 4) i.e. 128-dimensional

        # Decoder: a stack of Conv2D and UpSampling2D layers
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Convolution2D(1, 3, 3,
                                activation='sigmoid',
                                border_mode='same')(x)
        # Generate models
        # Primary model - VAE
        ## ==== Crossfire classifier =========
        #         c = Lambda(self.crosser, output_shape=(latent_dim,))(self.z_mean, self.z_log_var)
        classer = Dense(n_classes, init='normal', activation='softmax', name='classer')(encoded)
        self.model = Model(input_img, decoded)
        # Crossfilre network
        self.classifier = Model(x, classer)
        # Ok, now comes the tricky part. See these references:
        # https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
        self.crossmodel = Model(input=x, output=[decoded, classer])
        self.crossmodel.compile(optimizer='rmsprop',
                                loss={'main_output': self.vae_loss,
                                      'classer': 'categorical_crossentropy'},
                                loss_weights={'main_output': 1.0,
                                              'classer': 1.0})

        # build a model to project inputs on the latent space
        self.encoder = Model(x, encoded)
        # reconstruct digits from latent space
        # self.generator = Model(decoder_input, _x_decoded_mean_squash)
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        self.classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # self.classifier.compile(loss=self.custom_crossent, optimizer='adam', metrics=['accuracy'])

    def fit_crossmodel(self, x_dict, y_dict, batch_size=None, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
                       validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        callbacks_history = self.crossmodel.fit(x_dict, y_dict, batch_size, nb_epoch, verbose, callbacks,
                                                validation_split,
                                                validation_data, shuffle, class_weight, sample_weight)
        return callbacks_history

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K_backend.random_normal(shape=(self.batch_size, self.latent_dim),
                                          mean=0., std=self.epsilon_std)
        return z_mean + K_backend.exp(z_log_var) * epsilon

    def crosser(self, args):
        z_mean, z_log_var = args
        return z_mean

    def vae_loss(self, x, x_decoded_mean):
        # NOTE: binary_crossentropy expects a batch_size by dim
        # for x and x_decoded_mean, so we MUST flatten these!
        x = K_backend.flatten(x)
        x_decoded_mean = K_backend.flatten(x_decoded_mean)
        xent_loss = self.img_rows * self.img_cols * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K_backend.mean(
            1 + self.z_log_var - K_backend.square(self.z_mean) - K_backend.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss

    def custom_crossent(self, y_true, y_pred):
        '''Just another crossentropy'''

        #         y_pred = K_backend.clip(y_pred, self.epsilon_ce, 1.0 - self.epsilon_ce)
        #         y_pred /= K_backend.sum(y_pred, axis=-1, keepdims=True)
        cce = K_backend.categorical_crossentropy(y_pred, y_true)
        return cce

    def fit(self, x, y, batch_size=None, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        callbacks_history = self.model.fit(x, y, batch_size, nb_epoch, verbose, callbacks, validation_split,
                                           validation_data, shuffle, class_weight, sample_weight)
        return callbacks_history

    def fit_ae(self, x, batch_size=None, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
               validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        callbacks_history = self.model.fit(x, x, batch_size, nb_epoch, verbose, callbacks, validation_split,
                                           validation_data, shuffle, class_weight, sample_weight)
        return callbacks_history

    def crossfit(self, x, y, batch_size=None, nb_epoch_ae=5, nb_epoch_cf=10, verbose=1, callbacks=[],
                 validation_split=0.,
                 validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        valid_ae, valid_cf = validation_data
        for i in range(nb_epoch_ae):
            callbacks_history = self.model.fit(x, x, batch_size, 1, verbose, callbacks, validation_split,
                                               valid_ae, shuffle, class_weight, sample_weight)
        for i in range(nb_epoch_cf):
            callbacks_history = self.classifier.fit(x, y, batch_size, 1, verbose, callbacks, validation_split,
                                                    valid_cf, shuffle, class_weight, sample_weight)
        return callbacks_history


class Crossfire(object):
    """
    Covolutional VAE with a "crossfire" component - a classifier is bolted onto the end of the encoder and loss function
    can be parameterized to function from classifier loss instead of autoencoder loss. Ideally, the model starts in
    pure autoencoder mode to learn features, then as loss flattens out the network starts weighing classifier loss
    more heavily.
    Note to self: things to try:
    * Add burst /batch / epoch noise to input
    * Modularize out the activation from dropout ordering"""

    def __init__(self, input_shape=(28, 28, 1), latent_dim=2, intermediate_dim=256, batch_size=100, epsilon_std=1.0,
                 dropout_p=0.1, n_stax=0, n_classes=10):
        # input image dimensions
        self.input_shape = input_shape
        if len(input_shape) == 3:
            self.img_rows, self.img_cols, self.img_chns = input_shape
        elif len(input_shape) == 2:
            self.img_rows, self.img_cols = input_shape
            self.img_chns = 1
        else:
            raise IndexError("Invalid shape: {}".format(input_shape))
        self.batch_size = batch_size
        self.original_dim = np.prod(input_shape)
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.epsilon_std = epsilon_std
        self.epsilon_ce = 1.0e-9

        # number of convolutional filters to use
        nb_filters = 64
        # convolution kernel size
        nb_conv = 3

        if K_backend.image_dim_ordering() == 'th':
            self.original_img_size = (self.img_chns, self.img_rows, self.img_cols)
        else:
            self.original_img_size = (self.img_rows, self.img_cols, self.img_chns)

        x = Input(batch_shape=(batch_size,) + self.original_img_size, name='main_input')
        conv_1 = Convolution2D(self.img_chns, 2, 2, border_mode='same', activation='relu')(x)
        conv_2 = Convolution2D(nb_filters, 2, 2,
                               border_mode='same', activation='relu', subsample=(2, 2))(conv_1)
        conv_3 = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', activation='relu', subsample=(1, 1))(
            conv_2)
        for i in range(n_stax):
            conv_3 = BatchNormalization()(conv_3)
            conv_3 = Dropout(dropout_p)(conv_3)
            conv_3 = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', activation='relu',
                                   subsample=(1, 1))(conv_3)

        conv_3 = BatchNormalization()(conv_3)
        conv_4 = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', activation='relu', subsample=(1, 1))(
            conv_3)
        flat = Flatten()(conv_4)
        hidden = Dense(intermediate_dim, activation='relu', name='intermed')(flat)

        self.z_mean = Dense(latent_dim, name='z_mean')(hidden)
        self.z_log_var = Dense(latent_dim, name='z_log_var')(hidden)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_var])`
        z = Lambda(self.sampling, output_shape=(latent_dim,))([self.z_mean, self.z_log_var])

        ## ==== End of encoding portion ======

        ## ==== Crossfire classifier =========
        #         c = Lambda(self.crosser, output_shape=(latent_dim,))(self.z_mean, self.z_log_var)
        classer = Dense(n_classes, init='normal', activation='softmax', name='classer')(self.z_mean)

        # Slicer example
        # y = Lambda(lambda x: x[:,0,:,:], output_shape=(1,) + input_shape[2:])(x)




        # we instantiate these layers separately so as to reuse them later
        decoder_hid = Dense(intermediate_dim, activation='relu')
        decoder_upsample = Dense(nb_filters * 14 * 14, activation='relu')

        if K_backend.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, 14, 14)
        else:
            output_shape = (batch_size, 14, 14, nb_filters)

        decoder_reshape = Reshape(output_shape[1:])
        decoder_deconv_1 = Deconvolution2D(nb_filters, nb_conv, nb_conv, output_shape, border_mode='same',
                                           subsample=(1, 1), activation='relu')
        decoder_deconv_2 = Deconvolution2D(nb_filters, nb_conv, nb_conv, output_shape, border_mode='same',
                                           subsample=(1, 1), activation='relu')
        if K_backend.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, 29, 29)
        else:
            output_shape = (batch_size, 29, 29, nb_filters)
        decoder_deconv_3_upsamp = Deconvolution2D(nb_filters, 2, 2, output_shape,
                                                  border_mode='valid', subsample=(2, 2), activation='relu')
        decoder_mean_squash = Convolution2D(self.img_chns, 2, 2, border_mode='valid',
                                            activation='sigmoid', name='main_output')

        hid_decoded = decoder_hid(z)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

        # build a digit generator that can sample from the learned distribution
        # todo: (un)roll this
        decoder_input = Input(shape=(latent_dim,))
        _hid_decoded = decoder_hid(decoder_input)
        _up_decoded = decoder_upsample(_hid_decoded)
        _reshape_decoded = decoder_reshape(_up_decoded)
        _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
        _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
        _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
        _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)

        # Generate models
        # Primary model - VAE
        self.model = Model(x, x_decoded_mean_squash)
        # Crossfilre network
        self.classifier = Model(x, classer)
        # Ok, now comes the tricky part. See these references:
        # https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
        self.crossmodel = Model(input=x, output=[x_decoded_mean_squash, classer])
        self.crossmodel.compile(optimizer='rmsprop',
                                loss={'main_output': self.vae_loss,
                                      'classer': 'categorical_crossentropy'},
                                loss_weights={'main_output': 1.0,
                                              'classer': 5.0})

        # build a model to project inputs on the latent space
        self.encoder = Model(x, self.z_mean)
        # reconstruct digits from latent space
        self.generator = Model(decoder_input, _x_decoded_mean_squash)
        self.model.compile(optimizer='rmsprop', loss=self.vae_loss)
        self.classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #         self.classifier.compile(loss=self.custom_crossent, optimizer='adam', metrics=['accuracy'])


    def fit_crossmodel(self, x_dict, y_dict, batch_size=None, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
                       validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        callbacks_history = self.crossmodel.fit(x_dict, y_dict, batch_size, nb_epoch, verbose, callbacks,
                                                validation_split,
                                                validation_data, shuffle, class_weight, sample_weight)
        return callbacks_history

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K_backend.random_normal(shape=(self.batch_size, self.latent_dim),
                                          mean=0., std=self.epsilon_std)
        return z_mean + K_backend.exp(z_log_var) * epsilon

    def crosser(self, args):
        z_mean, z_log_var = args
        return z_mean

    def vae_loss(self, x, x_decoded_mean):
        # NOTE: binary_crossentropy expects a batch_size by dim
        # for x and x_decoded_mean, so we MUST flatten these!
        x = K_backend.flatten(x)
        x_decoded_mean = K_backend.flatten(x_decoded_mean)
        xent_loss = self.img_rows * self.img_cols * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K_backend.mean(
            1 + self.z_log_var - K_backend.square(self.z_mean) - K_backend.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss

    def custom_crossent(self, y_true, y_pred):
        '''Just another crossentropy'''

        #         y_pred = K_backend.clip(y_pred, self.epsilon_ce, 1.0 - self.epsilon_ce)
        #         y_pred /= K_backend.sum(y_pred, axis=-1, keepdims=True)
        cce = K_backend.categorical_crossentropy(y_pred, y_true)
        return cce

    def fit(self, x, y, batch_size=None, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        callbacks_history = self.model.fit(x, y, batch_size, nb_epoch, verbose, callbacks, validation_split,
                                           validation_data, shuffle, class_weight, sample_weight)
        return callbacks_history

    def fit_ae(self, x, batch_size=None, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
               validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        callbacks_history = self.model.fit(x, x, batch_size, nb_epoch, verbose, callbacks, validation_split,
                                           validation_data, shuffle, class_weight, sample_weight)
        return callbacks_history

    def crossfit(self, x, y, batch_size=None, nb_epoch_ae=5, nb_epoch_cf=10, verbose=1, callbacks=[],
                 validation_split=0.,
                 validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        valid_ae, valid_cf = validation_data
        for i in range(nb_epoch_ae):
            callbacks_history = self.model.fit(x, x, batch_size, 1, verbose, callbacks, validation_split,
                                               valid_ae, shuffle, class_weight, sample_weight)
        for i in range(nb_epoch_cf):
            callbacks_history = self.classifier.fit(x, y, batch_size, 1, verbose, callbacks, validation_split,
                                                    valid_cf, shuffle, class_weight, sample_weight)
        return callbacks_history

class CrossfireEEG(object):
    """
    Covolutional VAE with a "crossfire" component - a classifier is bolted onto the end of the encoder and loss function
    can be parameterized to function from classifier loss instead of autoencoder loss. Ideally, the model starts in
    pure autoencoder mode to learn features, then as loss flattens out the network starts weighing classifier loss
    more heavily.
    Note to self: things to try:
    * Add burst /batch / epoch noise to input
    * Modularize out the activation from dropout ordering"""

    def __init__(self, input_shape=(256, 16, 1), latent_dim=2, intermediate_dim=256, batch_size=100, epsilon_std=1.0,
                 dropout_p=0.1, n_stax=0, n_classes=10):
        # input image dimensions
        self.input_shape = input_shape
        if len(input_shape) == 3:
            self.img_rows, self.img_cols, self.img_chns = input_shape
        elif len(input_shape) == 2:
            self.img_rows, self.img_cols = input_shape
            self.img_chns = 1
        else:
            raise IndexError("Invalid shape: {}".format(input_shape))
        self.batch_size = batch_size
        self.original_dim = np.prod(input_shape)
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.epsilon_std = epsilon_std
        self.epsilon_ce = 1.0e-9

        # number of convolutional filters to use
        nb_filters = 64
        # convolution kernel size
        nb_conv = 3

        batch_size = 100
        if K_backend.image_dim_ordering() == 'th':
            self.original_img_size = (self.img_chns, self.img_rows, self.img_cols)
        else:
            self.original_img_size = (self.img_rows, self.img_cols, self.img_chns)

        x = Input(batch_shape=(batch_size,) + self.original_img_size, name='main_input')
        conv_1 = Convolution2D(self.img_chns, 2, 2, border_mode='same', activation='relu')(x)
        conv_3 = Convolution2D(nb_filters, 2, 2,
                               border_mode='same', activation='relu', subsample=(2, 2))(conv_1)
        # conv_3 = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', activation='relu', subsample=(1, 1))(
        #     conv_2)

        conv_3 = BatchNormalization()(conv_3)
        conv_3 = Dropout(dropout_p)(conv_3)
        conv_4 = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', activation='relu', subsample=(1, 1))(
            conv_3)
        flat = Flatten()(conv_4)
        hidden = Dense(intermediate_dim, activation='relu', name='intermed')(flat)

        self.z_mean = Dense(latent_dim, name='z_mean')(hidden)
        self.z_log_var = Dense(latent_dim, name='z_log_var')(hidden)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_var])`
        z = Lambda(self.sampling, output_shape=(latent_dim,))([self.z_mean, self.z_log_var])

        ## ==== End of encoding portion ======

        ## ==== Crossfire classifier =========
        #         c = Lambda(self.crosser, output_shape=(latent_dim,))(self.z_mean, self.z_log_var)
        classer = Dense(n_classes, init='normal', activation='softmax', name='classer')(self.z_mean)

        # Slicer example
        # y = Lambda(lambda x: x[:,0,:,:], output_shape=(1,) + input_shape[2:])(x)




        # we instantiate these layers separately so as to reuse them later
        decoder_hid = Dense(intermediate_dim, activation='relu')
        layersize = nb_filters * input_shape[0] * input_shape[0] // 4#nb_filters * 14 * 14
        decoder_upsample = Dense(layersize, activation='relu')

        # not sure if these need to be square or not
        if K_backend.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, input_shape[0]//2, input_shape[0]//2)
        else:
            output_shape = (batch_size, input_shape[0]//2, input_shape[0]//2, nb_filters)

        decoder_reshape = Reshape(output_shape[1:])
        decoder_deconv_1 = Deconvolution2D(nb_filters, nb_conv, nb_conv, output_shape, border_mode='same',
                                           subsample=(1, 1), activation='relu')
        decoder_deconv_2 = Deconvolution2D(nb_filters, nb_conv, nb_conv, output_shape, border_mode='same',
                                           subsample=(1, 1), activation='relu')

        # I have no idea where this 29 comes from
        if K_backend.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, input_shape[0]+1, input_shape[1]+1) #(batch_size, nb_filters, 29, 29)
        else:
            output_shape = (batch_size, input_shape[0]+1, input_shape[1]+1, nb_filters) # (batch_size, 29, 29, nb_filters)
        decoder_deconv_3_upsamp = Deconvolution2D(nb_filters, 2, 2, output_shape,
                                                  border_mode='valid', subsample=(2, 2), activation='relu')
        decoder_mean_squash = Convolution2D(self.img_chns, 2, 2, border_mode='valid',
                                            activation='sigmoid', name='main_output')

        hid_decoded = decoder_hid(z)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

        # build a digit generator that can sample from the learned distribution
        # todo: (un)roll this
        decoder_input = Input(shape=(latent_dim,))
        _hid_decoded = decoder_hid(decoder_input)
        _up_decoded = decoder_upsample(_hid_decoded)
        _reshape_decoded = decoder_reshape(_up_decoded)
        _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
        _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
        _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
        _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)

        # Generate models
        # Primary model - VAE
        self.model = Model(x, x_decoded_mean_squash)
        # Crossfilre network
        self.classifier = Model(x, classer)
        # Ok, now comes the tricky part. See these references:
        # https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
        self.crossmodel = Model(input=x, output=[x_decoded_mean_squash, classer])
        self.crossmodel.compile(optimizer='rmsprop',
                                loss={'main_output': self.vae_loss,
                                      'classer': 'categorical_crossentropy'},
                                loss_weights={'main_output': 1.0,
                                              'classer': 1.0})

        # build a model to project inputs on the latent space
        self.encoder = Model(x, self.z_mean)
        # reconstruct digits from latent space
        self.generator = Model(decoder_input, _x_decoded_mean_squash)
        self.model.compile(optimizer='rmsprop', loss=self.vae_loss)
        self.classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # self.classifier.compile(loss=self.custom_crossent, optimizer='adam', metrics=['accuracy'])

    def fit_crossmodel(self, x_dict, y_dict, batch_size=None, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
                       validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        callbacks_history = self.crossmodel.fit(x_dict, y_dict, batch_size, nb_epoch, verbose, callbacks,
                                                validation_split,
                                                validation_data, shuffle, class_weight, sample_weight)
        return callbacks_history

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K_backend.random_normal(shape=(self.batch_size, self.latent_dim),
                                          mean=0., std=self.epsilon_std)
        return z_mean + K_backend.exp(z_log_var) * epsilon

    def crosser(self, args):
        z_mean, z_log_var = args
        return z_mean

    def vae_loss(self, x, x_decoded_mean):
        # NOTE: binary_crossentropy expects a batch_size by dim
        # for x and x_decoded_mean, so we MUST flatten these!
        x = K_backend.flatten(x)
        x_decoded_mean = K_backend.flatten(x_decoded_mean)
        xent_loss = self.img_rows * self.img_cols * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K_backend.mean(
            1 + self.z_log_var - K_backend.square(self.z_mean) - K_backend.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss

    def custom_crossent(self, y_true, y_pred):
        '''Just another crossentropy'''

        #         y_pred = K_backend.clip(y_pred, self.epsilon_ce, 1.0 - self.epsilon_ce)
        #         y_pred /= K_backend.sum(y_pred, axis=-1, keepdims=True)
        cce = K_backend.categorical_crossentropy(y_pred, y_true)
        return cce

    def fit(self, x, y, batch_size=None, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        callbacks_history = self.model.fit(x, y, batch_size, nb_epoch, verbose, callbacks, validation_split,
                                           validation_data, shuffle, class_weight, sample_weight)
        return callbacks_history

    def fit_ae(self, x, batch_size=None, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
               validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        callbacks_history = self.model.fit(x, x, batch_size, nb_epoch, verbose, callbacks, validation_split,
                                           validation_data, shuffle, class_weight, sample_weight)
        return callbacks_history

    def crossfit(self, x, y, batch_size=None, nb_epoch_ae=5, nb_epoch_cf=10, verbose=1, callbacks=[],
                 validation_split=0.,
                 validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        valid_ae, valid_cf = validation_data
        for i in range(nb_epoch_ae):
            callbacks_history = self.model.fit(x, x, batch_size, 1, verbose, callbacks, validation_split,
                                               valid_ae, shuffle, class_weight, sample_weight)
        for i in range(nb_epoch_cf):
            callbacks_history = self.classifier.fit(x, y, batch_size, 1, verbose, callbacks, validation_split,
                                                    valid_cf, shuffle, class_weight, sample_weight)
        return callbacks_history



class Crossfire2(object):
    """
    Covolutional VAE with a "crossfire" component - a classifier is bolted onto the end of the encoder and loss function
    can be parameterized to function from classifier loss instead of autoencoder loss. Ideally, the model starts in
    pure autoencoder mode to learn features, then as loss flattens out the network starts weighing classifier loss
    more heavily.
    Note to self: things to try:
    * Add burst /batch / epoch noise to input
    * Modularize out the activation from dropout ordering"""

    def __init__(self, input_shape=(28, 28, 1), latent_dim=2, intermediate_dim=256, batch_size=100, epsilon_std=1.0,
                 dropout_p=0.1, n_stax=0, n_classes=10):
        hidden_activ = 'relu'
        final_activ = 'relu'
        # input image dimensions
        self.input_shape = input_shape
        if len(input_shape) == 3:
            self.img_rows, self.img_cols, self.img_chns = input_shape
        elif len(input_shape) == 2:
            self.img_rows, self.img_cols = input_shape
            self.img_chns = 1
        else:
            raise IndexError("Invalid shape: {}".format(input_shape))
        self.batch_size = batch_size
        self.original_dim = np.prod(input_shape)
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.epsilon_std = epsilon_std
        self.epsilon_ce = 1.0e-9

        # number of convolutional filters to use
        nb_filters = 64
        # convolution kernel size
        nb_conv = 3

        if K_backend.image_dim_ordering() == 'th':
            self.original_img_size = (self.img_chns, self.img_rows, self.img_cols)
        else:
            self.original_img_size = (self.img_rows, self.img_cols, self.img_chns)

        x = Input(batch_shape=(batch_size,) + self.original_img_size, name='main_input')
        conv_1 = Convolution2D(self.img_chns, 2, 2, border_mode='same', activation='relu')(x)
        conv_2 = Convolution2D(nb_filters, 2, 2,
                               border_mode='same', activation='relu', subsample=(2, 2))(conv_1)
        conv_3 = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', activation='relu', subsample=(1, 1))(
            conv_2)
        for i in range(n_stax):
            conv_3 = BatchNormalization()(conv_3)
            conv_3 = Dropout(dropout_p)(conv_3)
            conv_3 = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', activation='relu',
                                   subsample=(1, 1))(conv_3)

        conv_3 = BatchNormalization()(conv_3)
        conv_4 = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', activation='relu', subsample=(1, 1))(
            conv_3)
        flat = Flatten()(conv_4)
        hidden = Dense(intermediate_dim, activation='relu', name='intermed')(flat)

        self.z_mean = Dense(latent_dim, name='z_mean')(hidden)
        self.z_log_var = Dense(latent_dim, name='z_log_var')(hidden)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_var])`
        z = Lambda(self.sampling, output_shape=(latent_dim,))([self.z_mean, self.z_log_var])

        ## ==== End of encoding portion ======

        ## ==== Crossfire classifier =========
        #         c = Lambda(self.crosser, output_shape=(latent_dim,))(self.z_mean, self.z_log_var)
        classer = Dense(n_classes, init='normal', activation='softmax', name='classer')(self.z_mean)

        # Slicer example
        # y = Lambda(lambda x: x[:,0,:,:], output_shape=(1,) + input_shape[2:])(x)




        # we instantiate these layers separately so as to reuse them later
        decoder_hid = Dense(intermediate_dim, activation='relu')
        decoder_upsample = Dense(nb_filters * 14 * 14, activation='relu')

        if K_backend.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, 14, 14)
        else:
            output_shape = (batch_size, 14, 14, nb_filters)

        decoder_reshape = Reshape(output_shape[1:])
        decoder_deconv_1 = Deconvolution2D(nb_filters, nb_conv, nb_conv, output_shape, border_mode='same',
                                           subsample=(1, 1), activation='relu')
        decoder_deconv_2 = Deconvolution2D(nb_filters, nb_conv, nb_conv, output_shape, border_mode='same',
                                           subsample=(1, 1), activation='relu')
        if K_backend.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, 29, 29)
        else:
            output_shape = (batch_size, 29, 29, nb_filters)
        decoder_deconv_3_upsamp = Deconvolution2D(nb_filters, 2, 2, output_shape,
                                                  border_mode='valid', subsample=(2, 2), activation='relu')
        decoder_mean_squash = Convolution2D(self.img_chns, 2, 2, border_mode='valid',
                                            activation=final_activ, name='main_output')

        hid_decoded = decoder_hid(z)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

        # build a digit generator that can sample from the learned distribution
        # todo: (un)roll this
        decoder_input = Input(shape=(latent_dim,))
        _hid_decoded = decoder_hid(decoder_input)
        _up_decoded = decoder_upsample(_hid_decoded)
        _reshape_decoded = decoder_reshape(_up_decoded)
        _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
        _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
        _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
        _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)

        # Generate models
        # Primary model - VAE
        self.model = Model(x, x_decoded_mean_squash)
        # Crossfilre network
        self.classifier = Model(x, classer)
        # Ok, now comes the tricky part. See these references:
        # https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
        self.crossmodel = Model(input=x, output=[x_decoded_mean_squash, classer])
        self.crossmodel.compile(optimizer='rmsprop',
                                loss={'main_output': self.vae_loss,
                                      'classer': 'categorical_crossentropy'},
                                loss_weights={'main_output': 1.0,
                                              'classer': 5.0})

        # build a model to project inputs on the latent space
        self.encoder = Model(x, self.z_mean)
        # reconstruct digits from latent space
        self.generator = Model(decoder_input, _x_decoded_mean_squash)
        self.model.compile(optimizer='rmsprop', loss=self.vae_loss)
        self.classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #         self.classifier.compile(loss=self.custom_crossent, optimizer='adam', metrics=['accuracy'])


    def fit_crossmodel(self, x_dict, y_dict, batch_size=None, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
                       validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        callbacks_history = self.crossmodel.fit(x_dict, y_dict, batch_size, nb_epoch, verbose, callbacks,
                                                validation_split,
                                                validation_data, shuffle, class_weight, sample_weight)
        return callbacks_history

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K_backend.random_normal(shape=(self.batch_size, self.latent_dim),
                                          mean=0., std=self.epsilon_std)
        return z_mean + K_backend.exp(z_log_var) * epsilon

    def crosser(self, args):
        z_mean, z_log_var = args
        return z_mean

    def vae_loss(self, x, x_decoded_mean):
        # NOTE: binary_crossentropy expects a batch_size by dim
        # for x and x_decoded_mean, so we MUST flatten these!
        x = K_backend.flatten(x)
        x_decoded_mean = K_backend.flatten(x_decoded_mean)
        xent_loss = self.img_rows * self.img_cols * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K_backend.mean(
            1 + self.z_log_var - K_backend.square(self.z_mean) - K_backend.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss

    def custom_crossent(self, y_true, y_pred):
        '''Just another crossentropy'''

        #         y_pred = K_backend.clip(y_pred, self.epsilon_ce, 1.0 - self.epsilon_ce)
        #         y_pred /= K_backend.sum(y_pred, axis=-1, keepdims=True)
        cce = K_backend.categorical_crossentropy(y_pred, y_true)
        return cce

    def fit(self, x, y, batch_size=None, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        callbacks_history = self.model.fit(x, y, batch_size, nb_epoch, verbose, callbacks, validation_split,
                                           validation_data, shuffle, class_weight, sample_weight)
        return callbacks_history

    def fit_ae(self, x, batch_size=None, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
               validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        callbacks_history = self.model.fit(x, x, batch_size, nb_epoch, verbose, callbacks, validation_split,
                                           validation_data, shuffle, class_weight, sample_weight)
        return callbacks_history

    def crossfit(self, x, y, batch_size=None, nb_epoch_ae=5, nb_epoch_cf=10, verbose=1, callbacks=[],
                 validation_split=0.,
                 validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        valid_ae, valid_cf = validation_data
        for i in range(nb_epoch_ae):
            callbacks_history = self.model.fit(x, x, batch_size, 1, verbose, callbacks, validation_split,
                                               valid_ae, shuffle, class_weight, sample_weight)
        for i in range(nb_epoch_cf):
            callbacks_history = self.classifier.fit(x, y, batch_size, 1, verbose, callbacks, validation_split,
                                                    valid_cf, shuffle, class_weight, sample_weight)
        return callbacks_history
