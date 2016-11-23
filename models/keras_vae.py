'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

# global
nb_epoch = 50

class VariationalAutoencoder(object):
    def __init__(self, original_dim=784, latent_dim=2, intermediate_dim=256, batch_size=100, epsilon_std=1.0):
        #vae params
        self.batch_size = batch_size
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.epsilon_std = epsilon_std

        x = Input(batch_shape=(batch_size, original_dim))
        h = Dense(intermediate_dim, activation='relu')(x)
        self.z_mean = Dense(latent_dim)(h)
        self.z_log_var = Dense(latent_dim)(h)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(latent_dim,))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)


        self.model = Model(x, x_decoded_mean)
        self.model.compile(optimizer='rmsprop', loss=self.vae_loss)

        # build a model to project inputs on the latent space
        self.encoder = Model(x, self.z_mean)

        # build a digit generator that can sample from the learned distribution
        decoder_input = Input(shape=(latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        self.generator = Model(decoder_input, _x_decoded_mean)


    def vae_loss(self, x, x_decoded_mean):
        xent_loss = self.original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss


    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.,
                                  std=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def fit(self, x, y, batch_size=None, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None):
        pass


vaeclass = VariationalAutoencoder()
vae = vaeclass.model

# ==== dataset handling - train the VAE on MNIST digits ===
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test))


# ===== plotting encoder output
# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()


# ====== plotting decoder from latent space =======
# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()