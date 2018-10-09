from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras import optimizers

import numpy as np

import argparse

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

import sys

if sys.platform != 'darwin':
    import matplotlib.pyplot as plt

from random import randint

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    # filename = os.path.join(model_name, "vae_mean.png")
    # # display a 2D plot of the digit classes in the latent space
    # z_mean, S_log_var, C_mean, C_log_var, C_S_VECTER = encoder.predict(x_test, batch_size=batch_size)
    # plt.figure(figsize=(12, 10))
    # plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    # plt.colorbar()
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.savefig(filename)
    # plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


def ml_plot_result(encoder, decoder, x_train, y_train):
    def ten_digits(x, y):
        digits = {}
        while True:
            i = randint(0, len(y))
            d = y[i]
            if d not in digits:
                digits[d] = x[i, ...]
                if len(digits) == 10:
                    break
        return np.array([digits[i] for i in range(10)])

    content_x = ten_digits(x_train, y_train);
    print(content_x.shape)
    style_x = ten_digits(x_train, y_train);
    print(style_x.shape)
    latent_c_ = encoder.predict(content_x)[4]
    latent_s_ = encoder.predict(style_x)[4]
    # [content:10 style:10]
    cc = latent_c_[:, :10]
    ss = latent_s_[:, 10:]

    c_and_s = []
    for ci in range(10):
        content_pics = []
        for si in range(10):
            res = decoder.predict(np.array([np.concatenate([cc[ci], ss[si]])]))
            res = res.reshape([28, 28])
            content_pics.append(res)
        c_and_s.append(np.concatenate(content_pics, axis=1))
    c_and_s = np.concatenate(c_and_s, axis=0)

    np.save('stacked', c_and_s)

    print(np.concatenate(content_x, axis=0).reshape([-1, 28, 28]).shape)

    content_stack = np.concatenate([
        np.concatenate(content_x.reshape([-1, 28, 28]), axis=0),
        np.concatenate(decoder.predict(latent_c_).reshape([-1, 28, 28]), axis=0)
    ], axis=1)

    style_stack = np.concatenate([
        np.concatenate(style_x.reshape([-1, 28, 28]), axis=1),
        np.concatenate(decoder.predict(latent_s_).reshape([-1, 28, 28]), axis=1)
    ], axis=0)

    overall_stack = np.concatenate([
        np.concatenate([np.ones((28 * 2, 28 * 2)), content_stack], axis=0),
        np.concatenate([style_stack, c_and_s], axis=0)
    ], axis=1)

    np.save('overall_stack', overall_stack)

    # print(x_train.shape)
    # content = x_train[0,:]
    # style = x_train[7001,:]
    # print(content.shape)
    #
    # latent_c = encoder.predict([[content]])[4]
    # latent_s = encoder.predict([[style]])[4]
    #
    # print(latent_c.shape)
    # print(latent_s.shape)
    #
    # latent_c = latent_c[:,10:]
    # latent_s = latent_s[:,10:]
    #
    # print(latent_c.shape)
    # print(latent_s.shape)
    #
    # combine = np.concatenate([latent_c, latent_s], axis=1)
    # print(combine.shape)
    #
    # x_hat = decoder.predict(combine)
    # print(x_hat.shape)
    #
    # np.save('content', content.reshape([28,28]))
    # np.save('style', style.reshape([28,28]))
    # np.save('combine', x_hat.reshape([28,28]))


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def main():
    # 设置好参数
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()

    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    input_shape = (original_dim,)
    intermediate_dim = 500
    batch_size = 128
    latent_dim = 10
    latent_dim2 = 20
    epochs = 100

    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='tanh')(inputs)
    S_mean = Dense(latent_dim, name='S_mean')(x)
    S_log_var = Dense(latent_dim, name='S_log_var')(x)
    C_mean = Dense(latent_dim, name="C_mean")(x)
    C_log_var = Dense(latent_dim, name='C_log_var')(x)

    S_G = Lambda(sampling, output_shape=(latent_dim,), name='S_G')([S_mean, S_log_var])
    C_G = Lambda(sampling, output_shape=(latent_dim,), name='C_G')([C_mean, C_log_var])

    # concatenate the vector
    C_S_VECTER = concatenate([C_G, S_G], axis=-1)
    encoder = Model(inputs, [S_mean, S_log_var, C_mean, C_log_var, C_S_VECTER], name='encoder')
    encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim2,), name='sampling')
    x = Dense(intermediate_dim, activation='tanh')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[4])
    ml_vae = Model(inputs, outputs, name='ML_VAE')

    models = (encoder, decoder)
    data = (x_test, y_test)
    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs, outputs)

    z_mean = S_mean + C_mean
    z_log_var = S_log_var + C_log_var
    # set loss
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    ml_vae.add_loss(vae_loss)
    adam = optimizers.adam()
    ml_vae.compile(optimizer='adam',loss="")
    ml_vae.summary()
    plot_model(ml_vae,
               to_file='ml_vae.png',
               show_shapes=True)

    if args.weights:
        ml_vae.load_weights(args.weights)
    else:
        # train the autoencoder
        ml_vae.fit(x_train,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(x_test, None))
        ml_vae.save_weights('vae_mlp_mnist.h5')

    # plot_results(models,
    #              data,
    #              batch_size=batch_size,
    #              model_name="vae_mlp")

    ml_plot_result(encoder, decoder, x_train, y_train)


if __name__ == '__main__':
    main()
