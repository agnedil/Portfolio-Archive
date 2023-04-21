from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D
from keras.layers import Input, LSTM, RepeatVector, Lambda
from keras.layers.core import Flatten
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
import numpy as np
import sys, glob
import os
import pytest
import argparse
import cv2
import scipy
import matplotlib.pyplot as plt

import struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros


# define model

def generator_model(inputdim = 100, xdim = 4, ydim = 4):
    # xdim = 2, ydim = 2 results in prediction shape of (1, 3, 32, 32)
    # xdim = 4, ydim = 4 results in prediction shape of (1, 3, 64, 64)
    model = Sequential()
    model.add(Dense(input_dim=inputdim, output_dim=1024*xdim*ydim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape( (1024, xdim, ydim), input_shape=(inputdim,) ) )
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(512, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(256, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(128, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model

def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(128, 5, 5, subsample=(2, 2), input_shape=(3, 64, 64), border_mode = 'same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Convolution2D(1024, 5, 5, subsample=(2, 2), border_mode = 'same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(output_dim=1))
    model.add(Activation('sigmoid'))
    return model

def va_model(batch_size=5, original_dim = 5, latent_dim = 10, intermediate_dim = 20, epsilon_std = 0.01):
    # Generate probabilistic encoder (recognition network), which
    # maps inputs onto a normal distribution in latent space.
    # The transformation is parametrized and can be learned.
    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., std=epsilon_std)
        return z_mean + K.exp(z_log_sigma) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    return x, x_decoded_mean, z_mean


def vaencoder_model():
    x, x_decoded_mean, z_mean = va_model(batch_size=5, original_dim = 5, latent_dim = 10, intermediate_dim = 20, epsilon_std = 0.01)
    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)
    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)
    return encoder, vae

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    #discriminator.trainable = False
    return model


# train model

   ''' Resize image to 64x64 and shuffle axis to create 3 arrays (RGB) '''
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, (64, 64))) / 127.5 - 1
    img = np.rollaxis(img, 2, 0)
    return img

def noise_image():
    ''' Create noisy data that will be converted to an image
        Note size = (total number, number in sublist, length of subsublist )
    '''
    zmb = np.random.uniform(-1, 1, 100)
    #zmb = np.random.uniform(0, 1, 100)
    return zmb

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]



def train(path, batch_size, EPOCHS):

    #reproducibility
    #np.random.seed(42)

    fig = plt.figure()

    # Get image paths
    print "Loading paths.."
    paths = glob.glob(os.path.join(path, "*.jpg"))
    print "Got paths.."

    # Load images
    IMAGES = np.array( [ load_image(p) for p in paths ] )
    np.random.shuffle( IMAGES )

    #IMAGES, labels = load_mnist(dataset="training", digits=np.arange(10), path=path)
    #IMAGES = np.array( [ np.array( [ scipy.misc.imresize(p, (64, 64)) / 256 ] * 3 ) for p in IMAGES ] )

    #np.random.shuffle( IMAGES )

    BATCHES = [ b for b in chunks(IMAGES, batch_size) ]

    discriminator = model.discriminator_model()
    generator = model.generator_model()
    discriminator_on_generator = model.generator_containing_discriminator(generator, discriminator)
    #adam_gen=Adam(lr=0.0002, beta_1=0.0005, beta_2=0.999, epsilon=1e-08)
    adam_gen=Adam(lr=0.00002, beta_1=0.0005, beta_2=0.999, epsilon=1e-08)
    adam_dis=Adam(lr=0.00002, beta_1=0.0005, beta_2=0.999, epsilon=1e-08)
    #opt = RMSprop()
    generator.compile(loss='binary_crossentropy', optimizer=adam_gen)
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=adam_gen)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_dis)

    print "Number of batches", len(BATCHES)
    print "Batch size is", batch_size

    #margin = 0.25
    #equilibrium = 0.6931
    inter_model_margin = 0.10

    for epoch in range(EPOCHS):
        print
        print "Epoch", epoch
        print

        # load weights on first try (i.e. if process failed previously and we are attempting to recapture lost data)
        if epoch == 0:
            if os.path.exists('generator_weights') and os.path.exists('discriminator_weights'):
                print "Loading saves weights.."
                generator.load_weights('generator_weights')
                discriminator.load_weights('discriminator_weights')
                print "Finished loading"
            else:
                pass

        for index, image_batch in enumerate(BATCHES):
            print "Epoch", epoch, "Batch", index

            Noise_batch = np.array( [ noise_image() for n in range(len(image_batch)) ] )
            generated_images = generator.predict(Noise_batch)
            #print generated_images[0][-1][-1]

            for i, img in enumerate(generated_images):
                rolled = np.rollaxis(img, 0, 3)
                cv2.imwrite('results/' + str(i) + ".jpg", np.uint8(255 * 0.5 * (rolled + 1.0)))

            Xd = np.concatenate((image_batch, generated_images))
            yd = [1] * len(image_batch) + [0] * len(image_batch) # labels

            print "Training first discriminator.."
            d_loss = discriminator.train_on_batch(Xd, yd)

            Xg = Noise_batch
            yg = [1] * len(image_batch)

            print "Training first generator.."
            g_loss = discriminator_on_generator.train_on_batch(Xg, yg)

            print "Initial batch losses : ", "Generator loss", g_loss, "Discriminator loss", d_loss, "Total:", g_loss + d_loss

            #print "equilibrium - margin", equilibrium - margin

            if g_loss < d_loss and abs(d_loss - g_loss) > inter_model_margin:
                #for j in range(handicap):
                while abs(d_loss - g_loss) > inter_model_margin:
                    print "Updating discriminator.."
                    #g_loss = discriminator_on_generator.train_on_batch(Xg, yg)
                    d_loss = discriminator.train_on_batch(Xd, yd)
                    print "Generator loss", g_loss, "Discriminator loss", d_loss
                    if d_loss < g_loss:
                        break
            elif d_loss < g_loss and abs(d_loss - g_loss) > inter_model_margin:
                #for j in range(handicap):
                while abs(d_loss - g_loss) > inter_model_margin:
                    print "Updating generator.."
                    #d_loss = discriminator.train_on_batch(Xd, yd)
                    g_loss = discriminator_on_generator.train_on_batch(Xg, yg)
                    print "Generator loss", g_loss, "Discriminator loss", d_loss
                    if g_loss < d_loss:
                        break
            else:
                pass

            print "Final batch losses (after updates) : ", "Generator loss", g_loss, "Discriminator loss", d_loss, "Total:", g_loss + d_loss
            print

            if index % 20 == 0:
                print 'Saving weights..'
                generator.save_weights('generator_weights', True)
                discriminator.save_weights('discriminator_weights', True)

        plt.clf()
        for i, img in enumerate(generated_images[:5]):
            i = i+1
            plt.subplot(3, 3, i)
            rolled = np.rollaxis(img, 0, 3)
            #plt.imshow(rolled, cmap='gray')
            plt.imshow(rolled)
            plt.axis('off')
        fig.canvas.draw()
        plt.savefig('Epoch_' + str(epoch) + '.png')

def generate(img_num):
    '''
        Generate new images based on trained model.
    '''
    generator = model.generator_model()
    adam=Adam(lr=0.00002, beta_1=0.0005, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss='binary_crossentropy', optimizer=adam)
    generator.load_weights('generator_weights')

    noise = np.array( [ noise_image() for n in range(img_num) ] )

    print 'Generating images..'
    generated_images = [np.rollaxis(img, 0, 3) for img in generator.predict(noise)]
    for index, img in enumerate(generated_images):
        cv2.imwrite("{}.jpg".format(index), np.uint8(255 * 0.5 * (img + 1.0)))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str)
    parser.add_argument("--TYPE", type = str)
    parser.add_argument("--batch_size", type = int, default=50)
    parser.add_argument("--epochs", type = int, default = 2)
    #parser.add_argument("--handicap", type = int, default = 2)
    parser.add_argument("--img_num", type = int, default = 10)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()

    if args.TYPE == 'train':
        train(path = args.path, batch_size = args.batch_size, EPOCHS = args.epochs)

    elif args.TYPE == 'generate':
        generate(img_num = args.img_num)



# inspired by https://github.com/jhayes14/GAN 