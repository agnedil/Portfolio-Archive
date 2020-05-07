# AML UIUC

# load MINST data - hint from Martin Thoma at https://martin-thoma.com/classify-mnist-with-pybrain/
# saving nympy array to file - hint from askewchan at https://stackoverflow.com/questions/22557322/numpy-savetxt-formatted-as-integer-is-not-saving-zeroes

# My interpretation:
# real image X = binImg -> noisyImg, inferred image H = denoisedImg
# three cases of neighbors - 2, 3, or 4 neighbors that a pixel can have in a 28 x 28 matrix

import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
from numpy import zeros, float32, genfromtxt
from copy import deepcopy
from pylab import imshow, show, cm


def get_MNIST_images():                                             # load MNIST data

    filepath = "data/train-images-idx3-ubyte.gz"
    images = gzip.open(filepath, 'rb')                              # open w/gzip and read binary data

    images.read(4)                                                  # get metadata first; skip the magic_number
    numImg = images.read(4)
    numImg = unpack('>I', numImg)[0]                                # big endian unsigned integer - '>I'
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    x = zeros((20, rows, cols), dtype=float32)                      # initialize numpy array to get data for first 20 images
    for i in range(20):
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)
                tmp_pixel = unpack('>B', tmp_pixel)[0]              # just a single byte - '>B'
                x[i, row, col] = tmp_pixel
    print("Data loaded!")
    return x


def view_image(img):

    imshow(img, cmap=cm.gray)                                       # view single image, used for debugging
    show()


def binarize(images):                                               # turn pixels into '1' or '-1'

    images = images / 255
    for img in range(images.shape[0]):                              # mages.shape[0] = 20 (images) in this case
        for i in range(images.shape[1]):                            # mages.shape[1] adn [2] = 28 = image size 28 x 28
            for j in range(images.shape[2]):
                if images[img, i, j] <= 0.5:                        # less than 127.5 / 255
                    images[img, i, j] = -1                          # x[0,2] = x[0][2], but the former is faster
                else:
                    images[img, i, j] = 1
    print("Data binarized!")
    return images


def addNoise(images):

    noise = genfromtxt('NoiseCoordinates.csv', skip_header=1, delimiter=',', dtype=int)   # read coordinates where noise has to be introduced
    noise = np.delete(noise, 0, axis=1)                                                 # delete column 0 (axis=1 -> vertically)
    for img in range(20):
        this_noise = []                                                                 # noise for this particular image
        row1 = 2 * img                                                                  # location of row
        row2 = 2 * img + 1                                                              # location of column
        for col in range(noise.shape[1]):
            this_noise.append((noise[row1, col], noise[row2, col]))                     # get noise for this particular image from matrix noise
        for point in this_noise:
            images[img, point[0], point[1]] = -images[img, point[0], point[1]]          # apply noise to this particular image

    #for img in images:                                                                 # inspect images if needed
    #    view_image(img)

    return images

def vfe(pi, x):     # pi = 28 x 28 matrix, x = 28 x 28 matrix

    # ENTROPY PORTION OF ENERGY EQUATION = EQLogQ (MP8 Instructions)
    eps = 10e-10
    entropy = 0
    for i in range (x.shape[0]):
        for j in range (x.shape[1]):
            entropy += pi[i,j] * np.log(pi[i,j] + eps) + (1 - pi[i,j]) * np.log(1 - pi[i,j] + eps)

    # 2ND TERM OF LOG LIKELIHOOD PORTION OF ENERGY EQUATION (COMES FIRST BECAUSE EASIER TO ESTIMATE)
    term2log = 0
    for i in range (x.shape[0]):
        for j in range (x.shape[1]):
            term2log += 2 * (2*pi[i, j] - 1) * x[i, j]

    # 1ST TERM OF LOG LIKELIHOOD PORTION OF ENERGY EQUATION
    term1log = 0
    shape = x.shape[1]
    for i in range (x.shape[0]):
        for j in range (x.shape[1]):
            if (i in range(1, shape - 1) and j in range(1, shape - 1)):     # case w/4 neighbours
                term1log += 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i, j+1] - 1) + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i, j-1] - 1)\
                            + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i-1, j] - 1) + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i+1, j] - 1)

            elif i == 0 and j == 0:                                         # cases w/2 neighbours
                term1log += 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i, j+1] - 1) + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i+1, j] - 1)

            elif i == 0 and j == shape - 1:
                term1log += 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i, j-1] - 1) + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i+1, j] - 1)

            elif i == shape - 1 and j == 0:
                term1log += 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i-1, j] - 1) + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i, j+1] - 1)

            elif i == shape - 1 and j == shape - 1:
                term1log += 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i-1, j] - 1) + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i, j-1] - 1)

            elif j == 0:                                                    # cases w/3 neighbours
                term1log += 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i-1, j] - 1) + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i, j+1] - 1)\
                            + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i+1, j] - 1)

            elif i == 0:
                term1log += 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i+1, j] - 1) + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i, j+1] - 1)\
                            + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i, j-1] - 1)

            elif i == shape - 1:
                term1log += 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i-1, j] - 1) + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i, j+1] - 1)\
                            + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i, j-1] - 1)

            else:
                term1log += 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i-1, j] - 1) + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i, j-1] - 1)\
                            + 0.8 * (2 * pi[i, j] - 1) * (2 * pi[i+1, j] - 1)

    energy = entropy - (term1log + term2log)

    return energy

def denoise(noisyImg):

    denoisedImg = np.zeros((noisyImg.shape[0], noisyImg.shape[1], noisyImg.shape[2]))   # initialize array for denoised images H = 20 x 28 x 28
    shape = denoisedImg.shape[1]                                                        # 28
    updateOrder = genfromtxt('UpdateOrderCoordinates.csv', skip_header=1, delimiter=',', dtype=int)    # get the update order matrix
    updateOrder = np.delete(updateOrder, 0, axis=1)                                     # delete column 0 w/text (axis=1 -> vertically)

    VFE = np.zeros((20, 11))

    for img in range(20):
        pi = genfromtxt('InitialParametersModel.csv', delimiter=',')                    # initial matrix of pi; pd.read_csv faster, but different implications

        VFE[img, 0] = vfe(pi, noisyImg[img])                                            # initial energy for this image

        this_updateOrder = []                                                           # create update order for this particular image
        row1 = 2*img                                                                    # location of row to be updated
        row2 = 2*img + 1                                                                # location of column to be updated
        for col in range (784):
            this_updateOrder.append((updateOrder[row1, col], updateOrder[row2, col]))   # list of tuples of image's points coordinates in the required update order

        for iter in range(10):                                                          # run mean-field approximation 10 times, as per instructions
            power = np.zeros((shape, shape))                                            # 28 x 28 matix for values of the power for e for each pi[i, j]

            for point in this_updateOrder:                                              # update images in the required order
                i = point[0]
                j = point[1]
                term2 = 2 * noisyImg[img, i, j]                                         # fixed 2nd term in sum in power of e, as there is only one X

                if (i in range(1, shape - 1) and j in range(1, shape - 1)):             # pi case w/4 neighbours
                    power[i, j] = 0.8 * (2 * pi[i, j+1] - 1 + 2 * pi[i, j-1] - 1 + 2 * pi[i-1, j] - 1 + 2 * pi[i+1, j] - 1) + term2

                elif i == 0 and j == 0:                                                 # pi cases w/2 neighbours
                    power[i, j] = 0.8 * (2 * pi[i, j+1] - 1 + 2 * pi[i+1, j] - 1) + term2

                elif i == 0 and j == shape - 1:
                    power[i, j] = 0.8 * (2 * pi[i, j-1] - 1 + 2 * pi[i+1, j] - 1) + term2

                elif i == shape - 1 and j == 0:
                    power[i, j] = 0.8 * (2 * pi[i-1, j] - 1 + 2 * pi[i, j+1] - 1) + term2

                elif i == shape - 1 and j == shape - 1:
                    power[i, j] = 0.8 * (2 * pi[i-1, j] - 1 + 2 * pi[i, j-1] - 1) + term2

                elif j == 0:                                                            # pi cases w/3 neighbours
                    power[i, j] = 0.8 * (2 * pi[i-1, j] - 1 + 2 * pi[i, j+1] - 1 + 2 * pi[i+1, j] - 1) + term2

                elif i == 0:
                    power[i, j] = 0.8 * (2 * pi[i+1, j] - 1 + 2 * pi[i, j+1] - 1 + 2 * pi[i, j-1] - 1) + term2

                elif i == shape - 1:
                    power[i, j] = 0.8 * (2 * pi[i-1, j] - 1 + 2 * pi[i, j+1] - 1 + 2 * pi[i, j-1] - 1) + term2

                else:
                    power[i, j] = 0.8 * (2 * pi[i-1, j] - 1 + 2 * pi[i, j-1] - 1 + 2 * pi[i+1, j] - 1) + term2

                pi[i, j] = np.exp(power[i, j]) / (np.exp(power[i, j]) + np.exp(-power[i, j]))       # formula for pi from p. 263 of course textbook

                if pi[i, j] < 0.5:
                    denoisedImg[img, i, j] = -1                                         # select pixel value based on pi
                else:
                    denoisedImg[img, i, j] = 1

            VFE[img, iter + 1] = vfe(pi, denoisedImg[img])                             # energy of this image after each iteration


    with open('thisEnergy.csv', 'w') as f:                                             # save energies for every image / iteration (for debugging)
        np.savetxt(f, VFE, fmt='%1.9f', delimiter=",")

    #save VFE 2 x 2 in csv                                                             # save energy for images 10, 11 / iterations 0 and 1 (to submit)

    print("Images denoised! VFE estimated!")
    return denoisedImg

if __name__ == "__main__":

    rawImg = get_MNIST_images()                                                         # read 20 first MNIST images

    binImg = binarize(rawImg)                                                           # convert pixels to -1 or 1

    noisyImg = deepcopy(binImg)                                                         # keep copy of binarized raw images for comparison

    noisyImg = addNoise(noisyImg)                                                       # add predefined noise coordinates from file

    denoisedImg = denoise(noisyImg)                                                     # denoise images (w/special predefined update order from file)

    for img in denoisedImg:
        img[img < 0] = 0                                                                # convert image to binary 0 & 1 -> numpy indexing trick

    concatImg = np.concatenate(denoisedImg[10:], axis=1)                                # concatenation last 10 images by column (axis=1)
    with open('denoised.csv', 'w') as f:                                                # save images left to right in an csv file (to submit)
        np.savetxt(f, concatImg, fmt='%i', delimiter=",")                               # save as integer, fmt='%i'

    #for img in denoisedImg:                                                            # inspect images if needed
    #    view_image(img)

    print("Denoised images saved! Done!")
