# AML UIUC

# Expectation Maximization Algorithm (from scratch)
# 2a. Segment each of the three provided test images to 10, 20, and 50 segments
# 2b. Segment one these images to 20 segments using five different start points

# hints from https://stackoverflow.com/questions/8550912/python-dictionary-of-dictionaries
# hints from https://stackoverflow.com/questions/16333296/how-do-you-create-nested-dict-in-python

# see detailed comments in part 1; modifications were made

import sys, math, random, copy
import numpy as np
from scipy import misc
from scipy.spatial import distance
from scipy.cluster.vq import kmeans
from numpy.linalg import inv


def Q(x, numClust, mu, pi, w):

    sigma = 0
    for j in xrange(numClust):
        expValue = pi[j]*np.exp(-1*((((x-mu[j])**2).sum(1)))/2.0)
        dot = -1*(expValue)/2.0
        sigma += (dot + pi[j])*w[:,j]
    return np.sum(sigma)


def expect(x, numClust, mu, pi):

    denominator = np.zeros((x.shape[0]))
    w = np.zeros((x.shape[0], numClust))
    distances = distance.cdist(x, mu, 'euclidean')
    dMin = np.amin(distances, axis=1)
    dminSqr = np.square(dMin)
    for k in xrange(numClust):
        expValue = pi[k]*np.exp(-1*((((x-mu[k])**2).sum(1))-dminSqr)/2.0)
        denominator += expValue
    for k in xrange(numClust):
        expValue = pi[k]*np.exp(-1*((((x-mu[k])**2).sum(1))-dminSqr)/2.0)
        w[:,k] = expValue/denominator
    return w


def maximize(x, w, numClust):

    newPi = np.zeros((numClust))
    newMu = np.zeros((numClust, 3))
    threeW = np.zeros((w.shape[0],w.shape[1],3))
    threeW[:,:,0] = w
    threeW[:,:,1] = w
    threeW[:,:,2] = w
    for j in xrange(numClust):
        denominator = np.sum(w[:,j])
        numerator = np.sum(x*threeW[:,j,:], axis=0)
        newMu[j] = numerator/denominator
        newPi[j] = denominator/(x.shape[0])
    return newMu, newPi


def EM(x, numClust):

    temp = x
    mu, dist = kmeans(temp, numClust, iter=5)                                           # use kmeans to initialize cluster centers

    pi = np.full((numClust), 1.0/numClust)
    iter = 0
    difference = 1
    q = 0
    while iter < 100 and difference > 0.0001:                                           # per Piazza post @1218 (follow-ups from TA) - convergence criterion
        iter += 1
        print "iteration " + str(iter)
        w = expect(temp, numClust, mu, pi)
        mu, pi = maximize(temp, w, numClust)
        oldq = q
        q = Q(temp, numClust, mu, pi, w)
        difference = abs(q-oldq)/abs(q)
        if iter != 1:
            print "Difference between iterations = " + str(difference)

    result = {}
    result['clusterCentroids'] = {}
    result['parameters'] = {}
    for i in xrange(numClust):
        result['parameters'][i] = {}
        result['parameters'][i]['pi'] = pi[i]
        result['parameters'][i]['mu'] = mu[i]

    for index, pixel in enumerate(temp):
        pixelDist = np.zeros((numClust, 3))
        for i in xrange(numClust):
            pixelDist[i] = pixel
        distances = distance.cdist(pixelDist, mu, 'euclidean')
        cluster = np.argmin(distances)
        if cluster not in result['clusterCentroids']:
            result['clusterCentroids'][cluster]=[]
        result['clusterCentroids'][cluster].append(index)

    return result


def segmentImg(imgName, numClust, addInfo):

    fileName = imgName + '.jpg'                                                         # imgName is needed later in segmentImg() to create the
    img = misc.imread(fileName)                                                         # the name of the file to be written to disk
    imgTransf = []
    for row in img:
        for pixel in row:
            imgTransf.append(pixel)                                                     # convert to list

    imgNumpy = np.empty([len(img) * len(img[0]), 3])
    for index, item in enumerate(imgTransf):
        imgNumpy[index] = item                                                          # convert to numpy array

    result = EM(imgNumpy, numClust=numClust)                                            # run EM

    imgTofile = np.empty([len(img), len(img[0]), 3])                                    # save segmented image to file
    for i in xrange(numClust):
        for item in result['clusterCentroids'][i]:
            row = item // len(img[0]);
            col = item - row * len(img[0]);
            imgTofile[row][col] = result['parameters'][i]['mu']
    name = imgName + '_' + str(numClust) + '_segments' + addInfo + '.jpg'
    misc.imsave(name, imgTofile)
    print
    print(name + ' is done!')
    print


if __name__ == "__main__":

    print
    print("Part 2a. Segmenting the 3 provided images to 10, 20, and 50 segments")
    print
    listImg = ['RobertMixed03', 'smallstrelitzia', 'smallsunset']                               # '.jpg' will be added later segmentImg(), the handling
    listSegm = [10, 20, 50]                                                                     # is needed to modify file names when writing to disk
    for image in listImg:
        for numSegm in listSegm:
            print('segmenting ' + image + '.jpg to ' + str(numSegm) + ' segments')
            print
            addInfo = ''
            segmentImg(image, numSegm, addInfo)

    print
    print("Part 2b. Segmenting smallsunset.jpg to 20 segments using five different start points")
    print

    numSegm = 20
    for round in xrange(5):
        print('startpoint ' + str(round+1) + ' for smallsunset.jpg')
        print
        addInfo = '_startpoint' + str(round+1)
        segmentImg('smallsunset', numSegm, addInfo)
