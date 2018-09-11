# AML UIUC

# Modified MNIST dataset: training and testing Gaussian, Bernoulli Naive Bayes, Random Forest classifiers
# Random Forest with 10, 20, and 30 trees and a depth of 4, 8, and 16 in each case
# modifications: crop each MNIST image by ink boundaries, resize to 20 x 20, use for classification
# ideas from https://gist.github.com/fukuroder/caa351677bf718a8bfe6265c2a45211f
# ideas from https://stackoverflow.com/questions/44383209/how-to-detect-edge-and-crop-an-image-in-python
# some ideas from https://medium.com/@tenzin_ngodup/digit-prediction-using-multinomial-naive-bayes-in-python-6bf0d73f022e

import pandas as pd
import numpy as np
import cv2
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def processimg(imgfile, lblfile):

    listimg = []
    listlbl = []

    #all data files are downloaded and unpacked in the tmp/ directory inside the working directory
    imgfile1="tmp/" + imgfile
    lblfile1="tmp/" + lblfile

    #read images and labels
    with open(imgfile1, 'rb') as f:
        images = f.read()
    with open(lblfile1, 'rb') as f:
        labels = f.read()

    #format data
    images = [ord(d) for d in images[16:]]
    images = np.array(images, dtype=np.uint8)
    images = images.reshape((-1, 28, 28))

    #create directory to save images (so that you could see the result of bounding and stretching)
    #outdir = image_f + "_folder"
    #if not os.path.exists(outdir):
        #os.mkdir(outdir)

    #iterating over images: create bounding box, stretch images to 20 x 20, and store them in a list
    for k, image in enumerate(images):

        # threshold to get just the number from the image
        retval, thresh_gray = cv2.threshold(image, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

        # find where the number is and make a cropped region
        points = np.argwhere(thresh_gray == 255)        #find where the black pixels are
        points = np.fliplr(points)                      #store them in x,y coordinates instead of row,col indices
        x, y, w, h = cv2.boundingRect(points)           #create a rectangle around those points
        crop = image[y:y + h, x:x + w]                  #create a cropped region of the image

        # get the thresholded crop
        retval, thresh_crop = cv2.threshold(crop, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

        # resize image to 20 x 20
        image = cv2.resize(thresh_crop, (20, 20))
        listimg.append(image)

        #write each image to disk ((so that you could see the result of bounding and strethcing)
        #cv2.imwrite(os.path.join(outdir, '%05d.png' % (k,)), image)

    #iterate over labels and store them in a list
    for k, l in enumerate(labels[8:]):
        listlbl.append(ord(l))
    #write labels to file (in case they are needed with the cropped images)
    #labels = [outdir + '/%05d.png %d' % (k, ord(l)) for k, l in enumerate(labels[8:])]
    #with open('%s.txt' % label_f, 'w') as f:
        #f.write(os.linesep.join(labels))

    return listimg, listlbl

train_image = 'train-images-idx3-ubyte'
train_label = 'train-labels-idx1-ubyte'
test_image = 't10k-images-idx3-ubyte'
test_label = 't10k-labels-idx1-ubyte'

#download the original MNIST dataset from the original website to the tmp directory inside the working directory
for f in [train_image, train_label, test_image, test_label]:
    os.system('wget -P tmp --no-check-certificate http://yann.lecun.com/exdb/mnist/%s.gz' % (f,))

#unpack files
for f in [train_image, train_label, test_image, test_label]:
    os.system('gunzip tmp/%s.gz' % (f,))

#set features and labels
d3_features_train, labels_train = processimg(train_image, train_label)
d3_features_test, labels_test = processimg(test_image, test_label)

#making 3D array to be 2D to use with fit
d3_features_train = np.array(d3_features_train)
d3_features_test = np.array(d3_features_test)
nsamples, nx, ny = d3_features_train.shape
features_train = d3_features_train.reshape((nsamples,nx*ny))
nsamples, nx, ny = d3_features_test.shape
features_test = d3_features_test.reshape((nsamples,nx*ny))

#train GaussianNB and BernoulliNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
clf2=BernoulliNB()
clf2.fit(features_train, labels_train)

#apply trained models to test data
y_pred = clf.predict(features_test)
y_pred2 = clf2.predict(features_test)
y_true = labels_test

#print Gaussian and Bernoulli NB accuracies
print(" Gaussian NB accuracy: " + str(accuracy_score(y_true, y_pred)))
print("Bernoulli NB accuracy: " + str(accuracy_score(y_true, y_pred2)))

#the same for decision forest with the required depth and number of trees
#printing accuracies in a loop
trees = [10, 20, 30]
depths = [4, 8, 16]
for tree in trees:
    for depth in depths:
        clf3 = RandomForestClassifier(n_estimators=tree, max_depth=depth)
        clf3.fit(features_train, labels_train)
        y_pred3 = clf3.predict(features_test)
        accuracy = accuracy_score(y_true, y_pred3)
        print ("Random Forest accuracy with {} trees and a depth of {}: ".format(tree, depth) + str(accuracy))
