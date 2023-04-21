# AML UIUC

# MNIST dataset: training and testing Gaussian, Bernoulli Naive Bayes, Random Forest classifiers
# Random Forest with 10, 20, and 30 trees and a depth of 4, 8, and 16 in each case
# files train.csv and test.csv received from the original MNIST set by using the convertToCsv.R reader (enclosed)
# ideas from https://medium.com/@tenzin_ngodup/digit-prediction-using-multinomial-naive-bayes-in-python-6bf0d73f022e

import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#read data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#extract features and labels
features_train = train.iloc[:,0:784].as_matrix(columns=None)        #first 783 columns
labels_train = train.iloc[:,-1].as_matrix(columns=None)             #last column, 784:785 can be used instaed of -1 to denote it
features_test = test.iloc[:,0:784].as_matrix(columns=None)
labels_test = test.iloc[:,-1].as_matrix(columns=None)

#train GaussianNB and BernoulliNB
clf = GaussianNB()
clf.fit(features_train, labels_train.ravel())
clf2=BernoulliNB()
clf2.fit(features_train, labels_train.ravel())

#apply trained models to test data
y_pred = clf.predict(features_test)
y_pred2 = clf2.predict(features_test)
y_true = labels_test.ravel()

#print Gaussian and Bernoulli NB accuracies
print(" Gaussian NB accuracy: " + str(accuracy_score(y_true, y_pred)))
print("Bernoulli NB accuracy: " + str(accuracy_score(y_true, y_pred2)))

#the same for decision forest with the required number of trees and depth
#printing accuracies in a loop
trees = [10, 20, 30]
depths = [4, 8, 16]
for tree in trees:
    for depth in depths:
        clf3 = RandomForestClassifier(n_estimators=tree, max_depth=depth)
        clf3.fit(features_train, labels_train.ravel())
        y_pred3 = clf3.predict(features_test)
        accuracy = accuracy_score(y_true, y_pred3)
        print ("Random Forest accuracy with {} trees and a depth of {}: ".format(tree, depth) + str(accuracy))
