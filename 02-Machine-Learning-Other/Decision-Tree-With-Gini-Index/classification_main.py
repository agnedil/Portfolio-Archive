
# based on the article http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# read data
train = pd.read_csv("trainingmod.txt", sep=' ', header=None)
test = pd.read_csv("testingmod.txt", sep=' ', header=None)

# extract features and labels
X_train = train.iloc[:,1:129].as_matrix(columns=None)
y_train = train.iloc[:,0].as_matrix(columns=None)
X_test = test.iloc[:,0:128].as_matrix(columns=None)

# set depth and minimum number of samples per leaf
depth = 11
leaf = 3

# decision tree classifier with Gini index
# max_features = all 127 features and presort increase training time
# depth of 11  with 3 samples - precision 51.4, depth 50 and 21 - precision 49.5 (overfitting?)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_features=127,
                                  max_depth=depth, min_samples_leaf=leaf, presort = True)
# training and predicting
clf_gini.fit(X_train, y_train)
y_testPred = clf_gini.predict(X_test)

# writing to file
f = open("results.txt","w+")
for item in y_testPred:
    f.write(str(item) + "\n")
f.close()
print ("Decision tree with {} leaves and a depth of {} is ready!".format(leaf, depth))