import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB,  BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import nltk
#nltk.download('wordnet')    #already done (once is enough)
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

path=""
ftext="dataset/hygiene.dat"
fdatLables="dataset/hygiene.dat.labels"
finLabel="labels.txt"
fadd="dataset/hygiene.dat.additional"
stopwordsFile="stopwords_abbreviated_more.txt"

port = PorterStemmer()

def loadText():

   txt=[]
   idx = 0
   print("loading text")
   with open(path+ftext,'r') as f:
      for line in f.readlines():
          #if idx < 546:
          line = line.strip()
          line = line.lower()
          myline = " ".join([port.stem(i.strip()) for i in line.split()])
          txt.append(myline)
          idx += 1
   print("Length of text: ")
   print len(txt)
   return txt


def loadLabels():
   lab=[]
   i=0
   with open(path+fdatLables,'r') as f:
      for line in f.readlines():
         if i==546: break
         lab.append(int(line))
         i+=1
   print("Length of labels: ")
   print len(lab)
   return lab


def loadAdditional():

   add=[]
   idx = 0
   with open(path+fadd,'r') as f:
      for line in f.readlines():
         #if idx < 546:
         add.append(line)
         idx += 1
       
   print("Length of additional data: ")
   print len(add)
   return add   


# combines text data and additional data
def combine(data,add):

   d=[]
   for i in range(0,len(data)):
      d.append(add[i]+' '+data[i])
     
   #print d[0:2]
   print("Additional data added")
   return d


def stopwords():

    sw = []
    with io.open(path + stopwordsFile, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.lower()
            line = port.stem(line)
            if line == 'a' or line == 'i'or line == 'I' or len(line) > 1:
                sw.append(line)

    print("Stopwords read")
    return sw

# saves predicted labels to a file to be submitted to the leaderboard; algo = classifier name
def saveLabels(labels, algo):
   fileName = algo + "_" + finLabel
   print('Saving ' + fileName)
   with open(path + fileName,'w') as f:
      f.write('andrew\n')
      for label in labels:
         f.write(str(label)+'\n')


def clfSGD(data,labels, stwords):

   trainD1=data[0:546]
   testD=data[546:]

   trainX, testX, trainY, testY = train_test_split(trainD1, labels, test_size=0.2, random_state=42)
   clf = Pipeline([('vect', CountVectorizer(analyzer='word', stop_words=stwords, min_df=15, max_df=0.25, ngram_range=(1, 5))),    # initially min_df=15, max_df=0.22
                   ('tfidf', TfidfTransformer()),
                  ('clf', SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.151, alpha=1.01e-3, n_iter=11, random_state=42))
                   ,   
   ])    
   clf = clf.fit(trainX, trainY)
   predictions=clf.predict(testX)
   score = metrics.f1_score(testY, predictions)
   print("SGD F1-Score:   %0.3f" % score)

   finalPredictions=clf.predict(testD)                   # uncomment to save results
   saveLabels(finalPredictions, 'SGD')

def clfLinSVC(data, labels, stwords):

   trainD1=data[0:546]
   testD=data[546:]

   trainX, testX, trainY, testY = train_test_split(trainD1, labels, test_size=0.2, random_state=42)
   clf = Pipeline([('vect', CountVectorizer(analyzer='word', stop_words=stwords, min_df=15, max_df=0.25, ngram_range=(1, 2))),
                   ('tfidf', TfidfTransformer()),
                   ('clf', svm.LinearSVC()),
                 ])
   clf = clf.fit(trainX, trainY)
   predictions=clf.predict(testX)
   score = metrics.f1_score(testY, predictions)
   print("LinearSVC F1-Score:   %0.3f" % score)             # F1-Score = 0.993 !!!  SVC = 0.648 only, but maybe it does not overfit as much? Play with parameters

   #finalPredictions = clf.predict(testD)                   # uncomment to save results
   #saveLabels(finalPredictions, 'LinearSVC')


def clfNuSVC(data, labels, stwords):

   trainD1=data[0:546]
   testD=data[546:]

   trainX, testX, trainY, testY = train_test_split(trainD1, labels, test_size=0.2, random_state=42)
   clf = Pipeline([('vect', CountVectorizer(analyzer='word', stop_words=stwords, min_df=10, max_df=0.25, ngram_range=(1, 5))),
                   ('tfidf', TfidfTransformer()),
                   ('clf', svm.NuSVC()),
                 ])
   clf = clf.fit(trainX, trainY)
   predictions=clf.predict(testX)
   score = metrics.f1_score(testY, predictions)
   print("NuSVC F1-Score:   %0.3f" % score)                 # F1-Score = 0.834! but maybe it does not overfit as much? Play with parameters

   #finalPredictions = clf.predict(testD)                   # uncomment to save results
   #saveLabels(finalPredictions, 'NuSVC')


def clfLR(data, labels, stwords):

   trainD1=data[0:546]
   testD=data[546:]

   trainX, testX, trainY, testY = train_test_split(trainD1, labels, test_size=0.2, random_state=42)
   clf = Pipeline([('vect', CountVectorizer(analyzer='word', stop_words=stwords, min_df=15, max_df=0.25, ngram_range=(1, 5))),
                   ('tfidf', TfidfTransformer()),
                   ('clf', LogisticRegression()),
                 ])
   clf = clf.fit(trainX, trainY)
   predictions=clf.predict(testX)
   score = metrics.f1_score(testY, predictions)
   print("Logistic Regression F1-Score:   %0.3f" % score)   # F1-Score = 0.879! Play more w/param. Compare overfitting with other classifiers

   #finalPredictions = clf.predict(testD)                   # uncomment to save results
   #saveLabels(finalPredictions, 'LR')


def clfRF(data, labels, stwords):

   trainD1=data[0:546]
   testD=data[546:]

   trainX, testX, trainY, testY = train_test_split(trainD1, labels, test_size=0.2, random_state=42)
   clf = Pipeline([('vect', CountVectorizer(analyzer='word', stop_words=stwords, min_df=10, max_df=0.25, ngram_range=(1, 5))),
                   ('tfidf', TfidfTransformer()),
                   ('clf', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                                max_depth=None, max_features='auto', max_leaf_nodes=None,
                                                min_samples_leaf=2, min_samples_split=2,
                                                min_weight_fraction_leaf=0.0, n_estimators=420, n_jobs=1,
                                                oob_score=False, random_state=None, verbose=0,
                                                warm_start=False)),
                 ])                             # all parameters from another ancillary
   clf = clf.fit(trainX, trainY)
   predictions=clf.predict(testX)
   score = metrics.f1_score(testY, predictions)
   print("Random Forest F1-Score:   %0.3f" % score)         # F1-Score (plain) = 0.991 !!!! Note overfitting, play w/param
                                                            # F1-Score (w/param) = 0.985 !!!

   #finalPredictions = clf.predict(testD)                   # uncomment to save results
   #saveLabels(finalPredictions, 'RF')

def clfAB(data, labels, stwords):

   trainD1=data[0:546]
   testD=data[546:]

   trainX, testX, trainY, testY = train_test_split(trainD1, labels, test_size=0.2, random_state=42)
   clf = Pipeline([('vect', CountVectorizer(analyzer='word', stop_words=stwords, min_df=15, max_df=0.3, ngram_range=(1, 5))),
                   ('tfidf', TfidfTransformer()),
                   ('clf', AdaBoostClassifier()),
                 ])
   clf = clf.fit(trainX, trainY)
   predictions=clf.predict(testX)
   score = metrics.f1_score(testY, predictions)
   print("AdaBoost F1-Score:   %0.3f" % score)              # F1-Score = 0.928 !!! Note overfitting, play w/param

   print("Predicting the remaining lables")
   finalPredictions = clf.predict(testD)                   # uncomment to save results
   saveLabels(finalPredictions, 'AB')


def clfNB(data, labels, stwords):
    trainD1 = data[0:546]
    testD = data[546:]  # still need this for finalPredictions

    trainX, testX, trainY, testY = train_test_split(trainD1, labels, test_size=0.2, random_state=42)
    clf = Pipeline([('vect', TfidfVectorizer(stop_words=stwords, min_df=1, max_df=0.37, ngram_range=(1, 3))),
                    # lowercase=True by default
                    #('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),
                    ])
    clf = clf.fit(trainX, trainY)
    predictions = clf.predict(testX)
    score = metrics.f1_score(testY, predictions)
    print("MultinomialNB F1-Score:   %0.3f" % score)

    #clf = clf.fit(trainD1, labels)
    finalPredictions = clf.predict(testD)  # uncomment to save results
    saveLabels(finalPredictions, 'NB')


if __name__ == '__main__':
   textData = loadText()
   labels = loadLabels()
   additionalData = loadAdditional()
   fullData = combine(textData, additionalData)
   my_stopwords = stopwords()

   clfNB(fullData, labels, my_stopwords)        # the winner w/'english', 1, 0.37, ngrams(1, 3)
   #clfSGD(fullData, labels, my_stopwords)
   #clfLR(fullData, labels, my_stopwords)
   #clfLinSVC(fullData, labels, my_stopwords) #!!!!
   #clfNuSVC(fullData, labels, my_stopwords)
   #clfRF(fullData, labels, my_stopwords)
   #clfAB(fullData, labels, my_stopwords)

# TODO:
# 1) test-train split on 546 data points - this is for me only, but mention in the report. Uncomment an example in clfNB
# 2) 2 or 3 different text representations: BoW, bigram, mix of both, POS (see other ancillary)