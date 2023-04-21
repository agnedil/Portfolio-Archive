import io
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.classifier import StackingClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced
from sklearn import model_selection

path=""
ftext="dataset/hygiene.dat"
fdatLables="dataset/hygiene.dat.fullLabels"
finLabel="labels.txt"
fadd="dataset/hygiene.dat.additional"
stopwordsFile="stopwords_3000_google.txt"


def loadText():

   txt=[]
   with open(path+ftext,'r') as f:
      for line in f.readlines():
         txt.append(line)
   print("Length of text: ")
   print len(txt)
   return txt


def loadLabels():
   lab=[]
   i=0
   with open(path+fdatLables,'r') as f:
      for line in f.readlines():
         #if i==546: break
         lab.append(int(line))
         i+=1
   print("Length of labels: ")
   print len(lab)
   return lab


def loadAdditional():

   add=[]
   with open(path+fadd,'r') as f:
      for line in f.readlines():
         add.append(line)
       
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
    with open(path + stopwordsFile,'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.lower()
            if line == 'a' or len(line) > 1:
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

def clfNB(data, labels):
    #trainD1 = data[0:546]
    testFinal = data[546:]  # still need this for finalPredictions

    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2) #random_state=21)

    clf1 = MultinomialNB()
    #clf2 = svm.LinearSVC()
    clf2 = xgb.XGBClassifier(n_estimators=500, objective='binary:logistic', eval_metric='auc', eta=0.1, max_depth=6, subsample=1, colsample_bytree=0.3, silent=1)
    mc = MultinomialNB()

    clf = make_pipeline_imb( CountVectorizer(stop_words='english', min_df=1, max_df=0.5, ngram_range=(1, 3)),
                             #TfidfVectorizer(stop_words='english', min_df=1, max_df=0.37, ngram_range=(1, 3)),
                             TfidfTransformer(),
                             RandomUnderSampler(),
                             StackingClassifier(classifiers=[clf1, clf2],
                                                meta_classifier=mc))

    clf = clf.fit(trainX, trainY)
    predictions = clf.predict(testX)
    score = metrics.f1_score(testY, predictions)
    print("SCLF F1-Score:   %0.3f" % score)

    finalPredictions = clf.predict(testFinal)  # uncomment to save results
    saveLabels(finalPredictions, 'SCLF')

if __name__ == '__main__':
    textData = loadText()
    labels = loadLabels()
    additionalData = loadAdditional()
    fullData = combine(textData, additionalData)
    #my_stopwords = stopwords()
    clfNB(fullData, labels)
