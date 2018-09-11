from mnist import MNIST
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import neighbors
import time

def gaussian_nb(X_train, y_train, X_test, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    score_gnb = gnb.score(X_test, y_test)
    print 'Score gnb: ' + str(score_gnb)

def multinomial_nb(X_train, y_train, X_test, y_test, iterations=10):
    alpha_values = [0.0000000001, 0.000001, 0.0001, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]

    for a in alpha_values:
        mnb = MultinomialNB(alpha=a)
        mnb.fit(X_train, y_train)
        score = mnb.score(X_test, y_test)
        print('Score mnb(alpha='+str(a)+'): ' + str(score))

def bernoulli_nb(X_train, y_train, X_test, y_test):
    alpha_values = [0.0000000001, 0.000001, 0.0001, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]

    for a in alpha_values:
        bnb = BernoulliNB(alpha=a)
        bnb.fit(X_train, y_train)
        score = bnb.score(X_test, y_test)
        print('Score bnb(alpha='+str(a)+'): ' + str(score))

def knn(X_train, y_train, X_test, y_test):
    ns = [2, 3, 5, 10, 20, 50, 100]
    p=[1, 2, 3]
    metric=['euclidean', 'manhattan', 'minkowski']
    for w in ['uniform', 'distance']:
        for p_value in p:
            for m in metric:
                for n in ns:
                    start = time.time()
                    clf = neighbors.KNeighborsClassifier(n, weights=w, n_jobs=-1, p=p_value, metric=m)
                    clf.fit(X_train, y_train)
                    score = clf.score(X_test, y_test)
                    t = time.time() - start
                    print('Score KNN(w='+w+', n='+str(n)+', p='+str(p_value)+', metric='+m+'): ' + str(score) + ' (in '+ str(t/60.0) +' minutes)')


def main():
    start = time.time()
    print 'Loading training data...'
    mndata = MNIST('mnist')
    X, y = mndata.load_training()
    X_train = X
    y_train = list(y)
    print 'Loading testing data...'
    X, y = mndata.load_testing()
    X_test = X
    y_test = list(y)

    print '-> Gaussian Naive Bayes'
    gaussian_nb(X_train, y_train, X_test, y_test)
    print '-> Multinomial Naive Bayes'
    multinomial_nb(X_train, y_train, X_test, y_test)
    print '-> Bernoulli Naive Bayes'
    bernoulli_nb(X_train, y_train, X_test, y_test)
    print '-> KNN'
    knn(X_train, y_train, X_test, y_test)

    t = time.time() - start
    print 'Finished in: ' + str(t/60.0) + ' minutes'

if __name__ == '__main__':
   main()
