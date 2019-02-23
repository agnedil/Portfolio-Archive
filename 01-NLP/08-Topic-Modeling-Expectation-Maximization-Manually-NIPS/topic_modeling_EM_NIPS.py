# AML UIUC
# Expectation Maximization Algorithm (from scratch)
# using the NIPS dataset https://archive.ics.uci.edu/ml/datasets/Bag+of+Words, produce a graph of probabilities for 30 topics
# and a table of top 10 words in each topic

# hints from https://stackoverflow.com/questions/8550912/python-dictionary-of-dictionaries
# hints from https://stackoverflow.com/questions/16333296/how-do-you-create-nested-dict-in-python

import sys, math, random, copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.spatial import distance
from scipy.cluster.vq import kmeans
from numpy.linalg import inv


def Q(x, numClust, mu, pi, w):                                                  # calculate Q

    sigma = 0
    for j in xrange(numClust):                                                  # xrange more memory-efficient than range
        inner_prod = x*np.log(mu[j])
        sum = inner_prod + math.log(pi[j])
        sigma += sum*w[:,j,np.newaxis]
    return np.sum(sigma)


def expect(x, numClust, mu, pi):                                                # E step, log space

    logA = np.zeros((x.shape[0], numClust))
    for j in xrange(numClust):
        sigma = x*np.log(mu[j])
        logA[:,j] = np.log(pi[j]) + np.sum(sigma, axis=1)
    logAMax = np.zeros((x.shape[0],))
    logA.max(axis=1, out=logAMax)
    sum=0
    for j in xrange(numClust):
        sum += np.exp(logA[:,j] - logAMax)
    term3 = np.log(sum)
    logY = np.zeros((x.shape[0], numClust))
    for j in xrange(numClust):
        logY[:,j] = logA[:,j] - logAMax - term3
    y = np.exp(logY)
    w = y
    return w


def maximize(x, w, numClust):                                                   # M step

    newMu = np.zeros((numClust, x.shape[1]))
    newPi = np.zeros((numClust))
    for j in xrange(numClust):
        denominator = np.sum(np.sum(x, axis=1)*w[:,j])
        numerator = np.sum(x*w[:,j,np.newaxis], axis=0)
        newMu[j] = numerator/denominator
        newPi[j] = np.sum(w[:,j])/1500
    new_newMu = np.zeros((numClust, x.shape[1]))
    for k in xrange(numClust):
        new_newMu[k] = (newMu[k]+.0001)/(np.sum(newMu[k])+newMu.shape[1]/10000)
    return new_newMu, newPi


if __name__ == "__main__":

    # LOAD DATA, SET INITIAL PARAMETERS (SOME OF THEM RANDOMLY)
    input = np.loadtxt('docword.nips.txt', skiprows=3)                      # load data, skip three-line header
    print("input data loaded successfully")                                 # checkpoint
    print

    numClust = 30                                                           # num of clusters
    numWord  = 12419                                                        # from docword.nips.txt header, alternatively: np.max(input[:,1])
    numDoc   = 1500                                                         # from docword.nips.txt header, or the dataset website

    x = np.zeros((numDoc, numWord))                                         # matrix of word counts for all words in all docs
    for entry in input:
        x[entry[0]-1][entry[1]-1] = entry[2]                                # assign wordcount to x(ij), "-1" as numbers in the file start from 1 and not 0

    mu = np.zeros((numClust, x.shape[1]))                                   # random assign of all mu, note double parenthesis syntax
    for j in xrange(numClust):
       mu[j] = np.random.rand(x.shape[1])                                   # np.random.rand = random values in a given shape

    pi = np.full(numClust, 1.0/numClust)                                    # initial probab for each topic/cl, np.full=newarray(shape, value of each element)
    difference = 1                                                          # parameter to be checked for convergence
    q = 0                                                                   # initialize Q
    iter = 0

    # ITERATE EXPECTATION MAXIMIZATION ALGORITHM, RECORD RESULTS
    while iter < 100 and difference > 0.0001:                               # per Piazza post @1218 (follow-ups from TA): convergence criterion = 0.0001
        iter += 1                                                           # num iterations = 100,
        print('iteration ' + str(iter))
        w = expect(x, numClust, mu, pi)
        mu, pi = maximize(x, w, numClust)
        oldq = q
        q = Q(x, numClust, mu, pi, w)                                       # calculate new Q
        difference = abs(q-oldq)/abs(q)                                     # difference between iterations as convergence criterion
        if iter != 1:
            print "difference between iterations = " + str(difference)

    result = {}                                                             # store final results
    result['clusterCentroids'] = {}                                         # store centroids for each of 30 clusters
    result['parameters'] = {}
    for i in xrange(numClust):
        result['parameters'][i] = {}                                        # store mu and pi for each of 30 clusters
        result['parameters'][i]['pi'] = pi[i]
        result['parameters'][i]['mu'] = mu[i]

    for index, wordsinDoc in enumerate(x):
        distances = np.zeros((numClust,))                                   # initialize distance array, dtype=int by default due to ","
        for j in xrange(numClust):
            distances[j] = distance.euclidean(wordsinDoc, mu[j])            # find euclidean distances
        cluster = np.argmin(distances)                                      # find min. distance, np.argmin returns index of min value in distance
        if cluster not in result['clusterCentroids']:                       # NOTE: cluster is index!
            result['clusterCentroids'][cluster]=[]
        result['clusterCentroids'][cluster].append(index)

    # TOP 10 WORDS
    print
    print("top 10 words in each topic:")
    vocab = [line.strip() for line in open("vocab.nips.txt")]               # list of all words from vocabulary file
    csv = open('top10words.csv', 'w+')                                      # file to write the top 10 results to

    for idx in xrange(numClust):
        top10Index = result['parameters'][idx]['mu'].argsort()[-10:][::-1]  # argsort() returns indices that would sort array, default=quicksort
                                                                            # get last 10 indices from last mu
        top10Words = [vocab[index] for index in top10Index]                 # list of top 10 words for vocab

        startLine = 'Topic ' + str(idx+1)
        print startLine+":",                                                # this syntax allows to continue print on the same line - Python 2.7 (use print(startLine, end=" ") for Python 3
        csv.write(startLine+",")
        for word in top10Words[:-1]:
            toPrint = word + ","
            print toPrint,                                                  # print top 10 words on screen
            csv.write(toPrint)                                              # write top 10 words to csv file to insert into report
        print(top10Words[-1])                                               # last element at the end of line w/out a comma
        csv.write(top10Words[-1] + "\n")
    csv.close()

    # PLOT RESULTS
    plt.rcdefaults()
    topicsProbab = np.zeros(numClust)                                       # initialize array of topic/cluster probabilities for plotting
    for i in xrange(numClust):                                              # populate array
        topicsProbab[i]=result['parameters'][i]['pi']

    myX = range(1,31)                                                       # range for x axis
    fig, axis= plt.subplots()                                               # create bar chart grid
    axis.set_xlim([0,31])
    axis.set_ylim([0,np.max(topicsProbab)+.01])

    plt.bar(myX, topicsProbab, align='center', alpha=.5)                    # plot bar chart of probabilities
    plt.ylabel('Probablities')
    plt.xlabel('Topics')
    plt.title('Probability of selecting a topic')
    plt.show()

# TODO: create a separate EM function to be used in other files
