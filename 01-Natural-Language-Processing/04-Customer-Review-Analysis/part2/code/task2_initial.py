# There are 147 categories, however there are some categories have very little reviews so I will only use 50 topics
# that have the most reviews

import math
from gensim import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import *
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')                    # required, otherwise matplotlib will use ''Qt5Agg' generating error
import matplotlib.pyplot as plt
from matplotlib import *

from pathlib2 import Path

def vectorizer(contents, idf_usage = False, max_df = 0.7):
    vectorizer = TfidfVectorizer(stop_words='english', use_idf = idf_usage, max_df = max_df)
    matrix = vectorizer.fit_transform(contents)
    id2words ={}
    for i,word in enumerate(vectorizer.get_feature_names()):
        id2words[i] = word
    return matrix, np.vstack([cosine_similarity(val, matrix) for val in matrix]), id2words

#%matplotlib inline

def draw(matrix):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap=cm.coolwarm)
    ax.set_xticks(np.arange(matrix.shape[1]), minor=False)
    ax.set_yticks(np.arange(matrix.shape[0]), minor=False)
    ax.set_xticklabels(cuisines, rotation='vertical')
    ax.set_yticklabels(cuisines)
    fig.set_size_inches(15, 12)
    plt.show()

def task2_1():

    # task 2.1
    # Visualization = word frequencies (TF) + cosine distance to measure differences between cuisines
    # max_df = 0.7 to ignore the most common words
    # Every category has its own typical words and the similarity is small as word frequencies are different. Therefore,
    # only very similar cuisines will appear close (color red), e.g. Italian & Pizza or Traditional & American New
    # TODO: normalize text more in vectorizer?

    matrix_if, similarity_if, _ = vectorizer(contents)
    print("similarity matrix obtained")
    draw(similarity_if)

def task2_2():

    # task 2.2
    # Visualization = TF / IDF + cosine distance to measure differences between cuisines
    # max_df = 0.7 to ignore the most common words
    # Results similar to task 2.1 probably because exept for some very category-unique words, other words
    # often appear in many categories, so IF_IDF = IF (this contradicts the statement for task 2.1?)
    # TODO: normalize text more in vectorizer?

    matrix_ifidf, similarity_ifidf, id2words = vectorizer(contents, True)
    print("similarity matrix obtained")
    draw(similarity_ifidf)
    return matrix_ifidf, id2words


def draw_ntopic(numTopics):

    # note: matrix_ifidf, id2words used here are global var from main body of program, received after task2_2()
    corpus = matutils.Sparse2Corpus(matrix_ifidf, documents_columns=False)

    lda = models.LdaMulticore(corpus, num_topics=numTopics, id2word=id2words)

    topic_pro = np.zeros((len(cuisines), numTopics))

    idx = 0
    for val in lda.get_document_topics(corpus, minimum_probability=-1):
        for jdx, pro in val:
            topic_pro[idx][jdx] = pro
        idx += 1

    print "Number of topic:", numTopics
    draw(np.vstack([cosine_similarity(val.reshape(1, -1), topic_pro) for val in topic_pro]))

def task2_3():

    #task 2.3
    # use LDA to cluster, assign topics to categories;
    # get vector representation of each category (using the probability that every category belongs to every topic);
    # then, use cosine distance to measure differences between cuisines;
    # number of topics tested is 2, 5, and 10.
    # Evidently, 2 topics is too few - almost all categories are similar (cannot learn much from this visualization)
    # 5 topics is good as we can see which categories are similar (e.g. Asian Fusion & Chinese, Japanese, Thai, Vietnames)
    # 10 topics - maybe the map shows characteristics of individual cuisines (American(New) & Buffet -
    # means there are many American restaurants that offer Buffet (don't think so, just similar cuisines)

    for numTopics in [2, 5, 10]:
        draw_ntopic(numTopics)

allfiles = os.listdir('categories')
cuisines = [f.replace('.txt','') for f in allfiles]
contents = [Path('categories/'+f).read_text(encoding="utf8")
            .replace('\n',' ') for f in allfiles]
print("contents read")

#task2_1()
matrix_ifidf, id2words = task2_2()
task2_3()