# used code from https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/deepir.ipynb

import matplotlib
matplotlib.rcParams['backend'] = 'Qt4Agg'
matplotlib.rcParams['backend.qt4'] = 'PyQt4'
import logging
import re
from gensim.models import Word2Vec, Phrases
import multiprocessing
import pandas as pd                                         # for quick summing within doc
import numpy as np
import nltk
import string
from collections import Counter
from stop_words import safe_get_stop_words


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

stopwords = list(safe_get_stop_words('en')) + list(string.punctuation) + list(nltk.corpus.stopwords.words('english'))
print("stopwords created")
print(stopwords)

# load and preprocess Italian cuisine reviews
reviews = pd.read_csv('Italian.txt',sep='\n\n',encoding='utf-8',header=None, engine='python').as_matrix().tolist()
print("reviews loaded")

reviewSents = []
for review in reviews:
    out = re.sub('[^a-z]+', ' ', review[0].decode('utf-8').lower())
    sentence = [i for i in nltk.word_tokenize(out) if i not in stopwords]
    reviewSents.append(sentence)
print("stopwords removed from reviews")

# load and preprocess quality Italian cuisine phrases
qualityPhrases = pd.read_csv('00_complete_Italian_dishes.txt',sep='\n\n',encoding='utf-8',header=None, engine='python').as_matrix().tolist()
print("quality phrases loaded")

qualitySents = []
for phrase in qualityPhrases:
    out = re.sub('[^a-z]+', ' ', phrase[0].decode('utf-8').lower())
    sentence = [i for i in nltk.word_tokenize(out) if i not in stopwords]
    qualitySents.append(sentence)
print("stopwords removed from quality phrases")

# mine lists of phrases
reviewPhrases = Phrases(reviewSents)
qualityPhrases = Phrases(qualitySents)
print("two Phrase models applied")

# list of phrases as lists of words
ItCuisPhrases = []
for key in reviewPhrases.vocab.keys():
    onePhrase = key.split("_")
    if len(onePhrase) > 1:
        ItCuisPhrases.append(onePhrase)
print("list of phrases as lists of words done")

#reviewModel = Word2Vec(reviewPhrases[reviewSents], workers=multiprocessing.cpu_count(), iter=3, hs=1, negative=0, size=200, seed = 317)
qualityModel = Word2Vec(qualityPhrases[qualitySents], workers=multiprocessing.cpu_count(), iter=3, hs=1, negative=0)
print ("word2vec model is learned")

# inversion of the distributed representations
# we have 5 different word2vec language representations. Each 'model' has been trained conditional
# (i.e., limited to) text from a specific star rating. Apply Bayes rule to go from p(text|stars) to p(stars|text)
# For any new sentence we can obtain its likelihood (lhd; actually, the composite likelihood approximation;
# see the paper) using the score function in the word2vec class. We get the likelihood for each sentence
#  in the first test review, then convert to a probability over star ratings. Every sentence in the review is
# evaluated separately and the final star rating of the review is an average vote of all the sentences

"""
docprob takes two lists
* docs: a list of documents, each of which is a list of sentences
* models: the candidate word2vec models (each potential class)

it returns the array of class probabilities.  Everything is done in-memory.
"""

#def docprob(myphrase, mods):
    # score() takes a sentences here
    #myphrases = [ph for ph in doc]
    # the log likelihood of myphrases under w2v
    #llhd = mods.score(myphrase, 1)
    # now exponentiate to get likelihoods,
    #lhd = np.exp(llhd) # - llhd.max(axis=0)) # subtract row max to avoid numeric overload
    # normalize across models (stars) to get sentence-star probabilities
    #prob = pd.DataFrame( (lhd/lhd.sum(axis=0)).transpose() )
    # and finally average the sentence probabilities to get the review probability
    #prob["doc"] = [i for i,d in enumerate(docs) for s in d]
    #prob = prob.groupby("doc").mean()
    #return llhd

# get the probs (note we give docprob a list of lists of words, plus the models)
logprobs = []
probs = []
for i in range(100):
    logprob = qualityModel.score(ItCuisPhrases[i], 1)[0]
    prob = np.exp(logprob)[0]
    print(logprob, " : ", prob)
    logprobs.append( logprob )
    probs.append( prob )
