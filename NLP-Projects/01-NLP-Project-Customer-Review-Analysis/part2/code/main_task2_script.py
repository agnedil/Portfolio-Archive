# uses http://brandonrose.org/clustering

from gensim import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import *
#import pandas as pd
import numpy as np
import matplotlib
from matplotlib import *
from matplotlib import cm
matplotlib.use('Qt4Agg')                    # required, otherwise matplotlib will use ''Qt5Agg' generating error
import matplotlib.pyplot as plt
from stop_words import safe_get_stop_words
import string
from pathlib2 import Path
import nltk
import re
#from numpy import array
#from sklearn.cluster import KMeans

# tfidfvectorizer of raw content: each cuisine's reviews are concatenated into one string
def vectorizer(contents, idf_usage = True, max_df = 0.7):
    vectorizer = TfidfVectorizer(stop_words='english', use_idf = True, max_df = 0.7, min_df=2)
    matrix = vectorizer.fit_transform(contents)
    id2words ={}
    for i,word in enumerate(vectorizer.get_feature_names()):
        id2words[i] = word
    return matrix, np.vstack([cosine_similarity(val, matrix) for val in matrix]), id2words

#%matplotlib inline

# draws a similarity matrix
def draw(matrix):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap=cm.coolwarm)
    ax.set_xticks(np.arange(matrix.shape[1]), minor=False)
    ax.set_yticks(np.arange(matrix.shape[0]), minor=False)
    ax.set_xticklabels(cuisines, rotation='vertical')
    ax.set_yticklabels(cuisines)
    cbar = fig.colorbar(cax)
    fig.set_size_inches(31, 31)
    plt.show()

def task2_1():

    matrix_if, similarity_if, _ = vectorizer(contents)
    print("similarity matrix obtained")
    #draw(similarity_if)

def task2_2():

    matrix_ifidf, similarity_ifidf, id2words = vectorizer(contents, True)
    print("similarity matrix obtained")
    #draw(similarity_ifidf)
    return matrix_ifidf, id2words

# run LDA model with numTopics, then draw similarity matrix
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

    for numTopics in [4, 7, 15, 30]:
        draw_ntopic(numTopics)


#stopwords = list(safe_get_stop_words('en')) + list(string.punctuation) + list(nltk.corpus.stopwords.words('english'))
allfiles = os.listdir('categories')
cuisines = [f.replace('.txt','') for f in allfiles]
contents = [Path('categories/'+f).read_text(encoding="utf8")
            .replace('\n',' ') for f in allfiles]
print("contents read")

# removing non-words and using expanded list of stopwords from Task 3 proved inefficient + take a lot of time
# uncomment the three lines below to turn on different tasks selectively (was not used)
#task2_1()
#matrix_ifidf, id2words = task2_2()
#task2_3()

# code below was used to run the vecotrizer with different setting, such as noIDF, IDF, min_df, max_df for Tasks 2.1 and 2.2
# resulting similarity matrices were them visulized using draw()
# the matrix_if was later used for kmeans, and similarity_if for the agglomerative clustering and kmeans for Task 2.3

matrix_if, similarity_if, _ = vectorizer(contents)   #contents
print("similarity matrix obtained")
draw(similarity_if)
cuisines = [item.replace("_", " ") for item in cuisines] #!!!!

# save similarity matrix to file with cuisines as header, add column with cuisines to left, and use R script
# cormat_visualize.r to visualize each matrix
cuisines_header = ",".join(cuisines)
numpy.savetxt('2_2_noIDF.csv', similarity_if, delimiter=',', newline='\n', fmt='%.10f', header=cuisines_header)

# WARD CLUSTERING
dist = 1 - similarity_if
from scipy.cluster.hierarchy import ward, dendrogram
linkage_matrix = ward(dist)
fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=cuisines);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters


# KMEANS CLUSTERING
from sklearn.cluster import KMeans

num_clusters = 5
km = KMeans(n_clusters=num_clusters).fit(matrix_if)
clusters = km.labels_.tolist()

import os
import matplotlib as mpl
from sklearn.manifold import MDS
import pandas as pd

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#68a61e'}

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=cuisines))

cluster_names = {0: 'cluster 1',
                 1: 'cluster 2',
                 2: 'cluster 3',
                 3: 'cluster 4',
                 4: 'cluster 5'}
#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(19, 10)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')

ax.legend(numpoints=1)  # show legend with only 1 point

# add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

plt.show()  # show the plot

# uncomment the below to save the plot if need be
# plt.savefig('clusters_small_noaxes.png', dpi=200)