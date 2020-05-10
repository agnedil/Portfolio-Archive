import math
import json
import pickle
import random
from gensim import models
from gensim import matutils
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from nltk.tokenize import sent_tokenize
import glob
import argparse
import os
from sys import exit
path2files="yelp_dataset_challenge_academic_dataset/"
path2buisness=path2files+"yelp_academic_dataset_business.json"
path2reviews=path2files+"yelp_academic_dataset_review.json"


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
K_clusters = 10
vectorizer = TfidfVectorizer(max_df=0.7, #max_features=10000,  ####
                             min_df=2, stop_words='english',
                             use_idf=True)

if not os.path.isdir("categories"):
    print "you need to generate the cuisines files 'categories' folder first"
    exit(0)

text = []
c_names = []
cat_list = glob.glob("categories/*")
cat_size = len(cat_list)
if cat_size < 1:
    print "you need to generate the cuisines files 'categories' folder first"
    exit(0)

sample_size = min(100, cat_size)
cat_sample = sorted(random.sample(range(cat_size), sample_size))
# print (cat_sample)
count = 0
for i, item in enumerate(cat_list):
    if i == cat_sample[count]:
        li = item.split('/')
        cuisine_name = li[-1]
        c_names.append(cuisine_name[:-4].replace("_", " "))
        with open(item) as f:
            text.append(f.read().replace("\n", " "))
        count = count + 1

    if count >= len(cat_sample):
        print "generating cuisine matrix with:", count, "cuisines"
        break

if len(text) < 1:
    print "the 'categories' folder does not contain any cuisines. Run this program using the '--cuisine' option"
t0 = time()
print("Extracting features from the training dataset using a sparse vectorizer")
X = vectorizer.fit_transform(text)
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)

corpus = matutils.Sparse2Corpus(X, documents_columns=False)
lda = models.ldamodel.LdaModel(corpus, num_topics=100, passes=10, iterations=100)  ####

doc_topics = lda.get_document_topics(corpus)
cuisine_matrix = []  # similarity of topics
# computing cosine similarity matrix
for i, doc_a in enumerate(doc_topics):
    # print (i)
    sim_vecs = []
    for j, doc_b in enumerate(doc_topics):
        w_sum = 0
        if (i <= j):
            norm_a = 0
            norm_b = 0

            for (my_topic_b, weight_b) in doc_b:
                norm_b = norm_b + weight_b * weight_b

            for (my_topic_a, weight_a) in doc_a:
                norm_a = norm_a + weight_a * weight_a
                for (my_topic_b, weight_b) in doc_b:
                    if (my_topic_a == my_topic_b):
                        w_sum = w_sum + weight_a * weight_b

            norm_a = math.sqrt(norm_a)
            norm_b = math.sqrt(norm_b)
            denom = (float)(norm_a * norm_b)
            if denom < 0.0001:
                w_sum = 0
            else:
                w_sum = w_sum / (denom)
        else:
            w_sum = cuisine_matrix[j][i]
        sim_vecs.append(w_sum)

    cuisine_matrix.append(sim_vecs)

with open('cuisine_sim_matrix.csv', 'w') as f:
    for i_list in cuisine_matrix:
        s = ""
        my_max = max(i_list)
        for tt in i_list:
            s = s + str(tt / my_max) + " "
        s = s.strip()
        f.write(",".join(s.split()) + "\n")  # should the list be converted to m

with open('cuisine_indices.txt', 'w') as f:
    f.write("\n".join(c_names))

print("done!")