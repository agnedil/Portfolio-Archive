## Comprehensive Analysis of Customer Reviews with Python and R Using State-of-the-Art NLP Methods

Note: The code in this repository uses this dataset: https://www.kaggle.com/yelp-dataset/yelp-dataset
      except for Part 6 for which the dataset is included directly in the respective folder.


## Part 1
Topic modeling using gensim's LDA and PLSA (as scikit-learn's NMF with Kullback-Leibler divergence) for all reviews and for positive vs. negative reviews; visualization of results using graph-tool and wordclouds.
TODO: try tree-node diagram from D3.js instead of graph-tool.

Other major libraries used: NLTK, graph_tool, wordcloud, json, pickle

## Part 2
Data clustering and similarity matrix construction using TfidfVectorizer with IDF (with different min_df, max_df), gensim’s LDA, Ward hierarchical clustering, and scikit-learn's K-means clustering with visualization using matplotlib (Python) and ggplot2 and reshape2 (R language)

Other major libraries used: numpy, pandas

## Part 3
Dataset mining to discover common information using manual annotation, TopMine, SegPhrase / AutoPhrase with different annotation options, Word2Vec (similar_by_word function)

Other major libraries used: pattern.en, matplotlib.pyplot, gensim, collections (Counter), pandas, nltk, wordcloud, stop_words

## Part 4 and 5
Data-driven decisions: ranking of dishes and restaurants for their recommendation with the visualization of results in Tableau. I came up with my own ranking function which was further improved in Part 7. TextBlob used for sentiment analysis

Other major libraries used: numpy, collections (defaultdict)

## Part 6
Classification of restaurant reviews to predict the hygiene conditions and whether they will pass inspection using scikit-learn's CountVectorizer and TfidfTransformer for feature extraction, Naive Bays (multinomialNB), SGD, Logistic Regression, Linear SVC and NuSVC, Random Forest, AdaBoost as classifiers. Additional use of the xgboost module, StackingClassifier from the mlxtend module, RandomUnderSampler and RandomOverSampler from the imblearn module to account for the imbalanced dataset.

Comparison of stemming vs. lemmatization for feature extraction. Comparison of algorithm performance with unigrams, bigrams, trigrams, up to 5-grams. Experimenting with different stopword lists. First place on the leaderboard out of 12 participants!

Other major libraries used: NLTK

## Part 7
Web application using Flask based on the results from previous stages with preprocessing of a large dataset to be able to provide users with instantaneous dish and restaurant recommendations. Implemented with 5 dishes from each of 5 cuisines in the interest of time.

More cuisines and dishes can be easily added. If more reviews are preprocessed, those can be added seamlessly in real time. Visualization using complex D3.js library from JavaScript.

For a limited time, the webapp is available at http://agnedil.pythonanywhere.com/

Other major libraries used: collections (defaultdict), TextBlob, numpy
