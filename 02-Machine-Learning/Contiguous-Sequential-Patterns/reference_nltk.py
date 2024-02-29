# testing nltk

# myfile = open("reviews_sample.txt", "r")

import sys
import codecs
import nltk
from nltk.collocations import BigramCollocationFinder as bc

myfile = open("reviews_sample.txt").read()

words = nltk.word_tokenize(myfile)

# Calculate frequency distribution of unigrams
fdist = nltk.FreqDist(words)

f4 = open("patterns.txt", "w")
for word, frequency in fdist.most_common():
    if frequency >= 100:
        f4.write(u'{}:{}\n'.format(frequency, word))
f4.close()
"""
finder = bc.from_words(words)    #this uses BigramCollocationFinder (see the import statement)

f4 = open("patterns.txt", "w")
for phrase, frequency in finder.ngram_fd.items():
    if frequency >= 100:
        f4.write(u'{}:{};{}\n'.format(frequency, phrase[0], phrase[1]))
f4.close()
"""
print("Done!")

# Remove single-character tokens (mostly punctuation)
#words = [word for word in words if len(word) > 1]

# Remove numbers
#words = [word for word in words if not word.isnumeric()]

# Lowercase all words (default_stopwords are lowercase too)
#words = [word.lower() for word in words]

# Stemming words seems to make matters worse, disabled
# stemmer = nltk.stem.snowball.SnowballStemmer('german')
# words = [stemmer.stem(word) for word in words]

# Remove stopwords
#words = [word for word in words if word not in all_stopwords]

#bigrm = nltk.bigrams(words)
