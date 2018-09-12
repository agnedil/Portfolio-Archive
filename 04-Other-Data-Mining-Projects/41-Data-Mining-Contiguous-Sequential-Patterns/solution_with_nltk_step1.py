# finds raw FreqDist frequencies for unigrams, bigrams, trigrams, 4-grams, and 5-grams in the original text file (yelp reviews, one review per line)
# and writes them to intermediate file for further processing at step 2
# https://stackoverflow.com/questions/40669141/python-nltk-counting-word-and-phrase-frequency !!!!
# https://stackoverflow.com/questions/14364762/counting-n-gram-frequency-in-python-nltk
# https://github.com/tistre/nltk-examples/blob/master/freqdist_top_words.py

import nltk
import sys
from nltk.collocations import BigramCollocationFinder as bc
from nltk import ngrams, FreqDist, word_tokenize

myfile = open("reviews_sample.txt").read()

data = word_tokenize(myfile)

all_counts = dict()
for size in 2, 3, 4, 5:                                        #BRILLIANT!!!! Dictionary of ngrams / frequencies
    all_counts[size] = FreqDist(ngrams(data, size))            #as values and ngram-sizes as keys

f2 = open("patterns2_nltk.txt", "w")
for phrase, frequency in all_counts[2].most_common():                           #bigrams
    if frequency >= 100:
        f2.write('{}:{};{}\n'.format(frequency, phrase[0], phrase[1]))
f2.close()

f3 = open("patterns3_nltk.txt", "w")
for phrase, frequency in all_counts[3].most_common():                           #trigrams
    f3.write('{}:{};{};{}\n'.format(frequency, phrase[0], phrase[1], phrase[2]))
f3.close()

f4 = open("patterns4_nltk.txt", "w")                                            #4-grams
for phrase, frequency in all_counts[4].most_common():
    f4.write('{}:{};{};{};{}\n'.format(frequency, phrase[0], phrase[1], phrase[2], phrase[3]))
f4.close()

f5 = open("patterns5_nltk.txt", "w")                                            #5-grams
for phrase, frequency in all_counts[5].most_common():
    f5.write('{}:{};{};{};{};{}\n'.format(frequency, phrase[0], phrase[1], phrase[2], phrase[3], phrase[4]))
f5.close()

print("Done!")

#OTHER USEFUL NLTK TWEAKS

#prepare words
#myfile = open("reviews_sample.txt").read()
#words = nltk.word_tokenize(myfile)

#calculate frequency distribution of unigrams
#fdist = nltk.FreqDist(words)

#write to file from FrewDist object
#f4 = open("patterns.txt", "w")
#for word, frequency in fdist.most_common():
#    if frequency >= 100:
#        f4.write(u'{}:{}\n'.format(frequency, word))
#f4.close()

#finder = BigramCollocationFinder.from_words(words)    #BigramCollocationFinder should be imported

#write to file from BigramCollocationFinder object
#f4 = open("patterns.txt", "w")
#for phrase, frequency in finder.ngram_fd.items():
#    if frequency >= 100:
#        f4.write(u'{}:{};{}\n'.format(frequency, phrase[0], phrase[1]))
#f4.close()

#print("Done!")

#Remove single-character tokens (mostly punctuation)
#words = [word for word in words if len(word) > 1]

#Remove numbers
#words = [word for word in words if not word.isnumeric()]

#Lowercase all words (default_stopwords are lowercase too)
#words = [word.lower() for word in words]

#Stemming words seems to make matters worse, disabled
#stemmer = nltk.stem.snowball.SnowballStemmer('german')
#words = [stemmer.stem(word) for word in words]

#Remove stopwords
#words = [word for word in words if word not in all_stopwords]

#another way to create bigrams
#bigrm = nltk.bigrams(words)
