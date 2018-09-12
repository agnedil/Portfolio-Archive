# counting frequencies of any ngrams in a gi en text (huge file with yelp reviews)
# extremely fast, faster than nltk
# https://stackoverflow.com/questions/12488722/counting-bigrams-pair-of-two-words-in-a-file-using-python

import re
from itertools import islice, izip, tee
from collections import Counter

#get all the individual words from the original text file into variable "words"
words = re.findall('\w+', open('reviews_sample.txt').read())

#UNCOMMENT EACH SECTION TO SEE HOW IT PERFORMS
#find all unigrams and print those with frequency >= 100
#count = Counter(words)
#for item, freq in count.items():
#   if freq >= 100:
#       print(str(freq) + ":" + item)

#find all bigrams and print those with frequency >= 100
#count2 = Counter(izip(words, islice(words, 1, None)))
#for item, freq in count2.items():
#    if freq >= 100:
#        print(str(freq) + ":" + item[0] + ";" +item[1])

#get the frequency of any n-gram - replace number in the line with Counter to any n for ngram
#def ngrams(lst, n):
#  tlst = lst
#  while True:
#    a, b = tee(tlst)
#    l = tuple(islice(a, n))
#    if len(l) == n:
#      yield l
#      next(b)
#      tlst = b
#    else:
#      break

#count3 = Counter(ngrams(words, 3))
#for item, freq in count3.items():
#    if freq >= 2:
#        toPrint = ';'.join(item)
#        lineToPrint = str(freq) + ':' + toPrint
#        print(lineToPrint)
