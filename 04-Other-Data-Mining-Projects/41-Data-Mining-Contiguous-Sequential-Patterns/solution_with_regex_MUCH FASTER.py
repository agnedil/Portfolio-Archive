# Objective - find all ngrams with absolute support >= 100 in a huge file with yelp reviews (each review = one line)
# first, regex counts overall frequencies and prunes those below 100
# then, using regex, the support of each ngram is counted (counts once per review); output=exactly nltk
# much faster than the nltk version - about 15 min vs. an hour or so
# https://stackoverflow.com/questions/12488722/counting-bigrams-pair-of-two-words-in-a-file-using-python

import re
from itertools import islice, izip
from collections import Counter

#get all unigrams
words = re.findall('\w+', open('reviews_sample.txt').read())

#find all unigrams, freq >= 100 (total frequency, not support meaning may occur more than once in a review)
mylist = []
countuni = Counter(words)
for item, freq in countuni.items():
   if freq >= 100:
       mylist.append(item)

#find all bigrams, freq >= 100 (same comment as above)
countuni2 = Counter(izip(words, islice(words, 1, None)))
for item, freq in countuni2.items():
    if freq >= 100:
        mylist.append(" ".join(item))

#load original yelp file as list of strings (each string = one review)
listLines = []
with open("reviews_sample.txt") as myfile:
    for line in myfile:
        line = line.strip()
        listLines.append(line)

#count the support of each unigram and bigram - only ONE occurrence will be counted per string
finalCount = []
for phrase in mylist:
    if " " in phrase:                       #I know there are only bigrams and unigrams; otherwise this needs improvement
        count = 0                                                   #countint bigrams with support >= 100
        for line in listLines:
            temp = re.findall('\w+', line)
            c = zip(temp, islice(temp, 1, None))                    #zip - output=list; izip - iterator
            for i in range(len(c)):
                phrase1 = " ".join(c[i])
                if phrase == phrase1:
                    count += 1
                    break
        if count >= 100: print('{}:{}'.format(count, phrase))       #dubugging
    else:                                                           #counting unigrams with support >= 100
        count = 0
        for line in listLines:
            temp = re.findall('\w+', line)
            if phrase in temp:
                count += 1
        if count >= 100: print('{}:{}'.format(count, phrase))       #dubugging
        mylist = [count, phrase]
        finalCount.append(mylist)

#sort and write to file
finalCount = sorted(finalCount, key=lambda x: int(x[0]), reverse=True)
myfile2 = open("patterns.txt", "w")
for item in finalCount:
    if item[0] >= 100:
        wordParts = item[1].split(' ')
        wordToPrint = ';'.join(wordParts)
        lineToPrint = str(item[0]) + ':' + wordToPrint + '\n'
        myfile2.write(lineToPrint)
myfile2.close()
print("Done!")
