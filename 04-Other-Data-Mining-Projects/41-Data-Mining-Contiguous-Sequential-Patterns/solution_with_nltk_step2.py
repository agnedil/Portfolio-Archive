# takes a file with raw FreqDist ngram frequencies (just words, numbers removed), removes duplicate ngrams,
# recounts the support of each ngram pruning those below 100
# support - ONLY one occurrence per yelp review/line is counted

import nltk
from nltk import bigrams, word_tokenize

#create list of frequent ngrams found at step 1
listLines = []
with open("reviews_sample.txt") as myfile:
    for line in myfile:
        line = line.strip()
        listLines.append(line)

#crerate list of strings, each string = 1 reveiw/line from yelp file
listPhrases = []
with open("patterns_rawFrequencies.txt") as myfile2:
    for line in myfile2:
        line = line.strip()
        if line not in listPhrases:
            listPhrases.append(line)

#count the support
finalCount = []
for phrase in listPhrases:
    a = word_tokenize(phrase)
    if " " in phrase:                                               #I know there are only bigrams and unigrams; otherwise this needs improvement
        count = 0                                                   #countint bigrams
        for line in listLines:
            c = list(nltk.bigrams(line.split()))
            for i in range (len(c)):
                phrase1 = " ".join(c[i])
                if phrase == phrase1:
                    count += 1
                    break
        print('{}:{}'.format(count, phrase))
    else:                                                           #counting unigrams
        count = 0
        for line in listLines:
            b = nltk.tokenize.word_tokenize(line)
            if phrase in b:
                count += 1
        print('{}:{}'.format(count, phrase))
    mylist = [count, phrase]
    finalCount.append(mylist)

#sort and write to file
finalCount = sorted(finalCount, key=lambda x: int(x[0]), reverse=True)
myfile3 = open("patterns.txt", "w")
for item in finalCount:
    if item[0] >= 100:
        wordParts = item[1].split(' ')
        wordToPrint = ';'.join(wordParts)
        lineToPrint = str(item[0]) + ':' + wordToPrint + '\n'
        myfile3.write(lineToPrint)
myfile3.close()
print("Done!")
