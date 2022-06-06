# takes a file with ngram counts, removes duplicate ngrams, rechecks count in file with original text

import nltk
from nltk import bigrams, word_tokenize

listLines = []
with open("reviews_sample.txt") as myfile:
    for line in myfile:
        line = line.strip()
        listLines.append(line)

listPhrases = []
with open("patternsALL.txt") as myfile2:
    for line in myfile2:
        line = line.strip()
        if line not in listPhrases:
            listPhrases.append(line)

finalCount = []
for phrase in listPhrases:
    if " " in phrase:
        count = 0
        for line in listLines:
            c = list(nltk.bigrams(line.split()))
            for i in range (len(c)):
                phrase1 = " ".join(c[i])
                if phrase == phrase1:
                    count += 1
                    break
        mylist = [count, phrase]
        finalCount.append(mylist)
        print('{}:{}'.format(count, phrase))

finalCount = sorted(finalCount, key=lambda x: int(x[0]), reverse=True)
myfile3 = open("patterns2_cc.txt", "w")
for item in finalCount:
    if item[0] >= 100:
        wordParts = item[1].split(' ')
        wordToPrint = ';'.join(wordParts)
        lineToPrint = str(item[0]) + ':' + wordToPrint + '\n'
        myfile3.write(lineToPrint)
myfile3.close()
print("Done!")
