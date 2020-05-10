import random 
import os
import string
import sys

stopWordsList = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
            "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
            "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
            "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

delimiters = "\t,;.?!-:@[](){}_*/"                              #deleted space as it is handled by string.split()

def getIndexes(seed):
    random.seed(seed)
    n = 10000
    number_of_lines = 50000
    ret = []
    for i in range(0,n):
        ret.append(random.randint(0, 50000-1))
    return ret

def toWords(line):
    line=line.lower()
    line=line.decode('utf-8')
    for character in line:
        if character in delimiters:
            line=line.replace(character, " ")
    return line.split()                                           #remove more than one space, trailing and leading spaces

def addToDict(dicto, words):
    for word in words:
        if word not in stopWordsList:
            if word not in dicto:
                dicto[word] = 1
            else:
                dicto[word] += 1
    return dicto

def process(userID):
    indexes = getIndexes(userID)
    ret = []
    dicto=dict()
    # TODO
    data = sys.stdin.readlines()

    for i in indexes:
        line=data[i]
        line=toWords(line)
        dicto=addToDict(dicto, line)

    for key, value in sorted(dicto.iteritems(), key=lambda (k,v): (v,k), reverse=True):
        mylist=[key, value]
        ret.append(mylist)

    for i in range(25):                                                         #improve sorting words with the same count
        if ret[i][1] == ret[i+1][1]:
            if ret[i][0] < ret[i+1][0]:
                temp=ret[i][0]
                ret[i][0]=ret[i+1][0]
                ret[i+1][0]=temp

    #ret = sorted(interim, key=lambda x: int(x[1]), reverse=True)               # may be used to sort list of lists

    for i in range(20):
        print(ret[i][0])

    #for i in range 20:
        #ret.append(dicto[i])
                    
    #for word in ret:
        #print word

process(sys.argv[1])
