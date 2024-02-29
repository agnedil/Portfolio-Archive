#this script shall be run from the console by "cat input.txt | python MP0.py 1" OR "cat input.txt | python MP0.py 1"
#only specific lines from input.txt are processed through getIndexes()
#word count is run for those lines split by delimeters; top 20 words are printed
#words with the same count shall be ordered lexicographically

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

def getIndexes(seed):                                           #provides a list of indices of specific lines
    random.seed(seed)
    n = 10000
    number_of_lines = 50000
    ret = []
    for i in range(0,n):
        ret.append(random.randint(0, 50000-1))
    return ret

def toWords(line):                                              #normalizes and splits a line into a list of words
    line=line.lower()
    line=line.decode('utf-8')
    for character in line:
        if character in delimiters:
            line=line.replace(character, " ")
    return line.split()                                         #removes more than one space, trailing and leading spaces

def addToList(mylist, words):                                   #adds a list of words to a list of words and counts
    for word in words:
        if word not in stopWordsList:
            found = False
            for element in mylist:
                if element[0] == word:
                    found = True
                    element[1] += 1

            if not found:
                new = [word, 1]
                mylist.append(new)

    return mylist

def process(userID):
    indexes = getIndexes(userID)                                #get the word count from these lines only
    mylist=[]                                                   #list of lists (each element = word + count)
    data = sys.stdin.readlines()                                #read the data from stdin (input.txt)

    for i in indexes:
        line=data[i]                                            #get each line
        line=toWords(line)
        mylist=addToList(mylist, line)

    mywords, mycount = [], []                                   #split into separate lists of words and counts
    for element in sorted(mylist, key=lambda x: int(x[1]), reverse=True):
        mywords.append(element[0])
        mycount.append(element[1])

    found=True
    while found:                                                #lexicological sorting words with the same count
        j=0
        for i in range(25):
            if int(mycount[i]) == int(mycount[i+1]):
                if str(mywords[i]) > str(mywords[i+1]):
                    j += 1
                    temp=mywords[i]
                    mywords[i]=mywords[i+1]
                    mywords[i+1]=temp
        if j==0: found=False

    for i in range(20):                                         # final output to stdout
        print(mywords[i])

process(sys.argv[1])
