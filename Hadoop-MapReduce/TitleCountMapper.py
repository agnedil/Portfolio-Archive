#!/usr/bin/env python

import sys
import string

# function to lowercase and tokenize using the provided delimiters
def toWords(line):
    line=line.lower()
    #line=line.decode('utf-8')
    for character in line:
        if character in delimiters:
            line=line.replace(character, " ")
    return line.strip().split()                          # removes more than one space, trailing and leading spaces


stopWordsPath = sys.argv[1]
delimitersPath = sys.argv[2]

# getting a string of delimiters without a space
with open(delimitersPath) as f:
    delimiters=f.read()
delimiters = delimiters.replace(" ", "")

# getting a list of stopwords
swlist=[]
with open(stopWordsPath) as f:
    sw = f.read()
sw=sw.lower()
sw=sw.decode('utf-8')
for character in sw:
    if character in delimiters:
        sw = sw.replace(character, " ")
sw = sw.strip().split()
for wordy in sw:
    wordy = wordy.strip()
    swlist.append(wordy)

#myfile = open("output.txt", "w")
# process and send to stdout
for line in sys.stdin:
    line = toWords(line)
    for word in line:
        if not (word in swlist):
            if not (len(word) <= 1):
                print(word)
                #myfile.write(word + "\n")
#myfile.close()