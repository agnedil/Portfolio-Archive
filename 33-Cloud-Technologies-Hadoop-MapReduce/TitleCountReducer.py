#!/usr/bin/env python
from operator import itemgetter
import sys

mydict=dict()
#myfile = open("output1.txt", "w")
for word in sys.stdin:
#with open("output.txt") as f:
    #for word in f:
    word = word.strip()
    if word in mydict:
        mydict[word] += 1
    else:
        mydict[word] = 1

for key, value in mydict.items():
    print (key + "  " + str(value))
    #myfile.write(key + " " + str(value) + "\n")
#myfile.close()