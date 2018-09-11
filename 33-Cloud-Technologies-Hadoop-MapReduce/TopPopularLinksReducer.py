#!/usr/bin/env python
import sys

mydict=dict()
for line in sys.stdin:
#with open("output.txt") as f:
    #for line in f:
    line = str(line)
    line = line.strip()
    if line in mydict:
        mydict[line] += 1
    else:
        mydict[line] = 1

myList = []
for key, value in mydict.items():
    toList = [key, value]
    myList.append(toList)

mywords = []
for element in sorted(myList, key=lambda x: (x[1]), reverse=True):
    mywords.append(element)
mywords = mywords[:10]

for element in mywords:
    element[0] = str(element[0])

for element in sorted(mywords, key=lambda x: x[0]):                                          # final output to stdout
    a = element[0].strip()
    b = str(element[1]).strip()
    print(a + "\t" + b)