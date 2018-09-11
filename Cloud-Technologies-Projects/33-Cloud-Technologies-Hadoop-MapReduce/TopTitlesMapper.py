#!/usr/bin/env python
import sys

mylist=[]
for line in sys.stdin:
#with open("output1.txt") as f:
    #for line in f:
    prline = line.strip().split()
    if len(prline) >= 2: temp =[prline[0], int(prline[1])]
    if len(prline) >= 2: mylist.append(temp)

#myfile = open("output2.txt", "w")
mywords = []
for element in sorted(mylist, key=lambda x: int(x[1]), reverse=True):
    mywords.append(element)
mywords = mywords[:10]
for word in mywords:
    toPrint = word[0] + "   " + str(word[1])
    print(toPrint)
    #myfile.write(toPrint)
#myfile.close()