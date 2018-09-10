#!/usr/bin/env python
import sys

mylist=[]
#with open("output2.txt") as f:
for line in sys.stdin:
    #for line in f:
    prline = line.strip().split()
    if len(prline) >= 2: temp =[prline[0], int(prline[1])]
    if len(prline) >= 2: mylist.append(temp)

mylist = sorted(mylist, key=lambda x: x[0])

for element in mylist:                                          # final output to stdout
    print(element[0] + "    " + str(element[1]))
