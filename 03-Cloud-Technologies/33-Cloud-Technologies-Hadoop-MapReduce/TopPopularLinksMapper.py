#!/usr/bin/env python
import sys

myList = []
for line in sys.stdin:
    if line.strip():
        line = line.decode('utf-8')
        line = line.replace(":", " ")
        temp = line.split()
        for item in temp:
            item = item.strip()
        myList.append(temp)

#f = open("output.txt", "w")
for element in myList:
    del element[0]
    for item in element:
        print(item)
        #f.write(item + "\n")
#f.close