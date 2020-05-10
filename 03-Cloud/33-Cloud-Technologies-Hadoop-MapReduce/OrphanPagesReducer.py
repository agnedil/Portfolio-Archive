#!/usr/bin/env python
import sys

# create 2 lists - one w/pages, one w/links
mylist = []
for line in sys.stdin:
    if line.strip():
        line = line.decode('utf-8')
        line = line.replace(":", " ")
        temp = line.split()
        for item in temp:
            item = item.strip()
        mylist.append(temp)

# select orphan pages by looking in the links
result = []
for i in range(len(mylist)):
    print(str(i))
    findit = mylist[i][0]
    found = False
    for j in range(len(mylist)):
        if (findit in mylist[j]) and (findit != mylist[j][0]):
            found = True
            break
    if not found: result.append(findit)

#f = open("output.txt", 'w')
for element in sorted(result):
    print(element.strip())
    #f.write(element + "\n")
#f.close()