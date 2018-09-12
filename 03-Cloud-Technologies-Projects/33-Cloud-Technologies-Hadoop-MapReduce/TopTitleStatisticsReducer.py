#!/usr/bin/env python
import sys

wordList=[]
countList=[]
for line in sys.stdin:
    prline = line.strip().split()
    if len(prline) >= 2:
        wordList.append(prline[0])
        countList.append(int(prline[1]))

sum_all = sum(countList)
mean_count = sum_all / len(countList)
min_count = min(countList)
max_count = max(countList)
variance = sum((mean_count - value) ** 2 for value in countList) / len(countList)

print("Mean" + "\t" + str(mean_count).strip())
print("Sum" + "\t" + str(sum_all).strip())
print("Min" + "\t" + str(min_count).strip())
print("Max" + "\t" + str(max_count).strip())
print("Var" + "\t" + str(variance).strip())