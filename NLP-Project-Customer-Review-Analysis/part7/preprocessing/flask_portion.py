# this code uses a portion of the main task 4 and 5 program
# https://stackoverflow.com/questions/3199171/append-multiple-values-for-one-key-in-python-dictionary
# used https://stackoverflow.com/questions/491921/unicode-utf-8-reading-and-writing-to-files-in-python

from __future__ import division
import io
from collections import defaultdict
import numpy as np
from operator import itemgetter

selectedCuisine = 'Mediterranean'
selectedDish = 'greek salad'
rawFile = 'static/' + selectedCuisine + '/' + selectedCuisine + '.txt'

# my rating function
def rating(mylist):
    count = float(len(mylist))
    pos = 0
    neg = 0
    for item in mylist:
        if item < 3:
            neg += 1
        else:
            pos += 1
    totRate = float(pos - neg) / count
    if totRate > 0:
        totRate = totRate - 1 / count
    return totRate

# read in raw data
restRate = defaultdict(list)
restSent = defaultdict(list)
with io.open(rawFile, mode='r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip().split('\t')
        oldKey = line[0].strip().split(";")
        if oldKey[0].strip() == selectedDish:
            mylist = [int(item.strip()) for item in line[2:]]
            restRate[oldKey[1].strip()].extend(mylist)
            restSent[oldKey[1].strip()].append(float(line[1].strip()))

# rate/average each list in dict value
restRate = {key: value for key, value in restRate.items() if len(value) > 3}
for key, value in restRate.iteritems():
    restRate[key] = float(rating(value))

for key, value in restSent.iteritems():
    restSent[key] = np.mean(value)

# finally write to file
with io.open('static/' + selectedCuisine + '/' + selectedDish + '_data.tsv', mode='w', encoding='utf-8') as f2:
    f2.write(unicode('imya' + '\t' + 'number' + '\t' + 'another' + '\n'))
    for key, value in sorted(restRate.items(), key=itemgetter(1), reverse=True):
        if value >= 0:
            f2.write(unicode(key + '\t' + str(value) + '\t' + str(restSent[key]) + '\n'))

# clear memory
restRate = None
restSent = None