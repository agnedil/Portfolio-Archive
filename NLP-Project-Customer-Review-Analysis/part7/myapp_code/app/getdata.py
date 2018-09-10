from __future__ import division
import numpy as np
import io
from flask import render_template, flash, redirect, url_for, send_file
from app import myapp
from app.forms import mySelectForm
from collections import defaultdict
from operator import itemgetter

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

# exctract data from the general file and save separately 
def getData(selectedCuisine, selectedDish, rawDir):
     
    # read in raw data
    rawFile = rawDir + '/' + selectedCuisine + '/' + selectedCuisine + '.txt'
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
    with io.open(rawDir + '/' + selectedCuisine + '/' + 'data.tsv', mode='w', encoding='utf-8') as f2:
        f2.write(unicode('imya' + '\t' + 'number' + '\t' + 'another' + '\n'))
        for key, value in sorted(restRate.items(), key=itemgetter(1), reverse=True):
            if value >= 0:
                f2.write(unicode(key + '\t' + str(value) + '\t' + str(restSent[key]) + '\n'))

    # clear memory
    restRate = None
    restSent = None

if __name__ == "__main__":
    pass
