# https://stackoverflow.com/questions/3199171/append-multiple-values-for-one-key-in-python-dictionary
# used https://stackoverflow.com/questions/491921/unicode-utf-8-reading-and-writing-to-files-in-python

from __future__ import division
import json
import io
from collections import defaultdict
from textblob import TextBlob
import numpy as np

path2files="/home/andrew/Documents/2_UIUC/CS598_Data_Mining_Capstone/task1/yelp_dataset_challenge_academic_dataset/"
business=path2files+"yelp_academic_dataset_business.json"
reviews=path2files+"yelp_academic_dataset_review.json"

# map {business_id: name}
cuisine = "Italian"
id2name = dict()

print('mapping business_id to name' + '\n')
with io.open(business, mode='r', encoding='utf-8') as f:
    for line in f.readlines():
        business_json = json.loads(line)
        cats = business_json['categories']
        bid = business_json['business_id']
        bname = business_json['name']
        if cuisine in cats:
            id2name[bid] = bname
            if len(id2name) % 100 == 0:
                print(bname)

print "\nsaving restaurant ratings"
with io.open('business_id2name.txt', mode='w', encoding='utf-8') as f2:
    for key in id2name:
        f2.write(key + ": " + id2name[key] + "\n")
print "done!"

# map {business_id + dish: poslist, neglist}
print('mapping {business_id + dish: poslist, neglist}')
dishlist = []
with io.open('dishes.txt', mode='r', encoding='utf-8') as f4:
    for line in f4:
        line = line.strip()
        dishlist.append(line)

id2rating = defaultdict(list)                                   # to record the above map
id2sent = defaultdict(list)                                     # to record sentiment
for dish in dishlist:
    with io.open(reviews, mode='r', encoding='utf-8') as f3:
        for line in f3.readlines():
            business_json = json.loads(line)
            rating = business_json['stars']
            if rating == 3:
                continue                                        # skip 3-star reviews
            bid = business_json['business_id']
            if bid in id2name:                                  # if restaurant in the list of Italian restaurants
                revtext = business_json['text']
                if dish in revtext:                             # if dish is mentioned in this review
                    mykey = dish + "; " + bid
                    id2rating[mykey].append(rating)
                    if len(id2rating) % 100 == 0:
                        print(str(len(id2rating)))

                    toAnalyze = TextBlob(revtext)               # sentiment analysis part
                    sent = toAnalyze.sentiment.polarity
                    id2sent[mykey].append(sent)

# dict {dish OR restaurant: list of all ratings}                # lines 68 - 114 PROCESSING RATING from id2rating
print('creating LONG lists of ratings for 1) dishes, 2) restaurants' + '\n')
topdishLong = defaultdict(list)
toprestLong = defaultdict(list)

for key, value in id2rating.iteritems():
    mylist = key.strip().split(";")
    idx1 = mylist[0].strip()
    idx2 = mylist[1].strip()
    topdishLong[idx1].extend(value)
    toprestLong[idx2].extend(value)
    if len(topdishLong) % 100 == 0:
        print(str(len(topdishLong)))
print "done!"
#clearing from memory unneeded data structures
#id2rating = None


# dict {dish OR restaurant: [cumulative rating, count] }
print('creating SHORT lists of ratings for 1) dishes, 2) restaurants' + '\n')
topdish = defaultdict(list)
toprest = defaultdict(list)

for key, value in topdishLong.iteritems():
    count = len(value)
    pos = 0
    neg = 0
    for item in value:
        if item < 3:
            neg += 1
        else:
            pos += 1
    totRate = (pos - neg) / count
    topdish[key].append(totRate)
    topdish[key].append(count)
print "dishes done!"

for key, value in toprestLong.iteritems():
    count = len(value)
    pos = 0
    neg = 0
    for item in value:
        if item < 3:
            neg += 1
        else:
            pos += 1
    totRate = (pos - neg) / count
    toprest[key].append(totRate)
    toprest[key].append(count)
print "restaurants done!"

# slearing unneeded data structures from memory
#topdishLong = None
#toprestLong = None

# dict {dish OR restaurant: list of all sentiments}                              # lines 116 -   PROCESSING SENTIMENT from id2sent
print('creating lists of sentimetns for 1) dishes, 2) restaurants' + '\n')
topdishSent = defaultdict(list)
toprestSent = defaultdict(list)

for key, value in id2sent.iteritems():
    mylist = key.strip().split(";")
    idx1 = mylist[0].strip()
    idx2 = mylist[1].strip()
    topdishSent[idx1].extend(value)
    toprestSent[idx2].extend(value)
    if len(topdishSent) % 100 == 0:
        print(str(len(topdishSent)))
print "done!"
# clearing from memory
#id2sent = None


# dict {map dish OR restaurant: average sentiment}
topdishSentAve = defaultdict(list)
toprestSentAve = defaultdict(list)

for key, value in topdishSent.iteritems():              # average sentiment per dish
    topdishSentAve[key] = np.mean(value)
for key, value in toprestSent.iteritems():              # average sentiment per restaurant
    toprestSentAve[key] = np.mean(value)
# clearing unneeded data structures from memory
#topdishSent = None
#toprestSent = None


print('writing to file')
with io.open('topdishes.txt', mode='w', encoding='utf-8') as f5:
    for key, value in topdish.iteritems():
        f5.write(key + ';' + str(value[0]) + ';' + str(value[1]) + ';' + str(topdishSentAve[key]) + '\n')

with io.open('toprest.txt', mode='w', encoding='utf-8') as f5:
    for key, value in toprest.iteritems():
        f5.write(id2name[key] + ';' + str(value[0]) + ';' +  str(value[1]) + ';' + str(toprestSentAve[key]) +'\n')
print "done!"

# clearing from memory
#id2name = None
#topdish = None
#toprest = None
#topdishSentAve = None
#toprestSentAve = None


# these were supposed to cleared from memory before; check it
#id2sent = None
#id2rating = None
#topdishSent = None
#toprestSent = None
#topdishLong = None
#toprestLong = None
