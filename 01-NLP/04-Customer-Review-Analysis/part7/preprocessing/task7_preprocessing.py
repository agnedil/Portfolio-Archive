# this code uses a portion of the main task 4 and 5 program
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
cuisine = "Indian"                                                                  # CHANGE ACCORDINGLY

# map {business_id: name}
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

# read list of dishes into list
dishlist = []
with io.open('dish_lists/dishes_indian.txt', mode='r', encoding='utf-8') as f4:     # CHANGE ACCORDINGLY
    for line in f4:
        line = line.strip()
        dishlist.append(line)

# map {dish+bus_id: list of ratings, dish+bus_id: list of sentiments}
print('mapping {bus_id + dish: 2 lists}')
id2rate = defaultdict(list)                                     # to record the above map
id2sent = defaultdict(list)                                     # to record sentiment
for dish in dishlist:
    with io.open(reviews, mode='r', encoding='utf-8') as f3:
        for line in f3.readlines():
            business_json = json.loads(line)
            rating = business_json['stars']
            if rating == 3:
                continue                                        # skip 3-star reviews
            bid = business_json['business_id']
            revtext = business_json['text'].lower()
            if (bid in id2name) and (dish in revtext):          # if restaurant and dish
                sent = TextBlob(revtext).sentiment.polarity     # sentiment analysis part
                mykey = dish + "; " + bid
                id2rate[mykey].append(rating)
                id2sent[mykey].append(sent)
                if len(id2rate) % 100 == 0:
                    print(str(len(id2rate)))

print "\nsaving to file"
with io.open(cuisine + '.txt', mode='w', encoding='utf-8') as f2:
    for key, value in id2rate.iteritems():
        key2 = key.strip().split('; ')
        newkey = key2[0].strip() + '; ' + id2name[key2[1].strip()]
        sentAve = str(np.mean(id2sent[key]))
        f2.write(unicode(newkey + "\t" + sentAve + "\t" + "\t".join(str(item) for item in value) + "\n"))

print "done!"

# clear memory
id2name = None
id2sent = None
id2rate = None