import numpy as np
from PIL import Image
from os import path
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Qt4Agg')                    # required, otherwise matplotlib will use ''Qt5Agg' generating error
from matplotlib import pyplot as plt


def getData(fileName):

    with open(fileName) as f:
        mylist = []

        if len(f.readlines()) != 160:
            print("Not a 10 x 15 format")
        f.seek(0)

        print("Reading data")
        for i in range (10):
            freqDict = {}
            f.readline()
            for j in range (15):
                line = f.readline()
                line = line.strip().split(":")
                if len(line) != 2:
                    print("Irregular input for ", line)
                else:
                    freqDict[line[0]] = float(line[1])
            mylist.append(freqDict)

    print("Done!")
    return mylist


data = getData('nmf_sample_topics_positive.txt')
plt.figure()                                    # facecolor="gray" for entire figure's background

for k in range (10):
    print("Plotting figure " + str(k+1))
    plt.subplot(5, 2, k+1)
    wc = WordCloud(background_color="white", max_words=15, colormap="viridis")
    wc.generate_from_frequencies(data[k])
    plt.imshow(wc)
    plt.axis("off")
    Title = 'Topic ' + str(k+1)
    plt.title(Title)

plt.subplots_adjust(top=0.97, bottom=0.03, left=0.61, right=1.0, hspace=0.25, wspace=0.0)
#plt.tight_layout()
plt.show()

"""
fig, ax = plt.subplots(nrows=2, ncols=2)

for row in ax:
    for col in row:
        col.plot(x, y)

plt.show()
"""