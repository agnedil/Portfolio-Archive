#from matplotlib import *
from gensim.models import Phrases, Word2Vec
import pandas as pd
import nltk
import re
import string
from collections import Counter
from stop_words import safe_get_stop_words
# import graphlab as gl                             # use a different library
#from wordcloud import WordCloud

stopwords = list(safe_get_stop_words('en')) + list(string.punctuation) + list(nltk.corpus.stopwords.words('english'))
print("stopwords created")
print(stopwords)

# load and preprocess Italian cuisine reviews
reviews = pd.read_csv('Italian.txt',sep='\n\n',encoding='utf-8',header=None, engine='python').as_matrix().tolist()
print("reviews loaded")

reviewSents = []
for review in reviews:
    out = re.sub('[^a-z]+', ' ', review[0].decode('utf-8').lower())
    sentence = [i for i in nltk.word_tokenize(out) if i not in stopwords]
    reviewSents.append(sentence)
print("stopwords removed from reviews")

# load and preprocess quality Italian cuisine phrases
qualityPhrases = pd.read_csv('00_complete_Italian_dishes.txt',sep='\n\n',encoding='utf-8',header=None, engine='python').as_matrix().tolist()
print("quality phrases loaded")

qualitySents = []
for phrase in qualityPhrases:
    out = re.sub('[^a-z]+', ' ', phrase[0].decode('utf-8').lower())
    sentence = [i for i in nltk.word_tokenize(out) if i not in stopwords]
    qualitySents.append(sentence)
print("stopwords removed from quality phrases")

reviewPhrases = Phrases(reviewSents)
#qualityPhrases = Phrases(qualitySents)

reviewModel = Word2Vec(reviewPhrases[reviewSents], workers=multiprocessing.cpu_count(), iter=3, hs=1, negative=0, size=200, seed = 317)
#qualityModel = Word2Vec(qualityPhrases[qualitySents], hs=1, negative=0, size=200, seed = 317)
print ("word2vec model is learned")

# join quality phrases the word2vec style with "_"
joinedQualitySents = []
for phrase in qualitySents:
    joinedPhrase = "_".join(phrase)
    joinedQualitySents.append(joinedPhrase)

# filter to make sure all of them are in the wv model's vocabulary to avoid error messages
superQualityPhrases = filter(lambda x: x in reviewModel.wv.vocab.keys(), joinedQualitySents)
print(str(len(superQualityPhrases)))

# run filtered phrases through the wv model and get top 10 similar words for each phrase
newPhrases = []
for phrase in superQualityPhrases:
    similar = reviewModel.wv.similar_by_word(phrase, topn=10)
    mylist = []
    for item in similar:
        mylist.append(item[0])
    newPhrases.extend(mylist)

# order the new phrases by their count (also removing duplicates this way)
counter = Counter()      # dictionary object from collections; counts words fed to it; key=word, value=count
for phrase in newPhrases:
   if len(phrase.split('_'))>1:
       counter[phrase.replace('_',' ')] += reviewModel.wv.vocab[phrase].count
for word,cnt in counter.most_common(100):
   print(word.encode("utf-8") + " : " + str(cnt))

with open('dishes.txt','w') as f:
   f.write("\n".join([word for word,cnt in counter.most_common()]))
print("done!")

#debug wordcloud
#wc = WordCloud(background_color='white').generate_from_frequencies(counter)
#plt.imshow(wc)
#plt.axis("off")
#plt.show()