import sys
sys.path.append('bbclf/')
import numpy as np
import pandas as pd
#import spacy
#from spacytextblob.spacytextblob import SpacyTextBlob

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


#nlp = spacy.load("en_core_web_lg")
#nlp.add_pipe('spacytextblob')

negators = ['cannot', "can't", 'cant', 'not', 'no', 'never', 'nothing', 'neither', 'nor']
not_negators = [ ('not', 'just'), ('not', 'to', 'mention'), ]
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

additional_sentiment_data = 'bbclf/data/additional_sentiment.tsv'
additional_sentiment = pd.read_csv( additional_sentiment_data, sep='\t' )
additional_sentiment['word'] = additional_sentiment['word'].apply(lambda x: x.lower().strip())
additional_sentiment = dict( additional_sentiment.values )

# t._.polarity = range [-1.0, 1.0], t._.subjectivity = range [0.0, 1.0] where 1.0 is very subjective
#def textblob_sentiment( doc ):
#    '''
#        Return TextBlob sentiment for each word in doc as dict()
#    '''
#    textblob_sentiments = [ (t.text, round(t._.polarity, 4)) for t in doc ]
#    return { k: v for k,v in textblob_sentiments if v }

def penn_to_wordnet_pos_tags( tag ):
    """
    Convert the PennTreebank NLTK POS tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def vader_word_sentiment( word, lemma ):
    '''
        Return Vader sentiment for one word
    '''
    if not isinstance(word, str) or not word:
        return word, None

    if not isinstance(lemma, str) or not lemma:
        lemma = word

    scores = sia.polarity_scores( lemma )
    if scores:
        return word, scores['compound']
    else:
        return word, None


def vader_sentiment( doc ):
    '''
        Return Vader sentiment for each word in doc as dict()
    '''
    vader_sentiments = [ vader_word_sentiment( t.text, t.lemma_.lower() ) for t in doc ]
    return { k: v for k,v in vader_sentiments if v }


def sentiWordnet_word_sentiment( word, tag ):
    '''
        Look in the file with additional sentiment scores. If not found,
        Estimate approximated sentiment of one word based on SentiWordNet pos and neg scores
    '''

    # get WordNet POS tags; keep only the ones that matter
    wn_tag = penn_to_wordnet_pos_tags( tag )
    if word in additional_sentiment:
        return word, additional_sentiment[ word ]
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
        return word, None

    # WordNet lemmatizer; if no lemma, return 0.0
    lemma = lemmatizer.lemmatize( word, pos=wn_tag )
    if not lemma:
        return word, None
    if lemma in additional_sentiment:
        return word, additional_sentiment[ lemma ]

    # get all synsets for lemma
    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return word, None

    # get pos & neg SentiWordNet scores for each WorNet sense
    scores =  []
    for synset in synsets:
        swn_synset = swn.senti_synset(synset.name())
        if swn_synset:
            scores.append([ swn_synset.pos_score(), swn_synset.neg_score(), ])

    # count pos & neg scores and get max pos & neg scores
    pos_count = len([ i for i in scores if i[0] > i[1] ])
    neg_count = len([ i for i in scores if i[1] > i[0] ])
    pos_max   = max([ i[0] for i in scores ])
    neg_max   = max([ i[1] for i in scores ])

    # main logic to estimate sentiment
    if   pos_count > neg_count:
        return word, pos_max
    elif neg_count > pos_count:
        return word, -neg_max
    elif pos_count == neg_count:

        if pos_max == neg_max:                    # this also covers the case when every score is 0.0 (all neutrals)
            return word, None
        elif pos_max > neg_max:
            return word, pos_max
        else:
            return word, -neg_max

    return word, None


def sentiWordnet_sentiment( doc ):
    '''
        Return custom SentiWordnet sentiment for each word in doc as dict()
    '''
    swn_sentiments = [ sentiWordnet_word_sentiment( t.text.lower(), t.tag_ ) for t in doc ]
    return { k: v for k,v in swn_sentiments if v }


def get_sentiment( doc ):
    '''
        Return complex sentiment scores for words in doc. When merging dicts,
        last dict always takes precidence. Therefore, this is the precedence of the 3 source:
        * Vader
        * TextBlob
        * Approximated SentiWordnet score
    '''

    swn      = sentiWordnet_sentiment( doc )
    #textblob = textblob_sentiment( doc )
    vader    = vader_sentiment( doc )

    return { **swn, **vader }
