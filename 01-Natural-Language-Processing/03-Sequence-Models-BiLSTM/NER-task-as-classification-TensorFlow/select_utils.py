import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import hashlib
import string
import spacy
import en_core_web_lg
import random
from typing import List
import json
import itertools
import time
import datetime
import matplotlib.pyplot as plt
import gc
import sys
#from multiprocessing import Pool
#from tqdm import tqdm


EPOCHS     = 11
VOCAB_SIZE = 115000
OUTPUT_DIM = 10
BATCH_SIZE = 32
UNITS      = 128
#LR         = 0.0001
REDUCE_LR  = True
PREPROCESS = False

# CONSTANTS
pos_table = dict( zip([
                        '$',
                        "``",
                        "''",
                        ',',
                        '-LRB-',
                        '-RRB-',
                        '.',
                        ':',
                        'ADD',
                        'AFX',
                        'CC',
                        'CD',
                        'DT',
                        'EX',
                        'FW',
                        'GW',
                        'HYPH',
                        'IN',
                        'JJ',
                        'JJR',
                        'JJS',
                        'LS',
                        'MD',
                        'NFP',
                        'NIL',
                        'NN',
                        'NNP',
                        'NNPS',
                        'NNS',
                        'PDT',
                        'POS',
                        'PRP',
                        'PRP$',
                        'RB',
                        'RBR',
                        'RBS',
                        'RP',
                        'SP',
                        'SYM',
                        'TO',
                        'UH',
                        'VB',
                        'VBD',
                        'VBG',
                        'VBN',
                        'VBP',
                        'VBZ',
                        'WDT',
                        'WP',
                        'WP$',
                        'WRB',
                        'XX',
                        '_SP',
                        ],
                     range(2, 100)
                    )
                )


dep_table = dict( zip([
                        "ROOT",
                        "acl",
                        "acomp",
                        "advcl",
                        "advmod",
                        "agent",
                        "amod",
                        "appos",
                        "attr",
                        "aux",
                        "auxpass",
                        "case",
                        "cc",
                        "ccomp",
                        "compound",
                        "conj",
                        "csubj",
                        'csubjpass',
                        "dative",
                        "dep",
                        "det",
                        "dobj",
                        "expl",
                        "intj",
                        "mark",
                        'meta',
                        "neg",
                        "nmod",
                        "npadvmod",
                        "nsubj",
                        "nsubjpass",
                        "nummod",
                        "oprd",
                        "parataxis",
                        "pcomp",
                        "pobj",
                        "poss",
                        "preconj",
                        "predet",
                        "prep",
                        "prt",
                        "punct",
                        "quantmod",
                        "relcl",
                        "subtok",
                        "xcomp",
                        ],
                     range(2, 100)
                    )
                )


my_ent_table  = dict( zip([  'ADDRESS',
                              'ADDRESS_BLOCK',
                              'INTERSECTION',
                              'STREET',
                              'LOC',
                              'FAC',
                              'GPE',
                              'ORG',
                            ],
                        range(2, 50)
                           )
                      )


#ent_type_table  = dict( zip([ 'ADDRESS',
#                              'ADDRESS_BLOCK',
#                              'INTERSECTION',
#                              'STREET',
#                              'LOC',
#                              'FAC',
#                              'GPE',
#                              'ORG',
#                              'PERSON',
#                              'DATE',
#                              'CARDINAL',
#                              'TIME',
#                              'NORP',
#                              'QUANTITY',
#                              'ORDINAL',
#                              'DISEASE',
#                              'PRODUCT',
#                              'MONEY',
#                              'WORK_OF_ART',
#                              'EVENT',
#                              'PERCENT',
#                              'MILITARY_ORG',
#                              'AIRCRAFT_TYPE',
#                              'LAW',
#                              'EQM',
#                              'LANGUAGE',
#                              'EQD',
#                              'FLIGHT',
#                            ],
#                        range(2, 50)
#                           )
#                      )


def generalize_token( s ):

    domains = ['.org', '.net', '.mil', '.int', '.gov', '.edu', '.com']
    if any([i in s for i in domains]):
        return 'WEBSITE'

    s = s.translate(str.maketrans("", "", string.punctuation)).strip()
    s = s.translate(str.maketrans("", "", string.digits)).strip()
    if not s:
        return 'CARDINAL'
    elif s.lower() in ['am', 'pm', 'gmt']:
        return 'TIME'
    elif s.lower() in ['year', 'years', 'yr', 'yrs', 'month', 'months', 'hour', 'hours', 'hr', 'hrs', 'minute']:
        return 'DURATION'
    elif s.lower() == 'magnitude':
        return 'MAGNITUDE'
    elif s.lower() in ['tip', 'tips', 'stop', 'crime']:
        return 'PHONE'
    elif s.lower() in ['st', 'nd', 'rd', 'th']:
        return 'ORDINAL'
    elif s.lower() in ['news', 'fox', 'abc', 'nbc', ]:
        return 'NEWS AGENCY'
    elif s.lower() == 'block':
        return 'BLOCK NUMBER'
    elif s.lower() in ['foot', 'feet', 'meter', 'metre', 'meters', 'metres', 'mile', 'miles', 'acre', 'acres',
                       'inch', 'inches', 'ton', 'mph', 'kilometre', 'kilometer', 'kilometres', 'kilometers',
                       'acre', 'acres', 'km', 'kms', 'litre', 'liter', 'litres', 'liters', 'person' ]:
        return 'QUANTITY'

    return s


def hash_one_token(token: str, n=VOCAB_SIZE) -> int:

    token = token.translate(str.maketrans("", "", string.punctuation))
    token_generalized = generalize_token( token )
    if token_generalized in [ 'CARDINAL', 'WEBSITE', 'TIME', 'DURATION', 'MAGNITUDE', 'PHONE', 'ORDINAL',
                              'NEWS AGENCY', 'QUANTITY' ]:
        token = token_generalized

    token = token.encode("utf-8")
    token_hashint = int(hashlib.md5(token).hexdigest(), 16)
    return token_hashint % (n - 1) + 1


def plot_confusion_matrix(cm, START_TIME, classes,
                          title='CONFUSION MATRIX',
                          cmap=plt.cm.PuBu):              # plt.cm.Blues; also good: BuPu,RdPu,PuRd,OrRd,Oranges
    '''
    Plot the confusion matrix
    '''
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    plt.figure(figsize=(5,5))
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.05)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.tight_layout()
    plt.savefig( START_TIME.strftime('models/%Y%m%dT%H%M_confusion_matrix.png') )
    plt.show()


def generate_features( row ):

    # PREPARE BEST ENTITY
    best_ent = row['best_ent'][0]
    assert isinstance( best_ent, dict), f'Unknown type of best entity: {best_ent}'
    best_ent_start, best_ent_end = best_ent['startIndex'], best_ent['endIndex']

    # PREPARE DOC
    if len(row['doc']) > 300:
        doc = row['doc'][:300]
    else:
        doc = row['doc']

    try:
        best_ent_start >= 0 and best_ent_end > 0
    except TypeError:
        best_ent_start = int(best_ent_start)
        best_ent_end = int(best_ent_end)

    # CREATE FEATURES
    idx_counter, word_counter = 0, 0
    toks, poss, deps, ents, highlights = [], [], [], [], []
    for token in doc:

        toks.append( hash_one_token( token.text, n=VOCAB_SIZE ) )
        poss.append( pos_table.get( token.tag_, 1 ))
        deps.append( dep_table.get( token.dep_, 1 ))

        found = False
        for e in row['my_ents']:
            if e['startIndex'] <= idx_counter <= e['endIndex']:
                found = True
                ents.append( my_ent_table.get( e['type'], 1 ))
        if not found:
            ents.append(1)

        if best_ent_start <= idx_counter <= best_ent_end:
            highlights.append(2)
        else:
            highlights.append(1)

        word_counter = token.i
        if word_counter < len(doc)-1:
            idx_counter = doc[ word_counter+1 ].idx
        else:
            idx_counter += len(token) + 1

    row['text_feat'] = toks
    row['pos_feat'] = poss
    row['dep_feat'] = deps
    row['ent_type_feat'] = ents
    row['main_highlight_feat'] = highlights

    return row


def get_other_highlights( row ):

    assert isinstance( row['my_ents'], list), f'Unknown type of entity entities: {row["my_ents"]}'

    # PREPARE DOC
    if len(row['doc']) > 300:
        doc = row['doc'][:300]
    else:
        doc = row['doc']

    # GET HIGHLIGHTS FOR EACH entity ENTITY WHICH IS NOT THE BEST entity
    my_ents_no_best_ent = [ e for e in row['my_ents'] if e != row['best_ent'][0] ]
    all_highlights = []
    for e in my_ents_no_best_ent:

        idx_counter, word_counter = 0, 0
        highlights = []
        for token in doc:

            if e['startIndex'] <= idx_counter <= e['endIndex']:
                highlights.append(2)
            else:
                highlights.append(1)

            word_counter = token.i
            if word_counter < len(doc)-1:
                idx_counter = doc[ word_counter+1 ].idx
            else:
                idx_counter += len(token) + 1

        all_highlights.append( highlights )

    return all_highlights


def get_x( df_ ):
    '''
        Get X_pos for best entitys,
        and X_neg for other entitys
    '''

    # POSITIVE EXAMPLES (best entity)
    X_pos = df_[ df_['target'] == 1 ][[ 'text_feat', 'pos_feat', 'dep_feat', 'ent_type_feat',
                                                'main_highlight_feat' ]].values.tolist()

    # NEGATIVE EXAMPLES (not best entity)
    X_neg = []
    count = 0
    for i in df_.index:
        other_highlights = df_.loc[i]['other_highlights']
        if not isinstance(other_highlights, list) or not other_highlights:
            print(f'No other entity entities except best entity at idx {i} because "highlights" = {other_highlights}')
            count += 1
            continue

        constants = df_.loc[i][['text_feat', 'pos_feat', 'dep_feat', 'ent_type_feat']].values.tolist()
        assert constants and isinstance( constants, list )
        new_X = []
        for h in other_highlights:
            new_X.append( constants + [h] )
        X_neg.extend( new_X )

    print(f'\nNo other entity in {count} cases\n')

    return X_pos, X_neg


def oversample( a, n ):
    '''
        Oversample list a by n more samples and shuffle
    '''
    assert isinstance(a, list)
    assert isinstance(n, int)
    res = a + random.choices( a, k=n )
    random.shuffle( res )
    return res


def build_model( VOCAB_SIZE,
                 input_length,
                 output_dim,
                 num_pos,
                 num_deps,
                 num_ent_types,
                 num_highlights=3, ):

    dropout_rate1 = 0.2
    dropout_rate2 = 0.5
    units         = UNITS

    vectorizer = keras.layers.Embedding( input_dim    = VOCAB_SIZE+1,
                                         output_dim   = output_dim,
                                         input_length = input_length,
                                         name         = 'vectorizer',
                                       )
    text       = keras.Input( shape=[input_length],
                              name='text'
                            )
    text_vectorized = vectorizer( text )
    #text_vectorized = keras.layers.Dropout( dropout_rate1 )(text_vectorized)

    pos_tokens = keras.Input( shape=[input_length],
                              name='pos',
                            )
    pos_embed  = keras.layers.Embedding( num_pos+1,
                                         output_dim,
                                         name='pos_embed',
                                       )(pos_tokens)

    dep_tokens = keras.Input( shape=[input_length],
                              name='dep',
                            )
    dep_embed  = keras.layers.Embedding( num_deps+1,
                                         output_dim,
                                         name='dep_embed',
                                       )(dep_tokens)

    ent_type_tokens = keras.Input( shape=[input_length],
                                   name="ent_type",
                                 )
    ent_type_embed = keras.layers.Embedding( num_ent_types+1,
                                             output_dim,
                                             name='ent_type_embed',
                                           )(ent_type_tokens)

    highlight_tokens = keras.Input( shape=[input_length],
                                    name='highlights',
                                  )
    highlight_embed  = keras.layers.Embedding( num_highlights+1,
                                               output_dim,
                                               name='highlight_embed',
                                             )(highlight_tokens)

    concat = keras.layers.Concatenate(axis=2)([ text_vectorized,
                                                pos_embed,
                                                dep_embed,
                                                ent_type_embed,
                                                highlight_embed,
                                              ])
    #concat = keras.layers.SpatialDropout1D( dropout_rate1 )(concat)
    #concat = keras.layers.Dropout( 0.1 )(concat)

    bilstm = keras.layers.Bidirectional( keras.layers.LSTM( units=units,
                                                            return_sequences=True,
                                                            #recurrent_dropout=dropout_rate1,
                                                          ),
                                        merge_mode = 'concat'
                                        )(concat)
    bilstm = keras.layers.Dropout( dropout_rate1 )(bilstm)

    lstm = keras.layers.LSTM( units=units,
                              return_sequences=True,
                              #recurrent_dropout=dropout_rate2,
                            )(bilstm)
    lstm = keras.layers.Dropout( dropout_rate2 )(lstm)

    out = keras.layers.GlobalAveragePooling1D(data_format='channels_first')(lstm)

    out = keras.layers.Dense(1, activation="sigmoid")(out)

    model = keras.models.Model(
        inputs=[text, pos_tokens, dep_tokens, ent_type_tokens, highlight_tokens], outputs=[out]
    )

    model.compile(
                   keras.optimizers.RMSprop(learning_rate=LR),        # RMSprop for time_select
                   loss="binary_crossentropy",
                   metrics=["accuracy"],
    )
    print(model.summary())

    return model
