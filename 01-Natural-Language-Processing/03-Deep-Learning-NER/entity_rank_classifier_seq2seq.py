my_entimport os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'        # comment this out if you want to use GPU
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
import psycopg2
from sqlalchemy import create_engine
import time
import datetime
from collections import Counter
import matplotlib.pyplot as plt
import fire
import gc
import sys
import select_utils
from select_utils import VOCAB_SIZE, OUTPUT_DIM, BATCH_SIZE, EPOCHS
from select_utils import pos_table, dep_table, my_ent_table
from select_utils import plot_confusion_matrix, generate_features, get_other_highlights, get_x, oversample, build_model
#from multiprocessing import Pool
#from tqdm import tqdm


# GPU AVAILABLE?
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# LOAD AND PREPARE DATA
user   = ''
pwd    = ''
host   = ''
port   = ''
db     = ''
engine = create_engine(f'postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}')

try:

    # LOAD DATA FROM DB
    print( '\nLoading data from database ....' )
    query              = '''select * from table'''
    df                 = pd.read_sql( query, engine )
    df['article_id']   = df['article_id'].astype( str )
    df['label_entity'] = df['label_entity'].apply( lambda x: [json.loads(x)] )
    df['entities']     = df['entities'].apply( json.loads )
    df = df.rename( columns={
                               'article_id': 'id',
                             'article_body': 'body',
                             'label_entity': 'best_entity',
                            })
    print('Data loaded. Data size:', df.shape)
    print('Unique articles:', len( df['body'].unique() ))
    print('Null values:\n', df.isna().sum(), sep='')

    # DROP DUPLICATES IN ARTICLE_BODY
    print( 'Data size:', df.shape )
    print( 'Dropping duplicated articles ....' )
    df = df.drop_duplicates( subset=['body'],
                             ignore_index=True )
    print( 'Data size: ', df.shape )

    #nlp = spacy.load("en_core_web_lg")    # didn't work in ML box - global path?
    nlp = en_core_web_lg.load()

    # TOKENIZE WITH SPACY, LIMIT # TOKENS TO 300
    print('\nTokenizing....')
    start = time.time()
    df['doc'] = df['body'].apply( lambda x: nlp(x) )
    df['all_tokens'] = df['doc'].apply( lambda x: [t.text for t in x] )
    df['all_tokens'] = df['all_tokens'].apply( lambda x: x[:300] if len(x)>300 else x )
    end = time.time()
    print(f'Tokenization complete. Time elapsed: {round((end-start)/60, 4)} min')

    # INSPECT VOCABULARY - a LOT of capitalized words, some mixed alphanums (5-feet, 2-storey, etc.)
    all_tokens   = set([ item for sublist in df['all_tokens'].tolist() for item in sublist ])
    alpha_tokens = { i for i in all_tokens if i.isalpha() }
    other        = { i for i in all_tokens if i not in alpha_tokens }
    if len(all_tokens) > VOCAB_SIZE:
        temp = input( f'Vocabulary size = {VOCAB_SIZE} less than the number of tokens = {len(all_tokens)}. Type new vocabulary size ("0" for no change):' )
        try:
            temp = int(temp)
            if temp != 0 and temp > 0:
                VOCAB_SIZE = temp
        except:
            pass
    print( 'Vocabulary size:     ', VOCAB_SIZE )
    print( 'Number of tokens:    ', len(all_tokens) )
    print( 'Words:               ', len(alpha_tokens) )
    print( 'Alphanumeric tokens: ', len(other) )

    # DECREASE MEMORY FOOTPRINT
    del all_tokens, alpha_tokens, other
    df = df.drop('all_tokens', axis=1)
    gc.collect()

    # GET ONLY MY ENTITIES
    df['my_ents'] = df['entities'].apply( lambda x: [e for e in x if e['type'] in my_ent_table] )

    # ARE THERE CASES WHEN BEST ENTITY NOT AMONG ENTITIES?
    print( 'Cases when best entity is (True) / is not (False) among other entities:' )
    def check_best_entity( row ):
        return row['best_entity'][0] in row['entities']
    df['no_best_entity_in_ents'] = df.apply( check_best_entity, axis=1 )
    print( df['no_best_entity_in_ents'].value_counts() )

    # DROP ARTICLES WITH UNUSUALLY HIGH NUMBER OF THE ENTITIES - SMALL FRACTION OF DATA
    print( '\nDropping articles with > 25 my ents', '\nData size before the drop: ', df.shape, sep='' )
    df['num_my_ents'] = df['my_ents'].apply( lambda x: len(x) )
    df = df[ df['num_my_ents'] <= 26 ].reset_index( drop=True )
    print( 'Data size after:           ', df.shape )

    # GET FEATURES + TARGET
    print( 'Preparing features ....' )
    feat_cols = [ 'text_feat', 'pos_feat', 'dep_feat',
                  'ent_type_feat', 'main_highlight_feat',
                  'target']
    for col in feat_cols:
        df[col] = np.nan
    df['target'] = 1
    df = df.apply( generate_features, axis=1 )

    # CHECK IF THERE ARE CASES WITHOUT THE TARGET
    no_best_entity = [ idx for idx, i in enumerate(df['main_highlight_feat'].values) if not any([j > 1 for j in i]) ]
    print( 'Number of articles without the best entity:', len( no_best_entity ) )
    df.loc[ no_best_entity, 'target'] = 0
    print( 'Breakdown by having the best entity:\n', df['target'].value_counts(), sep='' )

    # GET HIGHLIGHTS FOR NEGATIVE EXAMPLES (WHEN IT'S NOT THE MAIN MY ENTITY OF EVENT)
    df['other_highlights'] = df.apply( get_other_highlights, axis=1 )

except:

    # IF NO DB CONNECTION, LOAD PRE-GENERATED DATA
    print( '\nDatabase not available. Loading pre-generated data ....' )
    df = pd.read_pickle('./data/20210610_df_my_ents_as_ent_layer.pkl')
    print('Data loaded. Data size:', df.shape)
    print('Unique articles:', len( df['body'].unique() ))
    print('Null values:\n', df.isna().sum(), sep='')


random_state = 25
train = df.sample( frac=0.8, random_state=random_state )
val   = df.drop( train.index )
test  = val.sample( frac=0.5, random_state=random_state )
val   = val.drop( test.index )
print( 'Original size of train, val, test sets:\n', train.shape, val.shape, test.shape )

datasets = []
for dset in [ train, val, test ]:
    X_pos, X_neg = get_x( dset )
    diff  = int( len(X_neg)-len(X_pos) )
    assert diff >= 0, f'Negative difference: {diff}'
    X_pos = oversample( X_pos, diff )
    random.shuffle( X_pos )
    random.shuffle( X_neg )

    y_pos, y_neg = [1]*len(X_pos), [0]*len(X_neg)
    X, y = X_pos+X_neg, y_pos+y_neg
    X, y = sklearn.utils.shuffle( X, y, random_state=random_state )
    datasets.append( (X,y) )

X_train, y_train = datasets[0]
X_val, y_val     = datasets[1]
X_test, y_test   = datasets[2]
print('Final size of train, val, test sets:\n', len(X_train), len(y_train), len(X_val), len(y_val), len(X_test), len(y_test) )

# CHECK INTEGRITY OF X
def check_integrity( X, name_ ):
    for idx, i in enumerate(X):
        assert isinstance(i, list) and len(i) == 5, f'No list at idx {idx}: {i}'
        for idx2, j in enumerate(i):
            assert isinstance(j, list) and len(j) > 25, f'No list at idx {idx}, 2-level idx {idx2}: {j}'
            for idx3, k in enumerate(j):
                assert isinstance(k, int) and k>=0, f'Third level incomplete: {k}'
    print('Integrity confirmed for', name_)

print('Checking integrity of X and y ....')
check_integrity( X_train, 'X_train' )
check_integrity( X_val,  'X_val' )
check_integrity( X_test,  'X_test' )

# CHECK INTEGRITY OF Y
print(y_train[:10], y_train[-10:])
print(y_val[:10],  y_val[-10:])
print(y_test[:10],  y_test[-10:])

del df, train, val, test
gc.collect()

input_length   = 300
output_dim     = OUTPUT_DIM
num_pos        = len(pos_table) + 1
num_deps       = len(dep_table) + 1
num_ent_types  = len(my_ent_table) + 1
num_highlights = 3
tf.random.set_seed(23)

model = build_model( VOCAB_SIZE,
                     input_length,
                     output_dim,
                     num_pos,
                     num_deps,
                     num_ent_types,
                   )

# CREATE DIRECTORY TO SAVE MODEL ARTIFACTS IF IT DOESN'T EXIST
path_to_models = os.path.join( os.getcwd(), 'models' )
if not os.path.exists(path_to_models) or not os.path.isdir(path_to_models):
    os.makedirs( path_to_models, exist_ok=False )

START_TIME = datetime.datetime.now()
checkpoint = keras.callbacks.ModelCheckpoint(
        START_TIME.strftime('models/%Y%m%dT%H%M-ckpt-epoch{epoch:02d}-val_accu{val_accuracy:.2f}_bestent_all_upsampled.hdf5'),
        #monitor='val_accuracy',
        #mode='max',
        save_best_only=False,
        verbose=1,
    )
reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=0.2,
        patience=2,
        min_lr=0.00005,
        verbose=1,
    )
csv_logger = keras.callbacks.CSVLogger(
        START_TIME.strftime('models/%Y%m%dT%H%M_log_all_data.tsv'),
        append=True,
        separator='\t',
    )
early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        min_delta=0,
        patience=4,
        baseline=None,                                             # training stops if no improvement over baseline
        restore_best_weights=False,
        verbose=1,
    )

history = model.fit(
                    x=[
                        keras.preprocessing.sequence.pad_sequences(
                            np.asarray([ i[0] for i in X_train ]), truncating="post", maxlen=300
                        ),
                        keras.preprocessing.sequence.pad_sequences(
                            np.asarray([ i[1] for i in X_train ]), truncating="post", maxlen=300
                        ),
                        keras.preprocessing.sequence.pad_sequences(
                            np.asarray([ i[2] for i in X_train ]), truncating="post", maxlen=300
                        ),
                        keras.preprocessing.sequence.pad_sequences(
                            np.asarray([ i[3] for i in X_train ]), truncating="post", maxlen=300
                        ),
                        keras.preprocessing.sequence.pad_sequences(
                            np.asarray([ i[4] for i in X_train ]), truncating="post", maxlen=300
                        ),
                    ],
                    y=np.asarray(y_train),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=[ reduce_lr, checkpoint, csv_logger, early_stop ],
                    validation_data=(
                    [
                        keras.preprocessing.sequence.pad_sequences(
                            np.asarray([ i[0] for i in X_val ]), truncating="post", maxlen=300
                        ),
                        keras.preprocessing.sequence.pad_sequences(
                            np.asarray([ i[1] for i in X_val ]), truncating="post", maxlen=300
                        ),
                        keras.preprocessing.sequence.pad_sequences(
                            np.asarray([ i[2] for i in X_val ]), truncating="post", maxlen=300
                        ),
                        keras.preprocessing.sequence.pad_sequences(
                            np.asarray([ i[3] for i in X_val ]), truncating="post", maxlen=300
                        ),
                        keras.preprocessing.sequence.pad_sequences(
                            np.asarray([ i[4] for i in X_val ]), truncating="post", maxlen=300
                        ),
                    ],
                    np.asarray(y_val) ),
    )


model2 = model
score = model2.evaluate(
    x=[
            keras.preprocessing.sequence.pad_sequences(
                np.asarray([i[0] for i in X_test]), truncating="post", maxlen=300
            ),
            keras.preprocessing.sequence.pad_sequences(
                np.asarray([i[1] for i in X_test]), truncating="post", maxlen=300
            ),
            keras.preprocessing.sequence.pad_sequences(
                np.asarray([i[2] for i in X_test]), truncating="post", maxlen=300
            ),
            keras.preprocessing.sequence.pad_sequences(
                np.asarray([i[3] for i in X_test]), truncating="post", maxlen=300
            ),
            keras.preprocessing.sequence.pad_sequences(
                np.asarray([i[4] for i in X_test]), truncating="post", maxlen=300
            ),
        ],
        y=np.asarray(y_test),
        batch_size=BATCH_SIZE,
        verbose = 1,
)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig( START_TIME.strftime('models/%Y%m%dT%H%M_model_accu_plot.png') )
plt.show()

# clear the plot
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig( START_TIME.strftime('models/%Y%m%dT%H%M_model_loss_plot.png') )
plt.show()


preds = model2.predict(
    x=[
            keras.preprocessing.sequence.pad_sequences(
                np.asarray([i[0] for i in X_test]), truncating="post", maxlen=300
            ),
            keras.preprocessing.sequence.pad_sequences(
                np.asarray([i[1] for i in X_test]), truncating="post", maxlen=300
            ),
            keras.preprocessing.sequence.pad_sequences(
                np.asarray([i[2] for i in X_test]), truncating="post", maxlen=300
            ),
            keras.preprocessing.sequence.pad_sequences(
                np.asarray([i[3] for i in X_test]), truncating="post", maxlen=300
            ),
            keras.preprocessing.sequence.pad_sequences(
                np.asarray([i[4] for i in X_test]), truncating="post", maxlen=300
            ),
        ]
)

labels = ['0', '1']
preds = [ np.round(np.mean(i), 0) for i in preds ]
class_report = classification_report(y_test, preds)
print( class_report )
with open(START_TIME.strftime('models/%Y%m%dT%H%M_classification_report.txt'), 'w') as f:
    f.write( class_report )


# PRINT THE CONFUSION MATRIX
cm = confusion_matrix(y_test, preds)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm, START_TIME, classes=labels)
plt.show()
