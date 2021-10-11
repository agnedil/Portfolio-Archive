import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'        # comment this out if you want to use GPU
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import spacy
import en_core_web_lg
import random
import json
import psycopg2
from sqlalchemy import create_engine
import time
import datetime
import matplotlib.pyplot as plt
import gc
import sys
import select_utils
from select_utils import VOCAB_SIZE, OUTPUT_DIM, BATCH_SIZE, UNITS, EPOCHS, REDUCE_LR, PREPROCESS
from select_utils import pos_table, dep_table, my_ent_table
from select_utils import plot_confusion_matrix, generate_features, get_other_highlights, get_x, oversample
#from multiprocessing import Pool
#from tqdm import tqdm


# GPU AVAILABLE?
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def cnn_tower(inputs, filters, widths):
    t1 = keras.layers.Conv1D(filters, widths[0], padding="same")(inputs)
    t1 = keras.layers.BatchNormalization()(t1)
    t1 = keras.layers.LeakyReLU()(t1)
    t2 = keras.layers.Conv1D(filters, widths[1], padding="same")(inputs)
    t2 = keras.layers.BatchNormalization()(t2)
    t2 = keras.layers.LeakyReLU()(t2)
    t3 = keras.layers.Conv1D(filters, widths[2], padding="same")(inputs)
    t3 = keras.layers.BatchNormalization()(t3)
    t3 = keras.layers.LeakyReLU()(t3)
    co = keras.layers.Concatenate()([t1, t2, t3])
    return co


def build_model( VOCAB_SIZE,
                 input_length,
                 output_dim,
                 num_pos,
                 num_deps,
                 num_ent_types,
                 num_highlights,
                 LR,
                 units,
                 dropout_rate1,
                 dropout_rate2,
                ):


    vectorizer = keras.layers.Embedding( input_dim    = VOCAB_SIZE+2,
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
    pos_embed  = keras.layers.Embedding( num_pos+2,
                                         output_dim,
                                         name='pos_embed',
                                       )(pos_tokens)

    dep_tokens = keras.Input( shape=[input_length],
                              name='dep',
                            )
    dep_embed  = keras.layers.Embedding( num_deps+2,
                                         output_dim,
                                         name='dep_embed',
                                       )(dep_tokens)

    ent_type_tokens = keras.Input( shape=[input_length],
                                   name="ent_type",
                                 )
    ent_type_embed = keras.layers.Embedding( num_ent_types+2,
                                             output_dim,
                                             name='ent_type_embed',
                                           )(ent_type_tokens)

    highlight_tokens = keras.Input( shape=[input_length],
                                    name='highlights',
                                  )
    highlight_embed  = keras.layers.Embedding( num_highlights+2,
                                               output_dim,
                                               name='highlight_embed',
                                             )(highlight_tokens)

    concat = keras.layers.concatenate([ text_vectorized,
                                        pos_embed,
                                        dep_embed,
                                        ent_type_embed,
                                        highlight_embed,
                                      ])

    cnn1 = cnn_tower(concat, 16, [1, 2, 5])
    cnn2 = cnn_tower(cnn1, 32, [1, 2, 5])
    cnn2 = cnn_tower(cnn2, 64, [1, 2, 5])
    cnn2 = cnn_tower(cnn2, 32, [1, 2, 5])
    cnn2 = cnn_tower(cnn2, 32, [1, 2, 5])
    cnn2 = cnn_tower(cnn2, 32, [1, 2, 5])
    rnn = keras.layers.Bidirectional(keras.layers.GRU(8))(cnn2)
    rnn = keras.layers.Dropout(0.1)(rnn)
    #rnn = keras.layers.Bidirectional(keras.layers.GRU(16), merge_mode='sum')(cnn2)
    #rnn = keras.layers.Bidirectional(keras.layers.LSTM(32))(rnn)
    out = keras.layers.Dense(1, activation="sigmoid")(rnn)

    model = keras.models.Model(
        inputs=[text, pos_tokens, dep_tokens, ent_type_tokens, highlight_tokens], outputs=[out]
    )

    model.compile(
                   keras.optimizers.RMSprop(learning_rate=0.0005),        # RMSprop for time_select
                   loss="binary_crossentropy",
                   metrics=["accuracy"],
    )
    print(model.summary())

    return model


# LOAD AND PREPARE DATA
user   = ''
pwd    = ''
host   = ''
port   = ''
db     = ''
engine = create_engine(f'postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}')

if PREPROCESS:

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
                             'label_entity': 'best_ent',
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

    # ARE THERE CASES WHEN BEST ENT NOT AMONG ENTITIES?
    print( 'Cases when best entity is (True) / is not (False) among my entities:' )
    def check_best_ent( row ):
        return row['best_ent'][0] in row['entities']
    df['no_best_ent_in_ents'] = df.apply( check_best_ent, axis=1 )
    print( df['no_best_ent_in_ents'].value_counts() )

    # DROP ARTICLES WITH UNUSUALLY HIGH NUMBER OF MY ENTITIES - SMALL FRACTION OF DATA
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
    no_best_ent = [ idx for idx, i in enumerate(df['main_highlight_feat'].values) if not any([j > 1 for j in i]) ]
    print( 'Number of articles without the best entity:', len( no_best_ent ) )
    df.loc[ no_best_ent, 'target'] = 0
    print( 'Breakdown by having the best entity:\n', df['target'].value_counts(), sep='' )

    # GET HIGHLIGHTS FOR NEGATIVE EXAMPLES (WHEN IT'S NOT THE MAIN ENTITY OF EVENT)
    df['other_highlights'] = df.apply( get_other_highlights, axis=1 )
    df.to_pickle('./data/20210611_df_my_ents_as_ent_layer.pkl')

else:

    # IF NO DB CONNECTION, LOAD PRE-GENERATED DATA
    print( '\nSkipping preprocessing. Loading the data preprocessed earlier ....' )
    df = pd.read_pickle('./data/20210611_df_my_ents_as_ent_layer.pkl')
    print('Data loaded. Data size:', df.shape)
    print('Unique articles:', len( df['body'].unique() ))
    print('Null values:\n', df.isna().sum(), sep='')


random_state = 25
train = df.sample( frac=0.95, random_state=random_state )
val   = df.drop( train.index )
print( 'Original size of train, val sets:\n', train.shape, val.shape )

datasets = []
for dset in [ train, val ]:
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
print( 'Final size of train, val sets:\n', len(X_train), len(y_train), len(X_val), len(y_val) )

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

# CHECK INTEGRITY OF Y
print(y_train[:10], y_train[-10:])
print(y_val[:10],  y_val[-10:])

del df, train, val
gc.collect()


LRs            = [ 0.0001, 0.0005, 0.001  ]
#dropout_rates = [ (0.0,0.0), (0.1,0.1), (0.2,0.2), (0.3, 0.1), (0.1, 0.3), (0.2,0.5), (0.5,0.2) ]
dropout_rates = [ (0.0,0.0), (0.1,0.1), ]

for LR in LRs:
    for dout1, dout2 in dropout_rates:

        input_length   = 300
        output_dim     = OUTPUT_DIM
        num_pos        = len(pos_table)
        num_deps       = len(dep_table)
        num_ent_types  = len(my_ent_table)
        num_highlights = 2
        tf.random.set_seed(23)

        model = build_model( VOCAB_SIZE,
                             input_length,
                             output_dim,
                             num_pos,
                             num_deps,
                             num_ent_types,
                             num_highlights,
                             LR,
                             UNITS,
                             dout1,
                             dout2,
                           )

        # CREATE DIRECTORY TO SAVE MODEL ARTIFACTS IF IT DOESN'T EXIST
        path_to_models = os.path.join( os.getcwd(), 'models' )
        if not os.path.exists(path_to_models) or not os.path.isdir(path_to_models):
            os.makedirs( path_to_models, exist_ok=False )

        START_TIME = datetime.datetime.now()
        checkpoint = keras.callbacks.ModelCheckpoint(
            START_TIME.strftime('models/%Y%m%dT%H%M-CNN-UPSAMPLE-ckpt-epoch{epoch:02d}-val_loss{val_loss:.2f}-val_accu{val_accuracy:.2f}-LR') + f'{LR}-dropout1{dout1}-dropout2{dout2}.hdf5',
                #monitor='val_accuracy',
                #mode='max',
                save_best_only=False,
                verbose=1,
            )
        if REDUCE_LR:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    mode='min',
                    factor=0.25,
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
                monitor='val_loss',
                mode='min',
                min_delta=0,
                patience=4,
                baseline=None,                                             # training stops if no improvement over baseline
                restore_best_weights=False,
                verbose=1,
            )

        if REDUCE_LR:
            callbacks=[ reduce_lr, checkpoint, csv_logger, early_stop ]
        else:
            callbacks=[ checkpoint, csv_logger, early_stop ]

        print(f'Fitting the model with OUTPUT DIMENSION {OUTPUT_DIM}, BATCH_SIZE {BATCH_SIZE}, and {UNITS} UNITS')


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
                            callbacks=callbacks,
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


        score = model.evaluate(
            x=[
                    keras.preprocessing.sequence.pad_sequences(
                        np.asarray([i[0] for i in X_val]), truncating="post", maxlen=300
                    ),
                    keras.preprocessing.sequence.pad_sequences(
                        np.asarray([i[1] for i in X_val]), truncating="post", maxlen=300
                    ),
                    keras.preprocessing.sequence.pad_sequences(
                        np.asarray([i[2] for i in X_val]), truncating="post", maxlen=300
                    ),
                    keras.preprocessing.sequence.pad_sequences(
                        np.asarray([i[3] for i in X_val]), truncating="post", maxlen=300
                    ),
                    keras.preprocessing.sequence.pad_sequences(
                        np.asarray([i[4] for i in X_val]), truncating="post", maxlen=300
                    ),
                ],
                y=np.asarray(y_val),
                batch_size=BATCH_SIZE,
                verbose = 1,
        )

        print("Final Score:   ", score[0])
        print("Final Accuracy:", score[1])

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
