import os
import time
import pickle
import random
from random import randint
import itertools
import numpy as np
import pandas as pd
import boto3
import shutil
import logging
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Concatenate, Bidirectional, LSTM, Dropout, Dense


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3 = boto3.client('s3')
    try:
        with open(file_name, "rb") as f:
            s3.upload_fileobj(f, bucket, object_name)
    except Exception as e:
        logging.error(e)
        return False

    return True


def get_f05( p, r ):
    '''
        Compute F0.5 score that puts more emphasis on precision
    '''
    beta = 0.5
    if p==0 and r==0:
        return 0
    try:
        return ((1 + beta**2)*p*r) / (p*(beta**2) + r)
    except Exception as e:
        print(f'\nError computing F0.5 score: {e}\n')
        return 0


logging.info('Downloading preprocessed data....')
s3 = boto3.client('s3')
BUCKET_NAME = ''
OBJECT_NAME = ''
FILE_NAME   = ''
print(f'Downloading file "{FILE_NAME}" from directory "{OBJECT_NAME}" in bucket "{BUCKET_NAME}"\n')
with open(FILE_NAME, 'wb') as f:
    s3.download_fileobj(BUCKET_NAME, OBJECT_NAME, f)

loaded  = np.load(FILE_NAME)
X_train = loaded['X_train']
y_train = loaded['y_train']
X_test  = loaded['X_test']
y_test  = loaded['y_test']
X_val   = loaded['X_val']
y_val   = loaded['y_val']
embedding_matrix = loaded['embedding_matrix']

print('Printing y_train:', y_train[:25])
print('\nPrinting y_train:', y_test[:25])
print('\nPrinting y_train:', y_val[:25])


###########################  TRAIN MODEL ################################

seed_value = 47
#import os
#os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed( seed_value )
np.random.seed( seed_value )
tf.random.set_seed( seed_value )

epochs         = 21
learning_rates = [ 1e-3, ]
batch_sizes    = [ 4, ]
dropouts       = [ 0.5, ]
units_list     = [ 30, ]
optimizers     = [ Adam, ]                # Adam, RMSprop, Nadam

maxlen         = 88
vocab_size     = 10295
EMBED_SIZE     = 300
emb            = 'Spacy'

wdir           = 'ckpts'
logs_dir       = 'logs'
results_dir    = 'results'

for dir in [ wdir ]:                                                  # only log saved in wdir; results_dir not used
    if not os.path.exists(dir):
        print(f"Creating the {dir} folder")
        os.makedirs(dir)

all_combinations = list(itertools.product(*[learning_rates, batch_sizes, dropouts, units_list, optimizers]))
all_results      = dict()

for learning_rate, batch_size, dropout, units, optimizer in all_combinations:

    time_stamp = time.strftime("%Y%m%dT%H%M")
    file_name  = f'{wdir}/log_{time_stamp}.txt'

    with open( file_name, 'w', encoding='utf-8' ) as f:

        experiment_name = '2 BiLSTM, MULTICLASS\n'
        f.write( experiment_name )

        optimizer_name = optimizer.__module__.split('.')[-1].capitalize()
        params = f'\nEmbeddings={emb}, LR={learning_rate}, batch_size={batch_size}, dropout={dropout}, units={units}, optimizer={optimizer_name}\n'
        message = '\nTimestamp: ' + time_stamp + params
        print( message )
        f.write( message + '\n' )

        deep_inputs     = Input(shape=(maxlen,))
        embedding_layer = Embedding(vocab_size, EMBED_SIZE, weights=[embedding_matrix], trainable=False)(deep_inputs)
        LSTM_1          = Bidirectional(LSTM( units, dropout=dropout, return_sequences=True ))(embedding_layer)
        LSTM_2          = Bidirectional(LSTM( units, dropout=dropout, return_sequences=False ))(LSTM_1)
        #dense_1         = Dense(60, activation=None)(LSTM_1)
        dense_output    = Dense(15, activation='softmax')(LSTM_2)
        model           = Model(inputs=deep_inputs, outputs=dense_output)
        model.summary()


        model_metric = 'sparse_categorical_accuracy'
        model.compile( loss='sparse_categorical_crossentropy', optimizer=optimizer(lr=learning_rate),
                       metrics=[ model_metric ] )

        early_stop = tf.keras.callbacks.EarlyStopping(
                                                       monitor = 'val_' + model_metric,
                                                       mode='max',
                                                       patience=7,
                                                       #min_delta=0.0025,
                                                       restore_best_weights=True,
                                                       verbose=1,
                                                     )

        reduce_lr  = tf.keras.callbacks.ReduceLROnPlateau(
                                                           monitor='val_loss',
                                                           mode='min',
                                                           patience=4,
                                                           factor=0.2,
                                                           min_lr=1e-4,
                                                           verbose=2,
                                                         )

        filepath   = wdir + '/' + time_stamp + '-epoch{epoch:02d}-val_accu_{val_sparse_categorical_accuracy:.2f}-val_loss_{val_loss:.2f}.hdf5'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                                         filepath,
                                                         monitor='val_' + model_metric,
                                                         save_best_only=True,
                                                         mode='max',
                                                         verbose=0,
                                                       )

        history = model.fit( X_train,
                             y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=2,
                             validation_data=(X_val, y_val),
                             callbacks=[ early_stop, reduce_lr, checkpoint, ]     # checkpoint
                            )

        score        = model.evaluate(X_test, y_test, verbose=1)
        test_results = f'Val Score: {score[0]}\n\tVal Accuracy: {score[1]}\n'
        predictions  = model.predict(X_test)
        #predictions  = np.round( predictions )
        predictions  = np.argmax(predictions, axis=1)
        clf_report   = classification_report(y_test, predictions, digits=4)

        print( test_results, '\n', clf_report, '\n', '='*70 )
        f.write( test_results + '\n' + clf_report + '\n' + '='*70 + '\n' )


# SAVE ALL_RESULTS
#time_stamp = datetime.now().strftime('%Y%m%dT%H%M_%S%f')                         # includes milliseconds

#all_results_path = f'{results_dir}/{time_stamp}_all_results.pkl'
#with open(all_results_path, 'wb') as f:
#    pickle.dump( all_results, f, protocol=pickle.HIGHEST_PROTOCOL )

# CONVERT ALL_RESULTS INTO DF FOR EASY SORTING
#cols = [ 'cl0_prec', 'cl0_rec', 'cl0_f1', 'cl1_prec', 'cl1_rec', 'cl1_f1', 'cl1_f05', 'micro_f1', 'macro_f1', 'score', ]
#cols = cols + [i+'_reg' for i in cols]
#df_res = pd.DataFrame.from_dict( all_results, orient='index', columns=cols )
#df_res['params'] = df_res.index
#df_res = df_res.reset_index(drop=True)
#df_res = df_res[ ['params']+cols ]

#df_res.to_pickle(f'{results_dir}/{time_stamp}_df_res.pkl')
#df_res.to_csv(f'{results_dir}/{time_stamp}_df_res.tsv', sep='\t', encoding='utf-8', index=True)

# MOVE RESULTS TO S3
for DIR_NAME in [wdir]:                                                  # only log saved in wdir; results_dir not used

    FILE_NAME_SHORT = f'{DIR_NAME}_{time_stamp}'                         # using timestamp from training
    FILE_NAME_FULL  = FILE_NAME_SHORT + '.zip'
    BUCKET_NAME     = 'whiq-nlp-experiments'
    OBJECT_NAME     = f'andrew/experim/2022-01-01/results_{time_stamp}/' + FILE_NAME_FULL

    shutil.make_archive(  FILE_NAME_SHORT, 'zip', DIR_NAME )
    result = upload_file( FILE_NAME_FULL, BUCKET_NAME, OBJECT_NAME)
    #logging.info(f"{FILE_NAME} uploaded successfully: {result}")
    print(f"{FILE_NAME_FULL} uploaded successfully: {result}")
