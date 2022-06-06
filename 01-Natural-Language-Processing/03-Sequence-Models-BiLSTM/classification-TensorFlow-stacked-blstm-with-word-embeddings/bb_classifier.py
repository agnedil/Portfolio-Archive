import sys
#sys.path.insert(1, '/bbclf')
sys.path.append('bbclf/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string, itertools, time, os, re
import sklearn, spacy, pickle
import ftfy, emoji

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors
from sentiment import get_sentiment
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import ( Activation, Dropout, Dense, Flatten, LSTM, Bidirectional, SpatialDropout1D,
                                      RepeatVector, MaxPooling1D, GlobalMaxPooling1D, GlobalMaxPool1D, Embedding,
                                      Input, Concatenate, Reshape, Flatten, Conv1D, GlobalAveragePooling1D )



class BinaryClassifier:

    sw = [ 'a', 'an', 'the', 'of', ]
    punctuation = ''.join([c for c in string.punctuation if c not in "'!?"])

    contractions = (
                 ("I'm",    "I am"),
                 ("'ll ",   " will "),
                 ("'d ",    " would "),
                 ("'ve ",   " have "),
                 ("'re ",   " are "),
                 ("what's", "what is"),
                 ("that's", "that is"),
                 ("it's",   "it is"),
                 ("here's", "here is"),
                 ("let's",  "let us"),
                 ("'cause", "because"),
                 ("can't",  "cannot"),
                 ("shan't", "shall not"),
                 ("won't",  "will not"),
                 ("n't",    " not"),
                )

    embeddings_switch = { 0: 'Word2vec',
                          1: 'Glove',
                          2: 'spaCy',
                        }

    def __init__( self,
                  model_path     = None,
                  tokenizer_path = None,
                  max_len        = 100,
                ):
        '''
            Initiate an empty instance for training or
            load a trained model from file for inference
        '''
        # load model from file if model path is provided
        if model_path is not None:
            self.model   = keras.models.load_model( model_path )
            self.max_len = max_len
            self.nlp     = spacy.load('en_core_web_lg')
            self.nlp.remove_pipe('parser')
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load( f )
            print('Loaded model from file for inference')

        # initiate an empty model instance for training
        else:
            self.nlp = spacy.load('en_core_web_lg')
            self.nlp.remove_pipe('parser')
            print('Instantiated model for training')


    def fetch_data( self,
                    data_path,
                    augment=False,
                    oversample=False,
                   ):
        '''
            Generate train and test sets (text is not preprocessed)
        '''
        ml_categories = [ 'bias_age', 'bias_gender', 'imposing_value', 'imposing_work', 'negative_expectation',
                  'restricting', 'superiority', 'undesired_work', 'unk', 'subordination', 'implicit_criticism',
                  'inconsequential_work', 'bias_rank', 'negative_result', 'uncontrollable_work', ]
        df = pd.read_csv( data_path , sep='\t', encoding='utf-8' )
        df = df[ df['label'].isin( ml_categories ) ]
        df['target'] = df['label'].apply( lambda x: 0 if x == 'unk' else 1 )

        # DEDUPE BETWEEN CATEGORIES. FAVOR CATEGORY 1
        df1 = df[ df['target'] == 1 ].copy()
        df0 = df[ df['target'] == 0 ].copy()
        df0, df1 = self.dedupe( df0, df1, 'sentence' )
        df = pd.concat([ df0, df1 ]).copy().sample(frac=1).reset_index(drop=True)

        # DEDUPE TRAIN / VAL / TEST SETS. FAVOR TEST, THEN VAL SET
        df_train = df[ df['subset'] == 'train' ].copy()
        df_val   = df[ df['subset'] == 'val' ].copy()
        df_test  = df[ df['subset'] == 'test' ].copy()
        df_train, df_test = self.dedupe( df_train, df_test, 'sentence' )
        df_val, df_test   = self.dedupe( df_val, df_test, 'sentence' )
        df_train, df_val  = self.dedupe( df_train, df_val, 'sentence' )
        df = pd.concat([ df_train, df_val, df_test ]).copy().sample(frac=1).reset_index(drop=True)

        if augment:
            df = self.augment_df(df)

        if oversample:
            df = self.oversample_df(df)

        # TRAIN TEST SPLIT AND SHUFFLE
        X_train = df[ df['subset'].isin([ 'train', 'val' ]) ]['sentence'].values
        y_train = df[ df['subset'].isin([ 'train', 'val' ]) ]['target'].values
        X_test  = df[ df['subset'].isin([ 'test' ]) ]['sentence'].values
        y_test  = df[ df['subset'].isin([ 'test' ]) ]['target'].values

        X_train, y_train = sklearn.utils.shuffle( X_train, y_train )
        X_test, y_test   = sklearn.utils.shuffle( X_test, y_test )
        print('Data fetched!')

        return X_train, y_train, X_test, y_test


    def fit( self,
             X_train,
             y_train,
             maxlen        = 100,
             learning_rate = 1e-3,
             batch_size    = 8,
             dropout       = 0.3,
             units         = 150,
             wdir          = 'checkpoints/',
             ):
        '''
            Train a new model
        '''
        # create directory to save model files to
        cwd  = os.getcwd()
        wdir = os.path.join( cwd, wdir )
        if not os.path.exists( wdir ):
            os.makedirs( wdir )

        print('Preprocessing text')
        X_train = [ self.preprocess(t) for t in X_train ]
        new_maxlen = max( [len(i.split()) for i in X_train] )
        maxlen     = max(maxlen, new_maxlen)
        self.max_len = maxlen

        # KERAS TOKENIZER
        print('Tokenizing text')
        self.tokenizer = Tokenizer( num_words=6500,
                                    lower=True,
                                    oov_token='oov',
                                    filters='"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n',        # removed '!' and '?'
                                  )
        self.tokenizer.fit_on_texts(X_train)
        X_train = self.tokenizer.texts_to_sequences(X_train)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print('Vocabulary size:', self.vocab_size)

        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

        switch = 2
        embedding_matrix, EMBED_SIZE = self.get_embeddings(switch)

        optimizer      = Adam
        emb            = self.embeddings_switch[ switch ]
        time_stamp     = time.strftime("%Y%m%dT%H%M")
        optimizer_name = optimizer.__module__.split('.')[-1].capitalize()
        params = f'\nEmbeddings={emb}, LR={learning_rate}, batch_size={batch_size}, dropout={dropout}, units={units}, optimizer={optimizer_name}'
        print( 'Classifier parameters:', params )
        print( 'Timestamp:', time_stamp)

        deep_inputs     = Input(shape=(maxlen,))
        embedding_layer = Embedding(self.vocab_size, EMBED_SIZE, weights=[embedding_matrix], trainable=False)(deep_inputs)
        LSTM_1          = Bidirectional(LSTM( units, dropout=dropout, return_sequences=True ))(embedding_layer)
        gmp1d           = GlobalMaxPool1D()(LSTM_1)
        dense_layer     = Dense(1, activation='sigmoid')(gmp1d)
        self.model      = Model(inputs=deep_inputs, outputs=dense_layer)


        self.model.compile( loss='binary_crossentropy', optimizer=optimizer(lr=learning_rate), metrics=['accuracy'] )

        early_stop = tf.keras.callbacks.EarlyStopping(
                                                       monitor='val_accuracy',
                                                       patience=5,
                                                       restore_best_weights=True,
                                                       verbose=2,
                                                     )

        reduce_lr  = tf.keras.callbacks.ReduceLROnPlateau(
                                                           monitor="val_loss",
                                                           patience=2,
                                                           factor=0.2,
                                                           min_lr=5e-5,
                                                           verbose=2,
                                                         )

        filepath   = wdir + time_stamp + '-epoch{epoch:02d}-val_accu_{val_accuracy:.2f}-val_loss_{val_loss:.2f}.hdf5'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                                            filepath,
                                                            verbose=0,
                                                          )

        history = self.model.fit( X_train,
                             y_train,
                             batch_size=batch_size,
                             epochs=21,
                             verbose=2,
                             validation_split=0.2,
                             callbacks=[ early_stop, reduce_lr, checkpoint ]
                            )

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.show()

        with open(f'{wdir}/{time_stamp}_tokenizer.pkl', 'wb') as f:
            pickle.dump( self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL )


    def evaluate( self,
                  X_test,
                  y_test,
                  verbose=1,
                  do_clean=True,
                  print_stats=False,
                ):
        '''
            Evaluate model on raw texts
        '''
        X_test        = [ self.preprocess(t, do_clean) for t in X_test ]
        X_sentim_test = [self.infer_sentiment(t) for t in X_test]
        X_sentim_test = pad_sequences(X_sentim_test, padding='post', truncating='post', maxlen=self.max_len)

        X_test = self.tokenizer.texts_to_sequences( X_test )
        X_test = pad_sequences(X_test, padding='post', truncating='post', maxlen=self.max_len)

        if print_stats:
            # PREDICT ALL
            predictions = self.model.predict(X_test)
            predictions = np.round( predictions )
            clf_report = classification_report(y_test, predictions, digits=4)
            print( 'Classification Report:\n\n', clf_report, sep='' )

        return self.model.evaluate([X_test, X_sentim_test], y_test, verbose=verbose)


    def predict(self, texts, do_clean=True):
        '''
            Make prediction on raw texts
        '''
        texts      = [ self.preprocess(t, do_clean) for t in texts ]
        sentiments = [self.infer_sentiment(t) for t in texts]
        sentiments = pad_sequences(sentiments, padding='post', truncating='post', maxlen=self.max_len)

        texts = self.tokenizer.texts_to_sequences( texts )
        texts = pad_sequences(texts, padding='post', truncating='post', maxlen=self.max_len)

        return self.model.predict([texts, sentiments])


    def preprocess(self, sentence, do_clean=True):
        '''
            Preprocess text before prediction
        '''
        if do_clean:
            sentence = self.clean_text( sentence )
        sentence = self.convert_emoticons( sentence )
        sentence = self.remove_emoji( sentence, to_text=True )
        sentence = self.unfold_contractions( sentence )
        sentence = self.mask_entities( sentence )
        sentence = sentence.lower()
        sentence = self.remove_non_alpha( sentence )
        sentence = self.remove_stopwords( sentence )

        return sentence


    def augment_df(self, df_):
        '''
            Use data points from Annotation Guidelines and
            pattern model's keywords to augment data
        '''
        # LOAD SENTENCES FROM ANNOTATION GUIDELINES
        print( 'Augmenting data....', end=' ', )
        file = 'bbclf/data/20210927_additional_dataset_from_guidelines.xlsx'
        df_aug = pd.read_excel( file, sheet_name='guidelines' )
        df_aug = df_aug.dropna()
        df_aug['target'] = 1
        df_aug['subset'] = 'train'
        df_aug['award_text'] = df_aug['sentence'].values
        df_aug = df_aug[['award_text', 'sentence', 'target', 'label', 'subset', 'source']]
        for col in df_.columns:
            if col not in df_aug.columns:
                df_aug[col] = np.nan

        # LOAD KEYWORDS FROM CURRENT PATTERNS MODEL
        file = 'bbclf/data/kw.xlsx'
        kw = pd.read_excel( file )
        kw = kw.rename( columns={'keyword': 'sentence'} )
        kw = kw.dropna()
        kw['target'] = 1
        kw['source'] = 'guide'
        kw['subset'] = 'train'
        kw['award_text'] = kw['sentence'].values
        kw = kw[['award_text', 'sentence', 'target', 'label', 'subset', 'source']]
        for col in df_.columns:
            if col not in kw.columns:
                kw[col] = np.nan

        df_ = pd.concat([ df_, kw, df_aug ])
        df_ = df_.sample( frac=1 ).reset_index(drop=True)
        print('Done! Shape of all data after augmentation:', df_.shape)
        return df_


    def oversample_df(self, df_):
        '''
            Upsample all minority bias categories for more even representation;
            avoid upsampling from data with source reeval_2021 (it's not arbitrated)
        '''
        return df_


    def dedupe(self, df1, df2, col_):
        '''
            Delete every entry in df1's column col_ if it occurs in df2's column col_
        '''
        original_length = df1.shape[0]
        df2_sents = df2[col_].values
        df1 = df1[ ~df1[col_].isin(df2_sents) ]
        print( f'\tDropping {original_length - df1.shape[0]} duplicates')
        return df1, df2


    def upsample( self, df_, to_oversample_ ):
        '''
            Upsample df_ by to_oversample_ more samples excluding re-evaluation data
        '''
        # EXCLUDE RE-EVALUATED DATA FROM OVERSMPLING IF IT'S PRESENT
        if 'source' in df_.columns:
            df_to_oversample = df_[ df_['source'] != 'reeval_2021' ]
            print( '\tData shape for this category without re-eval:', df_to_oversample.shape )
        else:
            df_to_oversample = df_

        # OVERSAMPLE AND CONCAT W/ORIGINAL DF_
        replace = False
        if len(df_to_oversample) < to_oversample_:
            replace = True

        df_upsampled = df_to_oversample.sample( n=to_oversample_, replace=replace )
        df_          = pd.concat([ df_, df_upsampled ])

        return df_.sample( frac=1 )


    def convert_sentiment(self, l):
        '''
            Convert list of sentiment scores into 20 bins to build embeddings later
        '''
        return [ round(i*10) + 10 for i in l ]


    def infer_sentiment(self, s, pos=False):
        '''
        Get sentiment scores as a dict only for select POS, and 0 score for rest
        '''
        #pos_included = ['ADJ', 'NOUN', 'PROPN', 'INTJ']
        pos_included = ['ADJ', 'NOUN', 'PROPN', 'VERB', 'ADV', 'INTJ']
        s = self.nlp(s)
        sentiments = get_sentiment(s)
        if pos:
            for t in s:
                if t.text.lower() in sentiments and t.pos_ not in pos_included:
                    sentiments[ t.text ] = None
                if t.lemma_.lower() in sentiments and t.pos_ not in pos_included:
                    sentiments[ t.lemma_ ] = None
        sentiments = {k: v for k,v in sentiments.items() if v}

        res = []
        for t in s:
            res.append( sentiments.get( t.text.lower(), 0.0 ) )

        return self.convert_sentiment( res )


    def repair_text(self, s):
        '''
            Clean up encoding, HTML leftovers, and other issues;
            full list of parameters - to enable flexibility when we need it.
            Examples:
                     "L&AMP;AMP;ATILDE;&AMP;AMP;SUP3;PEZ" ==> "LóPEZ"
                     "schÃ¶n" ==> "schön"
        '''
        return ftfy.fix_text(
                               s,
                               fix_encoding=True,
                               restore_byte_a0=True,
                               replace_lossy_sequences=True,
                               decode_inconsistent_utf8=True,
                               fix_c1_controls=True,
                               unescape_html=True,
                               remove_terminal_escapes=True,
                               fix_latin_ligatures=True,
                               fix_character_width=True,
                               uncurl_quotes=True,
                               fix_line_breaks=True,
                               fix_surrogates=True,
                               remove_control_chars=True,
                               normalization='NFC',
                               explain=False,
                            )


    def clean_text( self, s,
                    to_ascii=False,
                  ):
        '''
            NOTE:  THIS FUNCTION APPLIES ONLY LIGHT GENERAL NON-DESTRUCTUVE TEXT CLEANING
                   AS PART OF AN ETL PIPELINE. IT IS NOT MEANT TO DO TEXT PROCESSING FOR
                   A TEXT CLASSIFIER WHICH SHOULD BE A CLASSIFIER-SPECIFIC ACTIVITY
        '''
        # edge case
        if not isinstance(s, str) or not s:
            return s

        # TODO: need list of such special characters that evade repair_text()
        for char in ['�', '•']:
            if char in s:
                s = s.replace(char, '')

        # fix text encoding
        s = self.repair_text( s )

        # convert to ascii
        if to_ascii:
            try:
                s = s.encode('ascii', 'ignore').decode()
            except:
                pass

        # remove multiple spaces
        s = re.sub('\s+', ' ', s)

        return s.strip()


    def mask_entities(self, s):
        '''
            Replace simplified named entities with their type,
            if a word is named entity; otherwise, lemmatize word
        '''
        ent_types = {
                        'number':       ['CARDINAL', 'ORDINAL'],
                        'place':        ['LOC'],
                        'name':         ['PERSON'],
                        'percent':      ['PERCENT'],
                        'money':        ['MONEY'],
                        'organization': ['ORG'],
                    }

        # replace entities w/generic names, lemmatize
        out = []
        s = self.nlp(s)
        for t in s:
            ent_type = ''
            for key in ent_types:
                if t.ent_type_ in ent_types[key]:
                    ent_type = key
                    break
            if ent_type:
                out.append( ent_type )
            else:
                out.append( t.lemma_ )

        # remove entity masks repeated several times in a row
        stack = ['',]
        for item in out:
            if item == stack[-1]:
                continue
            stack.append( item )

        return ' '.join( stack[1:] )


    def convert_emoticons(self, s):
        '''
            Placeholder for a more robust method
        '''
        return s.replace(':)', 'happy_face').replace(':D', 'happy_face')


    def remove_emoji(self, s, to_text=False):
        '''
            Replace emojis in string s with their corresponding textual description if to_text=True;
            otherwise, delete all amojis
        '''
        # convert emojis to text                 - increases run time from 0.4 to 5 min. on 250K messages
        if to_text:
            s = emoji.demojize( s )

        # or delete emojis
        else:
            for e in emoji.UNICODE_EMOJI['en']:
                if e in s:
                    s = s.replace(e, '')

        return s


    def unfold_contractions(self, s ):
        '''
            Unfold common English contractions: e.g. "I'm" => "I am"
        '''
        for i,j in self.contractions:
            if i in s:
                s = s.replace(i,j)

        return s


    def remove_non_alpha(self, s):
        '''
        To avoid words being glued together, replace punct with spaces because
        there are complex words with '-' separator,
        and some people forget there should be a space after punctuation
        '''
        s = s.translate( str.maketrans( self.punctuation, ' '*len(self.punctuation) ) )
        s = s.translate( str.maketrans( string.digits, ' '*len(string.digits) ) )
        s = re.sub('\s+', ' ', s)
        return s.strip()


    def remove_stopwords(self, s):
        s = s.strip().split()
        s = [ w.strip() for w in s if self.remove_non_alpha(w).lower() not in self.sw ]
        return ' '.join(s)


    def get_embeddings(self, switch):

        # Embeddings Switch: 0 - Word2vec, 1 - Glove, 2 - Spacy
        print( 'Building embedding matrix ...', end=' ' )
        if switch == 0:

            # WORD2VEC
            word_vectors = KeyedVectors.load_word2vec_format('./pretrained_embeddings/GoogleNews-vectors-negative300.bin', binary=True)

            EMBED_SIZE=300
            embedding_matrix = np.zeros((self.vocab_size, EMBED_SIZE))
            for word, i in self.tokenizer.word_index.items():
                try:
                    embedding_vector = word_vectors[word]
                    embedding_matrix[i] = embedding_vector
                except KeyError:
                    embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBED_SIZE)
            del(word_vectors)
            print( 'using Word2vec!')

        elif switch == 1:

            # GLOVE
            embeddings_dictionary = dict()
            EMBED_SIZE = 300
            glove_file = open('pretrained_embeddings/glove.6B.300d.txt', encoding="utf8")
            for line in glove_file:
                records = line.split()
                word = records[0]
                vector_dimensions = np.asarray(records[1:], dtype='float32')
                embeddings_dictionary[word] = vector_dimensions
            glove_file.close()

            embedding_matrix = np.zeros((self.vocab_size, EMBED_SIZE))
            for word, index in self.tokenizer.word_index.items():
                embedding_vector = embeddings_dictionary.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
            print( 'using Glove!')

        elif switch == 2:

            # spaCy
            EMBED_SIZE = len(self.nlp('The').vector)
            embedding_matrix = np.zeros((self.vocab_size, EMBED_SIZE))
            for word, index in self.tokenizer.word_index.items():
                embedding_matrix[index] = self.nlp(word).vector
            print( 'using spaCy!')

        else:
            raise ValueError('Invalid embeddings switch provided')

        return embedding_matrix, EMBED_SIZE
