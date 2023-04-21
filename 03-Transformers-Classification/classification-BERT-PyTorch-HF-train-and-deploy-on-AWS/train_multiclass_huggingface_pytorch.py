import numpy as np
import pandas as pd
import os
import itertools
import random
import boto3
import shutil
import logging
from datetime import datetime
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


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


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return ( round( f1_score(labels_flat, preds_flat, average='micro'), 4 ),
             round( f1_score(labels_flat, preds_flat, average='macro'), 4 ),
           )


def accuracy_per_class(preds, labels):

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label}')
        print('Accuracy:', len(y_preds[y_preds==label])/len(y_true), '\n')


def get_dataloader( X, y, tokenizer, batch_size, maxlen ):

    # `batch_encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_data = tokenizer.batch_encode_plus( X,
                                                add_special_tokens    = True,
                                                return_attention_mask = True,
                                                pad_to_max_length     = True,
                                                max_length            = maxlen,
                                                return_tensors        = 'pt',
                                              )

    input_ids       = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels          = torch.tensor( y )

    dataset         = TensorDataset( input_ids, attention_masks, labels )
    dataloader      = DataLoader(    dataset,
                                     sampler    = RandomSampler( dataset ),
                                     batch_size = batch_size,
                                 )
    return dataloader


def evaluate( dataloader ):

    # put model in eval mode
    model.eval()

    loss_val_total = 0
    preds, true_vals = [], []

    for batch in dataloader:

        # add batch to device (GPU)
        batch = tuple(b.to(device) for b in batch)

        # unpack inputs from dataloader
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        # tell the model not to compute gradients => save memory, speed up prediction
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        loss_val_total += loss.item()
        logits = outputs[1]

        # move logits, labels to CPU (logits = raw classifier output)
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()

        preds.append( logits )
        true_vals.append( label_ids )

    loss_val_avg = loss_val_total/len(dataloader)

    preds = np.concatenate(preds, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, preds, true_vals


seed = random.randint(0,500)
random.seed( seed )
np.random.seed( seed )
torch.manual_seed( seed )
torch.cuda.manual_seed_all( seed )
print('Seed:', seed)


logging.info('Downloading preprocessed data....')
s3 = boto3.client('s3')
BUCKET_NAME = ''
OBJECT_NAME = 'preprocessed_pytorch_bert.pkl'
FILE_NAME   = '20220122_preprocessed_pytorch_bert.pkl'
print(f'Downloading file "{FILE_NAME}" from directory "{OBJECT_NAME}" in bucket "{BUCKET_NAME}"\n')
with open(FILE_NAME, 'wb') as f:
    s3.download_fileobj(BUCKET_NAME, OBJECT_NAME, f)

df = pd.read_pickle( FILE_NAME )
print('Data size:', df.shape)
maxlen = 125
print('Maxlen:   ', maxlen)

X_train = df[ df['subset'] == 'train' ]['sentence'].values
y_train = df[ df['subset'] == 'train' ]['target'].values
X_val   = df[ df['subset'] == 'val' ]['sentence'].values
y_val   = df[ df['subset'] == 'val' ]['target'].values
X_test  = df[ df['subset'] == 'test' ]['sentence'].values
y_test  = df[ df['subset'] == 'test' ]['target'].values

print( X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape, )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

epochs         = 5
learning_rates = [5e-5]#[ 5e-5, 5e-4, 5e-3, 5e-2 ]
batch_sizes    = [16]#[ 32, 16, 8, ]

all_combinations = list(itertools.product( *[learning_rates, batch_sizes] ))

wdir = 'ckpts'
if not os.path.exists(wdir):
    print(f"Creating the {wdir} folder")
    os.makedirs(wdir)

time_stamp1 = time.strftime("%Y%m%dT%H%M")
file_name   = f'{wdir}/log_{time_stamp1}.txt'

print('Seed:', seed)
with open( file_name, 'w', encoding='utf-8' ) as f:
    experiment_name = 'BERT PYTORCH\n'
    f.write( experiment_name )
    for LR, batch_size in all_combinations:

        time_stamp = time.strftime("%Y%m%dT%H%M")
        params = f'\nLR={LR}, batch_size={batch_size}'
        print( params )
        print( 'Timestamp:', time_stamp)
        f.write( params + '\nTimestamp: ' + time_stamp + '\n' )

        tokenizer  = BertTokenizer.from_pretrained( 'bert-base-uncased',
                                                    do_lower_case=True,
                                                  )
        dataloader_train = get_dataloader( X_train, y_train, tokenizer, batch_size, maxlen )
        dataloader_val   = get_dataloader( X_val, y_val, tokenizer, batch_size, maxlen )
        dataloader_test  = get_dataloader( X_test, y_test, tokenizer, batch_size, maxlen )

        model = BertForSequenceClassification.from_pretrained( "bert-base-uncased",
                                                               num_labels=15,
                                                               output_attentions=False,
                                                               output_hidden_states=False,
                                                               attention_probs_dropout_prob=0,
                                                               hidden_dropout_prob=0,
                                                             )
        optimizer = AdamW( model.parameters(),
                           lr=LR,                 # 1e-5
                           eps=1e-8,              # very small number to avoid division by 0
                         )

        scheduler = get_linear_schedule_with_warmup( optimizer,
                                                     num_warmup_steps=5,
                                                     num_training_steps=len(dataloader_train)*epochs,
                                                   )
        model.to(device)
        for epoch in range(1, epochs+1):

            model.train()
            loss_train_total = 0

            for batch in dataloader_train:

                model.zero_grad()
                batch = tuple(b.to(device) for b in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[2],
                         }

                outputs = model(**inputs)
                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

            print( f'\nEpoch {epoch}' )
            f.write( f'\nEpoch {epoch}' + '\n' )

            loss_train_avg = round( loss_train_total/len(dataloader_train), 4 )
            val_loss, preds, y_val = evaluate( dataloader_val )
            val_loss = round( val_loss, 4 )
            val_f1 = f1_score_func( preds, y_val )

            metrics = f'Training loss: {loss_train_avg}\n' + f'Validation loss: {val_loss}\n' +\
                      f'F1 Score (micro): {val_f1[0]}\n' + f'F1 Score (macro): {val_f1[1]}\n'

            print( metrics )
            f.write( metrics + '\n')

            filepath = wdir + '/' + time_stamp + f'-epoch_{epoch}-val_loss_{val_loss}-f1micro_{val_f1[0]}-f1macro{val_f1[1]}.model'
            torch.save(model.state_dict(), filepath )

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
