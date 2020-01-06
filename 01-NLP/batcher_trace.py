# PARSE A SPECIAL-PURPOSE LOG FILE
# PRINT AND SAVE RESULTS

import os
import sys
import numpy as np
import pandas as pd
from time import strftime
from utils import get_params
import warnings
warnings.filterwarnings("ignore")

# find a group of consecutive sentences and return all
def find_text(log_local, idx_local):
    
    text_found = []
    for j in range(idx_local,len(log_local)):
        if 'kb.KBProcessor' in log_local[j]:
            idx1 = log_local[j].find('kb.KBProcessor')
            idx2 = log_local[j].find(']', idx1)
            sent = log_local[j][idx2+1:].strip()
            text_found.append(sent)            
        else:
            break            
            
    return ' '.join(text_found), j

# find a group of consecutive concepts and return only those that are 'true'
def find_concepts(log_local, idx_local):
    
    concepts_found = []
    for j in range(idx_local, len(log_local)):
        if 'kb.KBSearcher' in log_local[j] and 'true' in log_local[j]:
            idx1 = log_local[j].find('kb.KBSearcher')
            idx2 = log_local[j].find('-', idx1)
            sent = log_local[j][idx2+1:].strip()
            sent = sent.split(',')
            if len(sent) != 8:
                print('Different length,', len(sent), j)
            concepts_found.append(sent[0].strip().replace('"', ''))
        elif 'kb.KBSearcher' in log_local[j] and 'true' not in log_local[j]:
            continue
        else:
            break        
    return concepts_found, j

# find a group of consecutive categories and return only accepted ones
def find_cats(log_local, idx_local):
    
    cats_found = []
    for j in range(idx_local, len(log_local)):
        if 'kb.KBStatisticalLayer' in log_local[j] and 'Accepted category' in log_local[j]:
            idx1 = log_local[j].find('Accepted category')
            idx2 = log_local[j].find('-', idx1)
            idx3 = log_local[j].find("'", idx2)
            cat = log_local[j][idx2+1:idx3].strip()
            cats_found.append(cat)
        elif 'kb.KBStatisticalLayer' in log_local[j] and 'Accepted category' not in log_local[j]:
            continue
        else:
            break        
    return cats_found, j

# initialize variables
texts, concepts, cats, i = [], [], [], 0

# read parameters from settings file and read log file
params = get_params()

if params['wdir'] and params['wdir'] != 'None':
    wdir = params['wdir'].strip().replace('\\', '/')
    if wdir[-1] != '/':
        wdir = wdir + '/'
else:
    wdir = ''

if params['encode']:
    encode = params['encode']
else:
    encode = 'utf-8'

with open(wdir + params['log_file'], encoding=encode) as f:
    log = f.readlines()
    
# CREATE RESULTS FOLDER
cwd = os.getcwd()
if not os.path.exists(cwd + '/results'):
    os.makedirs('results')
else:
    print("Warning! The 'results' directory already exists\n")

# iterate over the log
while i < len(log):
        
    # find lines with text, but exclude the 'Accepted' line that doesn't contain useful information
    if 'kb.KBProcessor' in log[i] and 'm_id' not in log[i] and 'm_externalId' not in log[i] and 'm_weight' not in log[i] and 'm_attributes' not in log[i]:
        text, i = find_text(log, i)
        texts.append(text)
                
    # find lines with concepts
    if 'kb.KBSearcher' in log[i]:
        concept_list, i = find_concepts(log, i)
        
        # add empty string if there were no concepts during previous iteration
        if len(texts) - len(concepts) > 1:
            concepts.append('')
            
        # add empty string if there was no text during previous iteration
        if len(texts) == len(concepts):
            texts.append('')
                        
        concepts.append(concept_list)
                
    # find lines with categories
    if 'kb.KBStatisticalLayer' in log[i]:
        cat_list, i = find_cats(log, i)
        
        # add empty string if there were no categories during previous iteration
        if len(texts) - len(cats) > 1 or len(concepts) - len(cats) > 1:
            cats.append('')
                
        # add empty string if there was no text during previous iteration
        if len(texts) == len(cats):
            texts.append('')
            
        # add empty string if there were no concepts during previous iteration
        if len(concepts) == len(cats):
            concepts.append('')
                        
        cats.append(cat_list)        
    i += 1
    
# print and save
with open('results/traceResults_' + strftime('%Y%m%d%H%M%S') + '.txt', 'w', encoding='utf-8') as f:
    for item in list(zip(texts, concepts, cats)):
        print('TEXT:')
        print(item[0] + '\n')
        print('CONCEPTS FOUND:')
        print(', '.join(item[1]) + '\n')
        print('CATEGORIES ACCEPTED:')
        print(', '.join(item[2]) + '\n')
        print('*'*100 + '\n')            

        f.write('TEXT:\n')
        f.write(item[0] + '\n\n')
        f.write('CONCEPTS FOUND:\n')
        f.write(', '.join(item[1]) + '\n\n')
        f.write('CATEGORIES ACCEPTED:\n')
        f.write(', '.join(item[2]) + '\n\n')
        f.write('*'*100 + '\n\n')
    print('Results saved in file with timestamped name in results folder')