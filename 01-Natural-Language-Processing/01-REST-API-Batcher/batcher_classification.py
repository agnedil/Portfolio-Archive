# MULTICLASS DOCUMENT CLASSIFICATION
# SEND MULTIPLE REST API CALLS TO MAGELLAN TEXT MINING ENGINE; PARSE PROCESS XML RESPONSES
# COMPUTE SEVERAL METRICS SUCH AS PRECISION, RECALL, ACCURACY, CLASSIFICATION REPORT
# PRINT AND SAVE RESULTS FOR UNI-LABEL AND MULTI-LABEL CASES (ONE DOCUMENT BELONGS TO SEVERAL CLASSES)

import os
import numpy as np
import pandas as pd
try: 
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")
from utils import post_request,find_tag,clean_text,part_accu,precision,recall,multilabel_metrics,unilabel_report,multilabel_report,get_setting,get_params


# CREATE RESULTS FOLDER
cwd = os.getcwd()
if not os.path.exists(cwd + '/results'):
    os.makedirs('results')
else:
    print("Warning! The 'results' directory already exists\n")

# READ PARAMETERS FROM FILE
params = get_params()

if params['wdir'] and params['wdir'] != 'None':
    wdir = params['wdir'].replace('\\', '/')
else:
    wdir = cwd + '/input'
        
if not os.path.isdir(wdir):
    raise RuntimeError('Invalid directory for test set') from None

if not params['url'] or params['url'] == 'None':
    raise RuntimeError('Please provide a valid URL for REST API calls') from None
else:
    url = params['url']

if params['tag_category'] and params['tag_category'] != 'None':
    tag_cat = params['tag_category'].lower()
else:
    tag_cat = ''
if params['tag_text'] and params['tag_text'] != 'None':
    tag_text = params['tag_text'].lower()
else:
    tag_text = ''

if params['tme_port'] and params['tme_host_ip'] != 'None':
    querystring = {'username' : params['username'], 'password' : params['password'],
                   'tme_host_ip' : params['tme_host_ip'], 'tme_port' : params['tme_port']}
    headers = {
    'Content-Type': "application/xml",
    'Host': params['tme_host_ip']
    }
    
else:
    querystring = ''
    headers = {
    'Content-Type': "application/xml"
    }

debug = params['debug']
    
if params['remove_eols'].lower() in ['yes', 'y']:
    remove_eols = True
else:
    remove_eols = False
    
if params['remove_nls'].lower() in ['yes', 'y']:
    remove_nls = True
else:
    remove_nls = False
    
if params['encode']:
    encode = params['encode']
else:
    encode = 'utf-8'
    

# BEGINNING AND END OF EACH REQUEST; TEXT TO CLASSIFY GOES IN BETWEEN
part1 = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
         <Nserver>
         <ResultEncoding>UTF-8</ResultEncoding> 
         <TextID>inft</TextID>
         <NSTEIN_Text> <![CDATA["""

part2 = """]]> </NSTEIN_Text>

               <Methods>
                   <ncategorizer>\n""" + params['req'] + """                          
                   </ncategorizer>
              </Methods>
            </Nserver>"""

# GO OVER TEST SET FILES AND SEND CLASSIFICATION REQUESTS

# dict to store categories and weights
results = dict()
print('Processing ....')

# walk through root directory
for dirName, subdirList, fileList in os.walk(wdir):
        
    # iterate over subdirectories and filenames
    print('\tDirectory', dirName)
    print()
    for fname in fileList:
                
        with open(dirName + '/' + fname, 'r', encoding=encode) as f:
                        
            if debug == '1' or debug == '2':
                print(fname)

            text, true_cat, pred_cat, pred_weight = '', '', '', ''
            file_text = f.read()
            parsed_text = BeautifulSoup(file_text)                        
                        
            if tag_text:
                text = parsed_text.find(tag_text).text
            else:            
                if '<![CDATA[' in file_text:
                    idx1 = file_text.find('<![CDATA[')
                    idx2 = file_text.find(']]>')
                    text = file_text[idx1+10:idx2]
                elif parsed_text.find('ns2:nsteintext'):
                    text = parsed_text.find('ns2:nsteintext').text
                elif parsed_text.find('nsteintext'):
                    text = parsed_text.find('nsteintext').text
                elif parsed_text.find('text'):
                    text = parsed_text.find('text').text
                                    
            if tag_cat:
                true_cat, _ = find_tag(file_text, tag_cat)
            else:
                if parsed_text.find('ns2:name'):
                    true_cat, _ = find_tag(file_text, 'ns2:name')
                if parsed_text.find('category'):
                    true_cat, _ = find_tag(file_text, 'category')
                            
            full_text = part1 + text + part2
            response = post_request(full_text, url, querystring, debug, headers)
            if response.text:
                pred_cat, pred_weight = find_tag(response.text, 'category')
            results[fname] = [text, true_cat, pred_cat, pred_weight]

print('Done\n')

# CONVERT DICTIONARY TO PANDAS DATAFRAME
df_final = pd.DataFrame.from_dict(results, orient='index',
                                  columns=['Document', 'True_Category', 'Predicted_Category', 'Predicted_Weight'])

#DETERMINE THE WAY TO PRINT / SAVE THE RESULTS
y = df_final['True_Category'].values.tolist()
y_pred = df_final['Predicted_Category'].values.tolist()

if len(y) != len(y_pred):
    raise RuntimeError('Different number of true and predicted categories') from None

# MULTILABEL VS. UNILABEL    
multilabel = False
for item in y:
    if isinstance(item, list):
        multilabel = True
        break
for item in y_pred:
    if isinstance(item, list):
        multilabel = True
        break

if multilabel:
    multilabel_report(df_final.copy())
else:
    unilabel_report(y, y_pred, df_final.copy())