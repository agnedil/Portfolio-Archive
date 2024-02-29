# GET LIST OF CONCEPTS FOR MULTIPLE DOCUMENTS
# SEND MULPITLE REST API CALLS TO MAGELLAN TEXT MINING ENGINE; PARSE AND PROCESS XML RESPONSES
# PRINT AND SAVE RESULTS

import os
import re
import requests
import numpy as np
import pandas as pd
import unicodedata
from time import strftime
try: 
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")
from utils import post_request,clean_text,get_setting,get_params


# F(X) TO RETURN CONCEPTS
def find_concepts(html, tag):
       
    parsed_html = BeautifulSoup(html)
    tags_all    = parsed_html.find_all(tag)

    # find all instances of concept and its frequency attribute, if any
    concepts = dict()    
    if tags_all:        
        for item in tags_all:
            freq = 0
            if 'frequency' in item.attrs:
                freq = int(item['frequency'])
                                
            if item.text in concepts:
                concepts[item.text] += freq
            else:
                concepts[item.text] = freq                                
            
    return concepts

    
# CREATE RESULTS FOLDER
cwd = os.getcwd()
if not os.path.exists(cwd + '/results'):
    os.makedirs('results')
else:
    print("Warning! The 'results' directory already exists and may contain previous results\n")

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

if params['tme_host_ip'] and params['tme_host_ip'] != 'None':
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

if params['debug']:
    debug = params['debug']
else:
    debug = ''

if params['concenum'] and params['concenum'] != 0:
    cnum = abs(int(params['concenum']))
else:
    cnum = 100

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
               <LanguageID>ENGLISH</LanguageID>
               <Methods>
                              
                   <nconceptextractor>                       
                       <ResultLayout>NCONCEPTEXTRACTOR</ResultLayout>
                       <ExcludeEntities/>
                                              
                       <SimpleConcepts>
                           <NumberOfSimpleConcepts>1000</NumberOfSimpleConcepts>
                       </SimpleConcepts>
                                              
                       <ComplexConcepts>
                           <NumberOfComplexConcepts>1000</NumberOfComplexConcepts>
                           <RelevancyLevel>FIRST</RelevancyLevel>
                       </ComplexConcepts>                                              
                   </nconceptextractor>
                                                           
                </Methods>                                
            </Nserver>"""

# GO OVER TEST SET FILES AND SEND CONCEPT EXTRACTION REQUESTS

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
                        
            # READ FILE CONTENT
            print(fname)
            file_text = f.read()
            parsed_text = BeautifulSoup(file_text)                      
                        
            # GET TEXT
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
            else:
                text = file_text
                                    
            # SEND REQUEST    
            full_text = part1 + text + part2
            my_response = post_request(full_text, url, querystring, debug, headers).text

            # FIND CONCEPTS
            concepts_all = find_concepts(my_response, 'concept')
            if concepts_all:
                for key, value in concepts_all.items():
                    if key in results:
                        results[key] += value
                    else:
                        results[key] = value

print('Done\n')

# CONVERT DICTIONARY TO PANDAS DATAFRAME AND SORT
df = pd.DataFrame.from_dict(results, orient='index', columns=['Count'])
df = df.sort_values(by='Count', ascending=False)

# SAVE RESULTS
if cnum > len(df):
    cnum = len(df)
df[:cnum].to_csv('results/' + str(cnum) + '_concepts_' + strftime('%Y%m%d%H%M%S') + '.csv')
print(df[:cnum].to_string())