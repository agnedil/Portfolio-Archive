# UTILITY FUNCTIONS FOR BATCHERS IN THIS FOLDER

import re
import requests
import numpy as np
import pandas as pd
import unicodedata
from time import strftime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
try: 
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")

# F(X) TO POST A REQUEST
def post_request(payload, url, query, debug, headers):
    
    if query:
        response = requests.request("POST", url, data=payload.encode('utf-8'), headers=headers, params=query)
    else:
        response = requests.request("POST", url, data=payload.encode('utf-8'), headers=headers)
    #print(payload)
    if debug == '1':
        print(response)
        print('*'*50)
        print()
    if debug == '2':
        print(response.text)
        print('*'*100)
        print()
    return response    


# F(X) TO RETURN TAG(S)
def find_tag(html, tag):
       
    # parse text
    parsed_html = BeautifulSoup(html)
    tags_all    = parsed_html.find_all(tag)
    
    # find all instances of tag and its weight attribute, if any
    tags = []
    weights = []
    if tags_all:        
        for item in tags_all:
            tags.append(item.text.strip())
            if 'weight' in item.attrs:
                weights.append(item['weight'])

    # return results in special cases
    if not tags:
        return '', ''
    elif 'NO CATEGORIES' in tags:
        return 'NO CATEGORIES', ''    
    elif len(tags) == 1 and len(weights) == 1:
        return tags[0], weights[0]
    elif len(tags) == 1 and not weights:
        return tags[0], ''
        
    # return results
    return tags, weights

# TEXT CLEANING
def clean_text(content, remove_eols, remove_nls):
        
    # remove line breaks
    if remove_eols:
        content = content.replace('\r', ' ').replace('\n', ' ')
            
    # remove anything that is not a letter, white space, or select punctuation marks
    if remove_nls:
        pattern = re.compile(r'([^\sa-zA-Z ,.!?:;<>])+', re.UNICODE)
        content = pattern.sub('', content)
    
    # remove control characters
    content = ''.join(ch for ch in content if unicodedata.category(ch)[0]!="C")
    
    while '  ' in content:
        content = content.replace('  ', ' ')
            
    return content

# UNILABEL PARITAL ACCURACY COLUMN FOR EACH DOCUMENT NEEDED TO CALCULATE THE MEAN PARTIAL ACCURACY IN THE END
def part_accu(row):
    
    intersect = len([item for item in list(row['Predicted_Category']) if item in list(row['True_Category'])])
    union = len(set(list(row['Predicted_Category']) + list(row['True_Category'])))
    
    if union != 0:
        return round(intersect / union, 2)
    else:
        return 0
        
# UNILABEL PRECISION        
def precision(row):
    
    intersect = len([item for item in row['Predicted_Category'] if item in row['True_Category']])
    union     = len(row['Predicted_Category'])
    
    if union != 0:
        return round(intersect / union, 2)
    else:
        return 0

# UNILABEL RECALL        
def recall(row):
    
    intersect = len([item for item in row['Predicted_Category'] if item in row['True_Category']])
    union = len(row['True_Category'])
    
    if union != 0:
        return round(intersect / union, 2)
    else:
        return 0

# MULTILABEL PRECISION AND RECALL     
def multilabel_metrics(df):
        
    # df conversion step 1 (remove file_name index, 'true' column to the left)
    df = df.reset_index(drop=True)
    df = df[['True_Category', 'Predicted_Category']]
    df['True_Category'] = df['True_Category'].apply(lambda x: [x] if not isinstance(x, list) else x)
    df['Predicted_Category'] = df['Predicted_Category'].apply(lambda x: [x] if not isinstance(x, list) else x)

    # df conversion step 1 (disambiguate the 'true' column)
    values = df[['True_Category', 'Predicted_Category']].values.tolist()
    transform = []
    for value in values:
        if np.nan in value or (not isinstance(value, list) and pd.isnull(value)):
            continue
        for item in value[0]:
            transform.append([item, value[1]])
    df2 = pd.DataFrame(transform, columns=['True_Category', 'Predicted_Category'])
    df2['True_Category'] = df2['True_Category'].apply(lambda x: [x])

    # get categories
    cats = df2['True_Category'].tolist() + df2['Predicted_Category'].tolist()
    cats = set([item for sublist in cats for item in sublist if isinstance(sublist, list)])

    # run counts and calculate precision and recall for each category
    res = dict()
    for cat in cats:
        count_true, count_pred, count_correct = 0, 0, 0
        for item in df2['True_Category'].tolist():
            if cat in item:
                count_true += 1
        for item in df2['Predicted_Category'].tolist():
            if cat in item:
                count_pred += 1
        for item in list(zip(df2['True_Category'].tolist(), df2['Predicted_Category'].tolist())):
            if cat in item[0] and cat in item[1]:
                count_correct += 1
        if count_pred != 0 and count_true != 0:
            res[cat] = [count_correct / count_pred, count_correct / count_true]
        elif count_pred == 0 and count_true != 0:
            res[cat] = [0, count_correct / count_true]
        elif count_pred != 0 and count_true == 0:
            res[cat] = [count_correct / count_pred, 0]
        else:
            res[cat] = [0, 0]
    return res
    
# PRINT / SAVE UNILABEL CLASSIFICATION REPORT
def unilabel_report(y, y_pred, df):
        
    # prepare to calculate metrics
    labels = [item.strip() for item in sorted(list(set(y+y_pred)))]
    cm = confusion_matrix(y, y_pred, labels=labels)
    max_label = max([len(item) for item in labels])
    print_str = '{:>' + str(max_label) + '}  '
    for i in range(len(cm[0])):
        print_str += '{:>4} '
        
    # copy comfusion matrix, accuracy and classification report into a string variable
    metrics = '\nTotal accuracy: ' + str(round(accuracy_score(y, y_pred), 2)) + '\n\n'
    metrics += 'Confusion matrix:\n'
    for item in zip(labels, cm):
        metrics += print_str.strip().format(item[0], *item[1]) + '\n'    
    metrics += '\n' + classification_report(y, y_pred)
        
    # print results    
    print(metrics)
    
    # save results
    with open('results/metrics_'  + strftime('%Y%m%d%H%M%S') + '.txt', 'w', encoding='utf-8') as f:
        f.write(metrics)
    df[['True_Category', 'Predicted_Category',
        'Predicted_Weight']].to_csv('results/documents_' + strftime('%Y%m%d%H%M%S') + '.csv')
    print('Results saved to disk')
    
# PRINT / SAVE MULTILABEL CLASSIFICATION REPORT
def multilabel_report(df):
        
    # get metrics
    df['True_Category'] = df['True_Category'].apply(lambda x: [x] if not isinstance(x, list) else x)
    df['Predicted_Category'] = df['Predicted_Category'].apply(lambda x: [x] if not isinstance(x, list) else x)
    df['Partial_accuracy'] = df.apply(lambda x: part_accu(x), axis=1)
    df['Precision'] = df.apply(lambda x: precision(x), axis=1)
    df['Recall'] = df.apply(lambda x: recall(x), axis=1)    
    df.loc['Mean'] = df.mean()
    metrics = '\t\tOVERALL\n'
    overall_pr = [round(item, 5) for item in df.iloc[-1][['Partial_accuracy', 'Precision', 'Recall']].values.tolist()]
    metrics += '{:>50} {:>10} {:>10}\n'.format('Partial_accuracy', 'Precision', 'Recall')
    metrics += '{:>50} {:>10} {:>10}\n'.format(*overall_pr)
    metrics += '\n\t\tBY CATEGORY\n'
    metrics += '{:>50} {:>10} {:>10}\n'.format('Category', 'Precision', 'Recall')
    
    multilabel_pr = multilabel_metrics(df[['True_Category', 'Predicted_Category']][:-1].copy())
    for key, value in multilabel_pr.items():
        metrics += '{:>50} {:>10} {:>10}\n'.format(key, round(value[0], 5), round(value[1], 5))    
    
    # print results
    print(metrics)
    
    # save results
    with open('results/metrics_'  + strftime('%Y%m%d%H%M%S') + '.txt', 'w', encoding='utf-8') as f:
        f.write(metrics)    
    df[['True_Category', 'Predicted_Category', 'Predicted_Weight', 'Partial_accuracy',
        'Precision', 'Recall']].to_csv('results/documents_' + strftime('%Y%m%d%H%M%S') + '.csv')
    print('Results saved to disk')

# RETURN SPECIFIC SETTING FROM SETTINGS FILE        
def get_setting(setting, settings):
    
    if setting == 'req':
                
        idx1 = settings.find('begin')
        idx2 = settings.find('end')
        if idx1 == -1 or idx2 == -1:
            print('Incorrect settings file configuration')
            value = 'None'
        else:
            value = settings[idx1+5:idx2].strip()
            if not value:
                print('No such setting:', setting)
                value = 'None'
    
    else:
        idx1 = settings.find(setting)
        if idx1 == -1:
            print('No such setting:', setting)
            value = 'None'
        else:
            idx2 = settings.find('=', idx1)
            if setting in settings.split('\n')[-1]:
                idx3 = len(settings) + 1
            else:
                idx3 = settings.find('\n', idx2)
            value = settings[idx2+1:idx3].strip()
                        
    return value.strip()
        
# GET ALL SETTINGS FROM SETTINGS FILE    
def get_params():
    
    params_local = dict()
    my_keys = ['req', 'wdir', 'url', 'concenum', 'encode', 'log_file', 'tag_category', 'tag_text', 'tme_host_ip', 'tme_port',
               'username', 'password', 'debug', 'remove_eols', 'remove_nls']
    with open('settings.txt', encoding='utf-8') as f:
        file_text = f.read()
    for my_key in my_keys:
        params_local[my_key] = get_setting(my_key, file_text)
    return params_local

if __name__ == "__main__":
    pass