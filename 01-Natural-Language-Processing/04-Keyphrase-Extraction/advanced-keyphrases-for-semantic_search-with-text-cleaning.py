import os
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
import re
import time
import ftfy
import multiprocessing
from unidecode import unidecode
from sklearn.metrics.pairwise import cosine_similarity
import hs_classifier_text_processing as tp
from space_corrector import space_corrector
from typing import Optional


def get_template() -> dict:
    '''
       Data structure to return one similarity found between a phrase from shipment and
       a phrase from the knowledge base
    '''
    return { 
             'score': 0,
             'phrase': '',
             'kb_phrase': '',
             'subindustry': '',
             'method': '',
           }




########################################################################################################################
## CLASS TO CLEAN TEXT ##

sw = { 'none', 'n/a' }

class Cleaner:
    '''
        
    HEAVY TEXT CLEANING USED FOR KEYWORD EXTRACTION
        
    Additional features, if needed:
        
    months_list  = { 'january', 'jan', 'february', 'feb', 'march', 'mar', 'april', 'apr', 'may', 'june', 'jun', 'july', 'jul',
                       'august', 'aug', 'september', 'sep', 'october', 'oct', 'november', 'nov', 'december', 'dec', }
    months_list.update({m.capitalize() for m in months_list})
    months       = '|'.join( months_list )
    month        = re.compile(f'(?<=\s)({months})(?=\s)')    
    year         = re.compile('(?<=\s)20\d{2}(?=\s)')
    ordinal      = re.compile('(?<=\s)(1\s*st|2\s*nd|3\s*rd|[1-9][0-9]*\s*th)(?=\s)')
    date         = re.compile('\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{4}|\d{4}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{1,2}')
    phone        = re.compile('\(\s*\d{3}\s*\)\s*\d{3}\s*[-]\s*\d{4}')
    best_percent = re.compile('(?<=\s)100\s*%')                      # pos. look-behind: if preceeded by space
    high_percent = re.compile('(?<=\s)[8,9][0-9]\s*%')
    percent      = re.compile('(?<=\s)[1-7][0-9]\s*%')
    dedupe       = re.compile(r'\b(\w+)(\s*\1\b)+')                  # remove duplicates (e.g., numeric numeric)
    
    s = self.ordinal.sub(r'ordinal', s)
    s = self.date.sub(r'date', s)
    s = self.phone.sub(r'phone', s)
    s = self.best_percent.sub(r'best percentage', s)
    s = self.high_percent.sub(r'high percentage', s)
    s = self.percent.sub(r'percentage', s)
    s = self.month.sub(r'month', s)
    s = self.year.sub(r'year', s)
    s = self.nums.sub(r'numeric', s)
    s = self.dedupe.sub(r'\1', s)
    
    COMMENTS:
    Not padding hyphen to preserve compound words for semantic search
    Padding punctuation and parenthesis => cases of 'KGS.' or 'CREATE(S)' can be removed from list of stopwords
    
    '''
    # char cleanup
    punct        = ",.@$/\-()'\s"                                # keep these initially (for NER)
    punct2       = ",.@$()'\s"                                   # pad these (excl. - and /) - for clean tokens
    punct3       = ",.'\-\s"                                     # keep only these, finally
    clean_text   = re.compile(f'[^a-zA-Z0-9{punct}]+')
    pad_punct    = re.compile(f'([{punct2}])')                   # pad select puncts
    clean_text2  = re.compile(f'[^a-zA-Z0-9{punct3}]+')  
        
    # NER
    domains      = 'com|edu|gov|biz|net|org|us|info|mil|co|uk|de|ru|es|fr|cn|me|tk|icu|top|xyz|nl|ga|cf'
    website      = re.compile(f'(https?)?([/:\s]+)?(www)\s*\.\s*\w+\s*\.\s*({domains})')
    email        = re.compile(f'[a-zA-Z0-9_]+\s*@\s*[a-zA-Z0-9_]+\s*\.\s*({domains})')
    amounts      = re.compile('([$]\s*[0-9]+(\s*[,.]\s*[0-9]+)?)')
        
    # numbers and apostrophees
    nums         = re.compile('([0-9]+(\s*[,.]\s*[0-9]+)?)')          # sequences of nums with decimal sep
    pad_apost_s  = re.compile('([\']s)')                              # pad "'s" on the left
    repl_apost   = re.compile('([\'](?!s))')                          # replace single apostrophees w/space    
    multi_spaces = re.compile('\s{2,}')                               # replace multiple spaces w/one

    
    def __init__( self,
                  stopwords      = sw,
                ): 
        self.stopwords = set(stopwords)

            
    def batch_clean(self, texts_, to_lower=True):
        '''
            Advantage over list comprehension is not clear
        '''
        for t in texts_:
            yield self.clean(t, to_lower=to_lower)


    def repair_text(self, s):
        '''
            Clean up encoding, HTML leftovers, and other issues;
            full list of parameters included to enable flexibility when needed.
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


    def clean_encoding( self, s, ):
        '''
            This functions fixes char encoding
        '''
        if not isinstance(s, str) or not s:
            return ''
        
        for char in ['�', '•']:                     # TODO: add more chars that evade repair_text()
            s = s.replace(char, '')

        return self.repair_text(s).strip()


    def clean(self, s, to_lower=True ):
        '''
            Remove stopwords, mask entities, remove special chars
            This is heavy text cleaning - used for keyword extraction
        '''
        s = self.clean_encoding(s)
        s = self.clean_text.sub(' ', s)
        s = self.pad_apost_s.sub(r' \1', s)
        s = self.repl_apost.sub(' ', s)
        s = self.pad_punct.sub(r' \1 ', s)
        s = self.nums.sub(r' \1 ', s)
        s = self.multi_spaces.sub(' ', s)
        
        s = self.website.sub(r'url', s)
        s = self.email.sub(r'email', s)
        s = self.amounts.sub(r'amount', s)

        if to_lower:
            s = s.strip().lower().split()
            s = [i for i in s if not i in self.stopwords]
        else:
            s = s.strip().split()
            s = [i for i in s if not i.lower() in self.stopwords]

        s = [w[1:] if w.startswith('-') else w for w in s]
        s = [w[:-1] if w.endswith('-') else w for w in s]
        
        consonants = 'bcdfghjklmnpqrstvwxz-/'
        s = [w for w in s if not all(c in consonants for c in w)]

        s = self.clean_text2.sub(' ', ' '.join(s))
        s = self.multi_spaces.sub(' ', s)

        return s.strip()

    
    
class LightCleaner:
    '''
        General purpose light cleaner.
        Difference from Cleaner: no entity masking and different sets of punct (unify?)
        Keep only letters and numbers, pad groups of numbers, remove double spaces and stopwords, 
    '''
    
    punct        = "\.\-\'\\/,;:@$%()"                                # to keep
    punct2       = "\.\-\\/,;:@$%()"                                  # to pad (excl. ')
    clean_text   = re.compile(f'[^a-zA-Z0-9{punct}]+')
    clean_text2  = re.compile('[^a-zA-Z0-9]+')
    pad_apost_s  = re.compile('([\']s)')                              # pad "'s" on the left
    repl_apost   = re.compile('([\'](?!s))')                          # replace single apostrophees w/space
    pad_punct    = re.compile(f'([{punct2}])')                        # pad select puncts
    nums         = re.compile('([0-9]+(\s*[,.]\s*[0-9]+)?)')          # sequences of nums with decimal sep    
    multi_spaces = re.compile('\s{2,}')                               # replace multiple spaces w/one

    def __init__( self,
                  stopwords = sw,
                ): 
        self.stopwords = set(stopwords)

            
    def batch_clean(self, texts_, to_lower=True):
        '''
            Advantage over list comprehension is not clear
        '''
        for t in texts_:
            yield self.clean(t, to_lower=to_lower)


    def repair_text(self, s):
        '''
            Clean up encoding, HTML leftovers, and other issues;
            full list of parameters included to enable flexibility when needed.
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


    def clean_encoding( self, s, ):
        '''
            This functions fixes char encoding
        '''
        if not isinstance(s, str) or not s:
            return ''
        
        for char in ['�', '•']:                     # TODO: add more chars that evade repair_text()
            s = s.replace(char, '')

        return self.repair_text(s).strip()


    def clean(self, s, to_lower=True, punct=False ):
        '''
            Remove stopwords, remove special chars
            punct=True keeps and pads basic punctuation
            punct=False proved more useful for Full Text Search
        '''
        s = self.clean_encoding(s)
        
        # punct=True keeps and pads basic punctuation. Close to Cleaner() for KP extraction (but not same)
        if punct:            
            s = self.clean_text.sub(' ', s)
            s = self.pad_apost_s.sub( r' \1', s )
            s = self.repl_apost.sub( ' ', s )
            s = self.pad_punct.sub(r' \1 ', s)        
            s = self.nums.sub(r' \1 ', s)        
            s = self.multi_spaces.sub(' ', s)                        
        # punct=False proved more useful for Full Text Search
        # why removing punct is better? E.g. 'precious metal' vs. 'precious - metal', punct is in the way
        else:
            s = self.clean_text2.sub(' ', s) 
            s = self.nums.sub(r' \1 ', s)        
            s = self.multi_spaces.sub(' ', s)

        # have been using to_lower() all the time
        if to_lower:
            s = s.strip().lower().split()
            s = [i for i in s if not i in self.stopwords]
        else:
            s = s.strip().split()
            s = [i for i in s if not i.lower() in self.stopwords]        

        return ' '.join(s).strip()
        
    
########################################################################################################################
## CLASS TO PARSE NOUN PHRASES ##


class Keyphrase_Extractor:
    
    def __init__(self):
        
        self.cleaner = Cleaner()

        # load data-specific stopwords
        with open('20220316_stopwords_semantic_search.txt', encoding='utf-8') as f:
            sw = [w.strip() for w in f.readlines()]
        print('Data-specific stopwords:', len(sw))


        # modify spaCy stopwords
        spacy_sw_to_remove = [ '‘d', '‘ll', '‘m', '‘re', '‘s', '‘ve', '’d', '’ll', '’m', '’re', '’s', '’ve',
                               "'d", "'ll", "'m", "'re", "'s", "'ve", 'about', 'above', 'across', 'after', 'fine',
                               'am', 'amount', 'and', 'at', 'bottom', 'top', 'by', 'down', 'up', 'empty', 'for',
                               'from','in', 'into', 'not', 'of', 'on', 'onto', 'upon', 'or', 'thru', 'to',
                               'until', 'via', 'with', 'well', 'over', 'within', 'without', 'us', 'under',
                               'front', 'back', 'full', 'side',
                             ]

        self.nlp = spacy.load("en_core_web_lg", exclude=['parser', 'ner'])
        print('Spacy model loaded; using pipes:', self.nlp.pipe_names)
        print('\nOOB stopwords:     ', len(self.nlp.Defaults.stop_words) )
        self.nlp.Defaults.stop_words -= set(spacy_sw_to_remove)
        self.nlp.Defaults.stop_words |= set(sw)
        print('Modified stopwords:', len(self.nlp.Defaults.stop_words) )

        # remove non-useful single "noun phrases" from the final list
        self.single_stopwords = [ 'and', 'for', 'in', 'of', 'or', 'top', 'into', 'on', 'onto', 'to', 'via', 'parts', 
                                  'and or', 'saidto', 'ofthe', 'style', 'software', 'material', 'product', 'ocean',
                                  'marks', 'heat', 'ofthe', 'accordingto', 'partof', 'lengthwis', 'ina', 'ofthis',
                                  'shouldbe', 'mustbe', 'inthe', 'ofamerica', 'doorto', 
                                ]

        self.single_stopwords = set(self.single_stopwords + spacy_sw_to_remove)


        # define spaCy matcher rules
        pattern1 = [             
                     {'POS': 'ADV',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                     {'POS': 'ADJ',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                     {'TAG': 'VBG',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                     {'TAG': 'VBN',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                     {'TAG': 'VBD',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                     {'POS': 'NOUN',  'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '?'},
                     {'POS': 'PROPN', 'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '?'},
                     {'POS': 'NOUN',  'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '+'},
                    ]

        pattern2 = [
                     {'POS': 'ADV',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                     {'POS': 'ADJ',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                     {'TAG': 'VBG',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                     {'TAG': 'VBN',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                     {'TAG': 'VBD',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                     {'POS': 'NOUN',  'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '?'},
                     {'POS': 'PROPN', 'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '?'},
                     {'POS': 'PROPN', 'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '+'},
                    ]

        pattern3 = [
                     {'POS': 'ADV',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                     {'POS': 'ADJ',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                     {'TAG': 'VBG',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                     {'TAG': 'VBN',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                     {'TAG': 'VBD',   'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '*'},
                    ]

        pattern4 = pattern3 +\
                   [
                     {'POS': 'NOUN',  'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '+'},             
                     {'TAG': 'HYPH',  'OP': '+'},             
                     {'POS': 'NOUN',  'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '+'},
                   ]

        pattern5 = pattern3 +\
                   [             
                     {'POS': 'PROPN', 'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '+'},
                     {'TAG': 'HYPH',  'OP': '+'},
                     {'POS': 'PROPN', 'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '+'},             
                   ]

        pattern6 = pattern3 +\
                   [
                     {'POS': 'PROPN', 'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '+'},             
                     {'TAG': 'HYPH',  'OP': '+'},             
                     {'POS': 'NOUN',  'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '+'},
                   ]

        pattern7 = pattern3 +\
                   [             
                     {'POS': 'NOUN',  'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '+'},
                     {'TAG': 'HYPH',  'OP': '+'},
                     {'POS': 'PROPN', 'IS_ALPHA': True, 'LENGTH': {'>': 1}, 'IS_STOP': False, 'OP': '+'},             
                   ]
                

        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add('noun phrases', [ pattern1, pattern2, pattern4, pattern5, pattern6, pattern7, ])        
        

    def split_long_nps( self, s: str ) -> list[str]:
        '''
            Split long houn phrase into shorter noun phrases
            in each of which there are no more than 2 nouns    
        '''
        idxs = []
        count_nouns = 0

        d = self.nlp(s)
        for t in d:
            if t.pos_ == 'NOUN' or t.pos_ == 'PROPN':
                count_nouns += 1
            if count_nouns == 2:
                if t.i < (len(d) - 1):
                    idxs.append(d[t.i+1].idx)
                    count_nouns = 0
        idxs = [0] + idxs + [None]
        return [ s[a:b].strip() for a,b in zip(idxs[:-1], idxs[1:]) ]


    def match_nounphrases( self, doc_ ):
        '''
        Get noun phrases based on POS patterns
            Arguments:
                text_      - text as one string to have noun phrases extracted from
            Returns:
                keyphrases - list of all noun phrases
        '''
        # find matches, remove overlaps    
        matches = self.matcher(doc_)
        if not matches:
            return matches
        spans   = [doc_[start:end] for _, start, end in matches]
        spans   = filter_spans(spans)                                     # remove overlaps
        spans   = [ span.text.strip() for span in spans ]                 # keep text only
        spans   = [s[1:] if s.startswith('-') else s for s in spans]     # sometimes, '-' is first char

        short_spans = []                                                  # split spans with > 2 nouns
        for span in spans:
            if len(span.split()) >= 5:
                short_spans.extend( self.split_long_nps(span) )
            else:
                short_spans.append( span )

        stack = [ short_spans[0] ]                                        # remove duplicates, but keep order    
        for span in short_spans[1:]:
            if span not in stack:
                stack.append(span)        
        stack = [ w for w in stack if w not in self.single_stopwords ]

        return stack


    def fetch_nounphrases(self, descriptions: list[str]) -> list[list[str]]:
        '''
        Extract useful noun phrases describing the goods shipped

        :param descriptions: iterable of shipment descriptions
        :returns: list consisting of lists of noun phrases, one per each shipment description
        :raises ValuError: if wrong data type is passed
        '''        
        if not isinstance(descriptions, list) and not isinstance(descriptions, np.ndarray):
            raise ValueError(f'Expected iterable; got {type(descriptions)}')

        if len(descriptions) == 0:
            return descriptions

        if not isinstance(descriptions[0], str):
            raise ValueError(f'Expected iterable of strings; got iterable of {type(descriptions[0])}')

        n_process    = int(multiprocessing.cpu_count() / 2)
        print(f'Attention: using {n_process} processors')
        descriptions = [ tp.space_corrector( i.lower(),
                                             add_spaces=False,
                                             aggressive_join=True, ) for i in descriptions ]
        descriptions = list( self.cleaner.preprocess(descriptions) )    
        docs         = list( self.nlp.pipe(descriptions, n_process=n_process) )
        res          = [ self.match_nounphrases(doc) for doc in docs ]

        return res
    
    
    

########################################################################################################################
## CLASS FOR SEMANTIC SEARCH WITH WORD EMBEDDINGS ##


class Semantic_Search:
        
    
    def __init__(self, kb_list: list[str], sub_ind: str) -> None:
        '''
            :param kb_list: list of manually pre-defined keyphrases for a specific topic (sub-industry),
            :param sub_ind: identifier for a specific sub-industry for kb_list
        '''
        self.nlp        = spacy.load("en_core_web_lg", exclude=['parser', 'ner'])
        print('Spacy model loaded; using pipes:', self.nlp.pipe_names)

        self.sub_ind    = sub_ind
        self.kb_list    = kb_list
        self.kb_vectors = np.array([ self.nlp(i).vector for i in self.kb_list ])
                
        
    def one_sim(self, kp: str) -> Optional[dict]:
        '''
        Compare one keyphrase from shipment description with all knowledge base phrases
        :param kp: one keyphrase from shipment description

        '''
        if not isinstance(kp, str):
            raise ValueError(f'Expected type str for kp; got {type(kp)}')

        use_lemmas = False    

        doc = self.nlp(kp)
        if use_lemmas:
            kp = ' '.join([ t.lemma_ for t in doc ])
            doc = self.nlp(kp)
        kp_vector = doc.vector

        sims      = cosine_similarity( kp_vector.reshape(1, -1),
                                       self.kb_vectors,
                                     ).reshape(len(self.kb_vectors))        
        idx_max = np.argmax(sims)

        if sims[idx_max] >= 0.75:
            res = get_template()
            res['score'] = round(sims[idx_max], 5)
            res['phrase'] = kp
            res['kb_phrase'] = self.kb_list[idx_max]
            res['subindustry'] = self.sub_ind
            res['method'] = 'semantic'
            return res
        else:
            return None


    def all_sims(self, kps: list[str]) -> list[ Optional[dict] ]:
        '''
        :params kps: list of keyphrases from shipment description    
        '''
        res_all = [ self.one_sim(i) for i in kps ]
        return [ i for i in res_all if i is not None ]
    
    
    
    
########################################################################################################################
## CLASS FOR FULL TEXT SEARCH ##


class Full_Text_Search:
    
    def __init__(self, fts_list: list[str], sub_ind: str) -> None:
        '''
            :param fts_list: list of manually pre-defined keyphrases for a specific topic (sub-industry),
            :param sub_ind: identifier for a specific sub-industry for fts_list
        '''
        self.sub_ind  = sub_ind
        self.fts_list = [i.lower() for i in fts_list]
        self.cleaner  = LightCleaner()
        
        
    def one_full_text_search(self, kp: str) -> Optional[dict]:
        '''
        Compare one keyphrase from shipment description with all knowledge base phrases
        :param kp: one keyphrase from shipment description

        '''
        results = []
        kp = kp.lower()
        for i in self.fts_list:
            if i in kp:
                res = get_template()
                res['score'] = 1.0
                res['phrase'] = kp
                res['kb_phrase'] = i
                res['subindustry'] = self.sub_ind
                res['method'] = 'full-text'
                results.append(res)

        return results


    def all_full_text_searches(self, kps: list[str]) -> list[ Optional[dict] ]:
        '''
        :param kps: list of keyphrases from shipment description    
        '''
        if not isinstance(kps, list):
            return kps
        elif len(kps) == 0:
            return np.nan

        res_all = []
        for i in kps:
            res_all.extend( self.one_full_text_search(i) )
        return res_all
        
    
    def full_text_search_in_text(self, text_: str) -> list[ Optional[dict] ]:
        
        results = []
        text_   = self.cleaner.clean(text_, to_lower=True, punct=False)
        text_   = space_corrector2(text_)
        for i in self.fts_list:
            if i in text_:
                res = get_template()
                res['score'] = 1.0
                res['phrase'] = i
                res['kb_phrase'] = i
                res['subindustry'] = self.sub_ind
                res['method'] = 'full-text'
                results.append(res)

        return results
    
                           

if __name__ == '__main__':
    '''
        Using keyphrase extractor in jupyter notebook with dataframes:
            from semantic_search import Keyphrase_Extractor
            extractor = Keyphrase_Extractor()
            df['np'] = extractor.fetch_nounphrases( df['shipment'].values )
            
            
        Using semantic search in jupyter notebook with dataframe:
            from semantic_search import Semantic_Search
            sub_ind    = '25201040'
            kb_list    = list( knowledgebase[ sub_ind ]['semantic'] )
            search     = Semantic_Search(kb_list, sub_ind)
            df['sims'] = df['np'].apply(lambda x: search.all_sims(x) if isinstance(x, list) else x)            
                  
                 
        Using full-text search in jupyter notebook with dataframe:
            from semantic_search import Full_Text_Search
            sub_ind    = '25201040'
            fts_list   = list( knowledgebase[ sub_ind ]['full_text'] )            
            fts_search = Full_Text_Search( fts_list, sub_ind )
                        
            df['fts']  = df['np'].apply( lambda x: fts_search.all_full_text_searches(x) if isinstance(x, list) else x)
            OR (works better)
            df['fts']  = df['shipment'].apply( lambda x: fts_search.full_text_search_in_text(x) if isinstance(x, str) else x)
            
            where:
                  knowledgebase - predefined knowledge base with key phrases for each sub-industry
                  '25201040'    - identifier for one sub-industry
                  df['np']      - contains a list of key phrases extracted for each shipment
    '''
    
        
    print('Keyphrase extractor sample usage:')
    extractor    = Keyphrase_Extractor()
    descriptions = [ '5 containers of household appliances with electric steam mops',
                     'Apple laptops, canned fruit, and tomato juice', ]     
    result       = extractor.fetch_nounphrases( descriptions )
    print(result)
    '''
    EXPECTED OUTPUT
    [['household appliances', 'electric steam mops'], ['apple laptops', 'canned fruit', 'tomato juice']]
    '''
        
    
    print('\nSemantic seach sample usage:')
    sub_ind = 'appliances'
    test_kb = { 'appliances': {'semantic': {'appliaces', 'household appliances'} } }
    kb_list = list( test_kb[ sub_ind ]['semantic'] )
    
    search     = Semantic_Search(kb_list, sub_ind)
    keyphrases = [ 'electric appliances', 'home appliances' ]
    result     = search.all_sims(keyphrases)
    print(result)    
    '''
    EXPECTED OUTPUT
    [{'score': 0.797, 'phrase': 'electric appliances', 'kb_phrase': 'household appliances', 'subindustry': 'appliances', 'method': 'semantic'},\
    {'score': 0.87, 'phrase': 'home appliances', 'kb_phrase': 'household appliances', 'subindustry': 'appliances', 'method': 'semantic'}]
    '''
    
    
    print('\nFull text seach sample usage:')
    sub_ind = 'appliances'
    test_kb = { 'appliances': {'full_text': { 'electric appliances', 'home appliances' } } }
    fts_list = list( test_kb[ sub_ind ]['full_text'] )
    
    search     = Full_Text_Search(fts_list, sub_ind)
    keyphrases = [ 'electric appliances in bulk', 'a vareity of home appliances' ]
    result     = search.all_full_text_searches(keyphrases)
    print(result)    
    '''
    EXPECTED OUTPUT
    [{'score': 1.0, 'phrase': 'electric appliances in bulk', 'kb_phrase': 'electric appliances', 'subindustry': 'appliances', 'method': 'full-text'},\
    {'score': 1.0, 'phrase': 'a vareity of home appliances', 'kb_phrase': 'home appliances', 'subindustry': 'appliances', 'method': 'full-text'}]
    '''        
    
