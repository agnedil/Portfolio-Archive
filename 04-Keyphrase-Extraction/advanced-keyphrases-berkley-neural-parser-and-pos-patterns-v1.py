import spacy, benepar
from spacy.matcher import Matcher
from spacy.util import filter_spans
from nltk.tree import Tree
from nltk.draw.tree import draw_trees
import colored                                                    # more here: https://gitlab.com/dslackw/colored
import re


def is_notebook():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


class Span_Detector:


    def __init__(self):

        # benepar works only at the Span._ and Token._ level
        self.nlp = spacy.load('en_core_web_lg', exclude=['ner'])
        self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        self.doc        = None
        self.confidence = None


    def build_matcher(self):
        '''
            Build Spacy's matcher based on select POS patterns
        '''

        # VERB PHRASES

        plus = [
                      {'TAG': 'RP',   'OP': '*'},                        # adverb particle (come ON)
                      {'TAG': 'IN',   'OP': '*'},                        # preposition or subordinating conjunction
                      {'TAG': 'PRP',  'OP': '*'},
                      {'TAG': {'IN': ['RB','RBR','RBS']}, 'OP': '*'},    # =POS:ADV, but avoiding TAG:WRB (what, when)
                    ]

        basic_verb = [
                      {'TAG': 'TO',   'OP': '*'},                        # infinitival 'to'
                      {'POS': 'AUX',  'OP': '*'},                        # 'were' in 'you were called out'
                      {'POS': 'VERB', 'OP': '+'},
                    ]

        #  e.g. taking on tirelessly
        verb_plus = basic_verb + plus

        # e.g. dynamically translating, just left, to best communicate
        adv_verb = [
                      {'TAG': 'TO',   'OP': '*'},                        # infinitival 'to'
                      {'TAG': {'IN': ['RB','RBR','RBS']}, 'OP': '+'},    # =POS:ADV, but avoiding TAG:WRB (what, when)
                      {'POS': 'VERB', 'OP': '+'},
                    ]

        # e.g. constantly taking on
        adv_verb_plus = adv_verb + plus

        modal_vphrase = [
                      {'TAG': 'MD',   'OP': '+'},
                      {'POS': 'VERB', 'OP': '+'},
        ]

        modal_vphrase_plus = modal_vphrase + plus

        # want to thank [and learn]
        want_to_thank = [
                      {'TAG': 'MD',    'OP': '*'},
                      {'POS': 'VERB',  'OP': '+'},
                      {'TAG': 'TO',    'OP': '+'},                       # infinitival 'to'
                      {'POS': 'VERB',  'OP': '+'},
                      {'POS': 'CCONJ', 'OP': '*'},
                      {'POS': 'VERB',  'OP': '*'},
                    ]

        want_to_thank_plus = want_to_thank + plus

        # want to thank you [both]
        want_to_thank2 = want_to_thank + [
                      {'TAG': 'RP',   'OP': '*'},                        # adverb particle (come ON)
                      {'TAG': 'IN',   'OP': '*'},                        # preposition or subordinating conjunction
                      {'TAG': 'PRP',  'OP': '+'},
                      {'TAG': 'DT',   'OP': '*'},
                    ]

        # e.g. try to figure things out (try to help me, try to help me out)
        want_to_thank3 = want_to_thank + [
                      {'TAG': 'PRP',  'OP': '*'},
                      {'POS': 'ADJ',  'OP': '*'},
                      {'POS': 'NOUN', 'OP': '*'},
                      {'TAG': 'RP',   'OP': '+'},                        # adverb particle (come ON)
                    ]

        # e.g. making changes
        making_changes = [
                      {'TAG': 'VBG',  'OP': '+'},
                      {'TAG': 'RP',   'OP': '*'},                        # adverb particle (come ON)
                      {'POS': 'ADJ',  'OP': '*'},
                      {'POS': 'NOUN', 'OP': '+'},
                    ]


        # e.g. am/is/was amazed, has(VBZ) demonstrated(VBN)
        verb_phrase1 = [
                      {'TAG': {'IN': ['VBP','VBZ','VBD','AUX']}, 'OP': '+'},
                      {'TAG': 'VBN', 'OP': '+'},
        ]

        verb_phrase1_plus = verb_phrase1 + plus

        # e.g. am/is/was contemplating
        verb_phrase2 = [
                      {'TAG': {'IN': ['VBP','VBZ','VBD','AUX']}, 'OP': '+'},
                      {'TAG': 'VBG', 'OP': '+'},
                    ]

        verb_phrase2_plus = verb_phrase2 + plus

        # e.g. was grueling(JJ) - VBG mistaken for JJ
        verb_phrase3 = [
                      {'TAG': {'IN': ['VBP','VBZ','VBD','AUX']}, 'OP': '+'},
                      {'TAG': 'JJ', 'OP': '+'},
                    ]

        # e.g. was grueling(JJ) today(NN) - VBG mistaken for JJ, ADV for NN
        verb_phrase4 = [
                      {'TAG': {'IN': ['VBP','VBZ','VBD', 'AUX']}, 'OP': '+'},
                      {'TAG': 'JJ', 'OP': '+'},
                      {'TAG': 'NN', 'OP': '+'},
                    ]

        # there is
        verb_phrase5 = [
                      {'TAG':   'EX',  'OP': '+'},
                      {'LEMMA': 'be',  'OP': '+'},
                      {'POS':   'ADV', 'OP': '*'},
                    ]

        # 'be safe', 'stay confident'
        verb_phrase6 = [
                      {'POS': 'ADV',  'OP': '*'},
                      {'POS': 'VERB', 'OP': '+'},
                      {'POS': 'ADJ',  'OP': '+'},
                    ]

        # 'whatever comes'
        verb_phrase7 = [
                      {'TAG': 'WDT',  'OP': '+'},
                      {'POS': 'VERB', 'OP': '+'},
                    ]

        # 'all while doing'
        verb_phrase8 = [
                      {'TAG': 'DT',  'OP': '*'},
                      {'TAG': 'IN',  'OP': '+'},
                      {'TAG': 'VBG', 'OP': '+'},
                    ]

        # 'deserving of note'
        verb_phrase9 = [
                      {'POS': 'ADJ',  'OP': '?'},
                      {'TAG': 'VBG',  'OP': '+'},
                      {'TEXT': {'IN': ['of','Of','OF']}, 'OP': '+'},
                      {'POS': 'ADV',  'OP': '*'},
                      {'POS': 'ADJ',  'OP': '*'},
                      {'POS': 'NOUN', 'OP': '+'},
                    ]

        # 'proud of you'
        verb_phrase10 = [
                      {'POS': 'ADJ',  'OP': '?'},
                      {'TAG': 'VBG',  'OP': '?'},
                      {'TAG': 'VBD',  'OP': '?'},
                      {'TEXT': {'IN': ['of','Of','OF']}, 'OP': '+'},
                      {'POS': 'PRON', 'OP': '+'},
                    ]

        # 'for working hard'
        verb_phrase11 = [
                     {'TAG': 'IN',   'OP': '*'},                        # preposition
                     {'TAG': 'VBG',  'OP': '+'},
                     {'TAG': {'IN': ['RB','RBR','RBS']},  'OP': '+'},   # =POS:ADV, but avoiding TAG:WRB (what, when)
                    ]

        # 'that you do'
        verb_phrase12 = [
                      {'TAG': 'WDT',  'OP': '+'},
                      {'TAG': 'PRP',  'OP': '+'},
                      {'TAG': {'IN': ['RB','RBR','RBS']},  'OP': '*'},  # =POS:ADV, but avoiding TAG:WRB (what, when)
                      {'POS': 'VERB', 'OP': '+'},
                    ]

        # 'that Dana does'
        verb_phrase14 = [
                      {'TAG': 'WDT',  'OP': '+'},
                      {'POS': 'DET',  'OP': '*'},                      # POS:DET includes possessive pronouns (TAG:PRP$)
                      {'POS': 'ADV',  'OP': '*'},
                      {'POS': 'ADJ',  'OP': '*'},
                      {'POS': {'IN': ['NOUN','PROPN']}, 'OP': '+'},
                      {'TAG': {'IN': ['RB','RBR','RBS']},  'OP': '*'},  # =POS:ADV, but avoiding TAG:WRB (what, when)
                      {'POS': 'VERB', 'OP': '+'},
                    ]

        # I know that / why / which / who
        verb_phrase15 = [
                     {'POS': 'PRON',  'OP': '+'},
                     {'POS': 'VERB',  'OP': '+'},
                     {'POS': 'SCONJ', 'OP': '*'},
                     {'TAG': 'WRB',   'OP': '*'},
                     {'TAG': 'WDT',   'OP': '*'},
                     {'TAG': 'WP',    'OP': '*'},
                    ]

        verb_phrase16 = [
                     {'TAG': 'MD',   'OP': '+'},
                     {'TAG': 'RB',   'OP': '+'},
                     {'TAG': 'VB',   'OP': '*'},
                     {'TAG': 'DT',   'OP': '*'},
                    ]

        verb_phrases = [ basic_verb, verb_plus, adv_verb, adv_verb_plus, modal_vphrase,
                         modal_vphrase_plus, want_to_thank, want_to_thank_plus, want_to_thank2,
                         want_to_thank3, making_changes, verb_phrase1, verb_phrase1_plus, verb_phrase2,
                         verb_phrase2_plus, verb_phrase3, verb_phrase4, verb_phrase5,
                         verb_phrase6, verb_phrase7, verb_phrase8, verb_phrase9, verb_phrase10,
                         verb_phrase11, verb_phrase12, verb_phrase14, verb_phrase15, verb_phrase16,
                       ]


        # NOUN PHRASES

        basic_noun = [
                     {'POS': {'IN': ['NOUN','PROPN']},  'OP': '+'},
                    ]

        basic_hyphn_noun = [
                     {'POS': {'IN': ['NOUN','PROPN']},  'OP': '+'},
                     {'TAG': 'HYPH',  'OP': '+'},
                     {'POS': {'IN': ['NOUN','PROPN']},  'OP': '+'},
                     {'POS': 'NOUN',   'OP': '*'},
                    ]

        noun_plus = [
                     {'POS': 'ADJ',   'OP': '*'},
                    ]

        noun_plus2 = [
                     {'POS': 'DET',   'OP': '*'},
                     {'TAG': 'PRP$',  'OP': '*'},
                     {'POS': 'ADV',   'OP': '*'},
                     {'POS': 'ADJ',   'OP': '*'},
                     {'POS': 'NUM',   'OP': '*'},
                    ]

        basic_noun_plus         = noun_plus + basic_noun
        basic_noun_plus2        = noun_plus2 + basic_noun
        basic_hyphn_noun_plus   = noun_plus + basic_hyphn_noun
        basic_hyphn_noun_plus2  = noun_plus2 + basic_hyphn_noun

        noun_phrase1 = [
                     {'TAG': 'DT',  'OP': '*'},
                     {'TAG': 'IN',  'OP': '+'},                        # 'as for me'
                     {'TAG': 'PRP', 'OP': '+'},
                    ]

        noun_phrase2 = [
                     {'POS':   'NUM', 'OP': '+'},
                     {'LEMMA': '%',   'OP': '+'}
                    ]

        noun_phrase3 = [
                     {'TAG': 'IN',    'OP': '+'},                       # preposition
                     {'TAG': 'VBN',   'OP': '+'},                       # beyond appreciated
                   ]

        noun_phrase4 = [
                     {'TAG': 'RB',    'OP': '+'},                       # so incredibly proud
                     {'POS': 'ADJ',   'OP': '+'},
                   ]

        # [the / his] continued efforts
        noun_phrase5 = [
                     {'POS': 'DET',   'OP': '*'},
                     {'TAG': 'PRP$',  'OP': '*'},
                     {'POS': 'ADV',   'OP': '*'},
                     {'TAG': 'VBN',   'OP': '+'},
                     {'POS': 'NOUN',  'OP': '+'},
                   ]

        # on-going issues (ADV + VBG + NOUN)
        noun_phrase6 = [
                     {'TAG': 'IN',    'OP': '*'},
                     {'POS': 'ADV',   'OP': '*'},
                     {'TAG': 'HYPH',  'OP': '*'},
                     {'TAG': 'VBG',   'OP': '+'},
                     {'POS': 'ADJ',   'OP': '*'},
                     {'POS': 'NOUN',  'OP': '+'},
                   ]

        # no matter where / what
        noun_phrase7 = [
                     {'TEXT': {'IN': ['no','No','NO']},             'OP': '+'},
                     {'TEXT': {'IN': ['matter','Matter','MATTER']}, 'OP': '+'},
                     {'TAG': 'WRB',  'OP': '*'},
                     {'TAG': 'WP',   'OP': '*'},
                   ]

        # current way of working
        noun_phrase8 = basic_noun + [
                     {'TEXT': {'IN': ['of','Of','OF']}, 'OP': '+'},
                     {'POS': 'NOUN', 'OP': '+'},
                   ]

        # current way of working
        noun_phrase8_plus = basic_noun_plus + [
                     {'TEXT': {'IN': ['of','Of','OF']}, 'OP': '+'},
                     {'POS': 'NOUN', 'OP': '+'},
                   ]

        # the month of March
        noun_phrase9 = basic_noun + [
                     {'TEXT': {'IN': ['of','Of','OF']}, 'OP': '+'},
                     {'POS': 'PROPN', 'OP': '+'},
                   ]

        noun_phrase9_plus = basic_noun_plus + [
                     {'TEXT': {'IN': ['of','Of','OF']}, 'OP': '+'},
                     {'POS': 'PROPN', 'OP': '+'},
                   ]

        # 'for all'
        noun_phrase10 = [
                     {'TAG': 'IN', 'OP': '+'},
                     {'TAG': 'DT', 'OP': '+'},
                    ]

        noun_phrase11 = [
                     {'POS': 'NOUN',  'OP': '+'},                         # 'check-in counters'
                     {'TAG': 'HYPH',  'OP': '+'},
                     {'TAG': 'RP',    'OP': '+'},
                     {'POS': 'NOUN',  'OP': '+'},
                    ]

        noun_phrase11_plus = noun_plus + [
                     {'POS': 'NOUN',  'OP': '+'},                         # 'check-in counters'
                     {'TAG': 'HYPH',  'OP': '+'},
                     {'TAG': 'RP',    'OP': '+'},
                     {'POS': 'NOUN',  'OP': '+'},
                    ]

        noun_phrase12 = basic_noun + [
                     {'TAG': 'IN',   'OP': '+'},                          # 'attention to detail'
                     {'POS': 'NOUN', 'OP': '+'},
                    ]

        noun_phrase12_plus = basic_noun_plus + [
                     {'TAG': 'IN',   'OP': '+'},                          # 'attention to detail'
                     {'POS': 'NOUN', 'OP': '+'},
                    ]

        noun_phrase14 = [
                     {'POS': 'ADV',   'OP': '+'},                         # 'above and beyond'
                     {'POS': 'CCONJ', 'OP': '+'},
                     {'POS': 'ADV',   'OP': '+'},
                    ]

        noun_phrase15 = [
                     {'TAG': 'VBG',   'OP': '+'},                         # 'tracking and reporting to us'
                     {'TAG': 'CC',    'OP': '+'},
                     {'TAG': 'VBG',   'OP': '+'},
                     {'TAG': 'IN',    'OP': '*'},
                     {'TAG': 'PRP',   'OP': '*'},
                    ]

        noun_phrases = [ basic_noun, basic_hyphn_noun, basic_noun_plus, basic_noun_plus2,
                         basic_hyphn_noun_plus, basic_hyphn_noun_plus2,
                         noun_phrase1, noun_phrase2, noun_phrase3, noun_phrase4, noun_phrase5,
                         noun_phrase6, noun_phrase7, noun_phrase8, noun_phrase8_plus, noun_phrase9,
                         noun_phrase9_plus, noun_phrase10, noun_phrase11, noun_phrase11_plus,
                         noun_phrase12, noun_phrase12_plus, noun_phrase14, noun_phrase15,
                       ]

        matcher = Matcher(self.nlp.vocab)
        matcher.add( 'noun_phrases', noun_phrases )
        matcher.add( 'verb_phrases', verb_phrases )

        return matcher


    def fetch_keyphrases(self, text_, display_tree=False):
        '''
            Uses Berkley Neural Parser to extract noun, prepositional, verb and other phrases with
            max_len = max number of words in phrase. The remaining single words are combined into phrases using an extended set
            of part-of-speech patterns.
            If display_tree = True, the parsed constituency tree is visualized
            Arguments:
                text_      - text as one string to have keywords extracted from
            Returns:
                keyphrases - list of all keyphrases as json objects
        '''
        if not isinstance(text_, str):
            print(f"Warning: expected type str, got {type(text_)} as input")
            return []
        elif not text_:
            print('Warning: got empty string as input')
            return []

        punct        = set([ '`', "'", '"', '#', '-', '_', ',', '.', ';', ':', '?', '!', '+',
                             '[', ']', '(', ')', '$', '%', '&', '@', '\\', '/', '<', '>', '^', '|', '~', ])
        res          = []
        words        = []
        seen_tokens  = set()
        max_len      = 5
        self.doc     = self.nlp( text_ )


        ##### CONSTITUENCY PARSING (Berkeley Neural Parser) #####

        for sent in list(self.doc.sents):

            # display the constituency tree
            if display_tree:
                tree = Tree.fromstring( sent._.parse_string )
                if is_notebook():
                    display(tree)
                else:
                    draw_trees(tree)

            for c in list(sent._.constituents):
                # skip whole sentences and long spans
                if 'S' in c._.labels or\
                   len(c.text.split()) > max_len or\
                   c.text.strip().lower() in punct:
                    continue

                children_labels = list( sum([i._.labels for i in list(c._.children)], ()) )

                # split NPs w/2+ NPs, PPs w/2+ PPs, VPs w/2+ NPs
                if ('NP' in c._.labels and children_labels.count('NP') > 1) or\
                   ('NP' in c._.labels and children_labels.count('PP') > 1) or\
                   ('PP' in c._.labels and children_labels.count('NP') > 1) or\
                   ('PP' in c._.labels and children_labels.count('PP') > 1) or\
                   ('VP' in c._.labels and children_labels.count('NP') > 1) or\
                   ('VP' in c._.labels and children_labels.count('PP') > 1):
                    continue

                # split VPs w/NP inside if phrase is too long
                if ('VP' in c._.labels and 'NP' in children_labels):
                   #and len(c.text.split()) >= 5:
                    continue

                if c.start not in seen_tokens and c.end - 1 not in seen_tokens:
                    if len(c.text.split()) > 1:
                        res.append(c)
                        seen_tokens.update(range(c.start, c.end))
                        #print(f'Appending constituent {c.text}')
                    else:
                        words.append(c)
                        #print(f'Appending word {c.text}')


        ##### FIT IN POS PATTERNS #####

        # get additional spans based on POS patters
        matcher = self.build_matcher()
        matches = matcher(self.doc)
        spans   = [self.doc[start:end] for _, start, end in matches]
        spans   = [ span for span in spans if len(span.text.split()) > 1 ]
        #print('Spacy spans:', spans)

        # sort additional spans, fit in
        get_sort_key = lambda span: (span.end - span.start, -span.start)
        sorted_spans = sorted(spans, key=get_sort_key, reverse=True)

        for span in sorted_spans:
            if span.start not in seen_tokens and span.end - 1 not in seen_tokens:

                # additional precaution since constituents may be shorter
                mid_point1 = span.start + int((span.end - span.start)*0.4)
                mid_point2 = span.start + int((span.end - span.start)/0.8)
                if mid_point1 not in seen_tokens and mid_point2 not in seen_tokens:
                    res.append(span)
                    seen_tokens.update(range(span.start, span.end))


        ##### FIT IN WORDS #####
        for word in words:
            if word.start not in seen_tokens and word.end - 1 not in seen_tokens:
                res.append(word)
                seen_tokens.update(range(word.start, word.end))

        res = sorted(res, key = lambda x: x.start)

        # if only 1 span and it's the whole sentence (usually a very short one)
        #if len(res) == 1 and res[0].text == text_:
        #    return [ {'text':  token.text,
        #              'start': token.idx,
        #              'end':   token.token.idx + len(token.text), } for token in self.doc ]

        return [ {'text':  span.text,
                  'start': span.start_char,
                  'end':   span.end_char, } for span in res ]


    def apply_cutoff(self, res_, cutoff_, reverse=False):
        '''
            Look for drop in importance and curtail res_
            Looking from start w/max_cutoff
            Looking from end with min_cutoff
        '''
        if not isinstance(res_, list) or len(res_) < 2:
            return res_

        if not reverse:
            for i in range(len(res_)-1):
                    if 1 - res_[i+1][0]/res_[i][0] >= cutoff_:
                        res_ = res_[:i+1]
                        break
        else:
            for i in range(len(res_)-1, 0, -1):
                    if 1 - res_[i][0]/res_[i-1][0] >= cutoff_:
                        res_ = res_[:i]
                        break

        return res_


    def filter_spans_local(self, spans):
        '''
            Modified spaCy's filter_spans(). spans = List[(importance, span)]
            Filter a sequence of spans and remove duplicates or overlaps. When spans overlap, the (first)
            longest span is preferred over shorter spans.
        '''
        get_sort_key = lambda span: (span[1]['end'] - span[1]['start'], -span[1]['start'])
        sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
        result       = []
        seen_tokens  = set()
        for span in sorted_spans:
            # Check for end - 1 here because boundaries are inclusive
            if span[1]['start'] not in seen_tokens and span[1]['end'] - 1 not in seen_tokens:
                result.append(span)
                seen_tokens.update(range(span[1]['start'], span[1]['end']))
        result = sorted(result, reverse=True, key=lambda x: x[0])                     # adds computational complexity

        return result


    def color_print(self, kps_):
        '''
            Print evaluated sentence and
            highlight biased keywords
        '''
        # transform keyphrase (kp)
        sent_   = self.doc.text
        kps_     = [i[1] for i in kps_]                                        # remove importance scores
        kps_flat = []
        for item in kps_:                                                     # flatten lists of tuples, if any
            if isinstance(item, list) or isinstance(item, tuple):
                kps_flat.extend(item)
            else:
                kps_flat.append(item)
        kps_flat = sorted( kps_flat, key=lambda x: x['start'] )                # deduplicate after flattening
        stack = ['']
        for i in kps_flat:
            if i != stack[-1]:
                stack.append(i)
        kps_flat = stack[1:]

        #kps_flat_short = []                                                  # remove smaller chunks if they are part of bigger
        #kps_text       = [i['text'] for i in kps_flat]
        #for i in kps_flat:
        #    if not any(i['text'] in j for j in kps_text):
        #        kps_flat_short.append(i)
        kps_text = [i['text'] for i in kps_flat]                               # get text (excluding start/end indices)

        # break sentence into segments using keyphrase (kp) boundaries
        chunks = []
        bounds = [(kp['start'], kp['end']) for kp in kps_flat]
        bounds = [item for sublist in bounds for item in sublist]
        if bounds[0]  != 0:          bounds.insert(0, 0)
        if bounds[-1] < len(sent_):  bounds.append( len(sent_) )
        chunks = [sent_[i:j] for i,j in list(zip(bounds, bounds[1:]+[None])) if sent_[i:j]]

        # color print biased keyphrases
        highlight = colored.fg('black') + colored.attr("bold") + colored.bg('light_goldenrod_2c')
        regular   = colored.fg('black')
        color_str, curr_length = [], 0
        for chunk in chunks:
            if chunk in kps_text:
                color_str.append( colored.stylize(chunk, highlight) )
            else:
                color_str.append( colored.stylize(chunk, regular) )

        print(''.join(color_str).strip())
