# Workhuman IQ
# @author: Andrew Nedilko


import re
import ftfy
import pycld2 as cld2
import emoji


contractions = (
                 ("I'm",    "I am"),
                 ("i'm",    "i am"),
                 ("'ll ",   " will "),
                 ("'d ",    " would "),
                 ("'ve ",   " have "),
                 ("'re ",   " are "),
                 ("won't",  "will not"),
                 ("what's", "what is"),
                 ("that's", "that is"),
                 ("it's",   "it is"),
                 ("here's", "here is"),
                 ("let's",  "let us"),
                 ("'cause", "because"),
                 ("can't",  "cannot"),
                 ("ain't",  "is not"),
                 ("n't",    " not"),
                )


def repair_text( s ):
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


def clean_text( s,
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
    s = repair_text( s )

    # convert to ascii
    if to_ascii:
        try:
            s = s.encode('ascii', 'ignore').decode()
        except:
            pass

    # remove multiple spaces
    s = re.sub('\s+', ' ', s)

    return s.strip()


def detect_lang( s ):
    '''
        Return the language(s) in string s.
        Naive Bayes classifier under the hood -
        results are less certain for strings that are too short.
        Returns up to three languages with confidence scores.
        More on usage: https://pypi.org/project/pycld2/
    '''
    _, _, details = cld2.detect( s )
    return details[0][0]


def unfold_contractions( s ):
    '''
        Unfold common English contractions: e.g. "I'm" => "I am"
    '''
    for i,j in contractions:
        if i in s:
            s = s.replace( i, j )

    return s


def remove_emoji(s, to_text=False):
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
