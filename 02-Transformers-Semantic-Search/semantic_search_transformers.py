import pandas as pd
import time
from sentence_transformers import SentenceTransformer, util
import torch


model_name = 'all-MiniLM-L12-v2'
embedder   = SentenceTransformer( model_name )

corpus = None
corpus_embeddings = None
sub_ind = None


def get_template() -> dict:
    return { 
             'score': 0,
             'phrase': '',
             'kb_phrase': '',
             'subindustry': '',
             'method': '',
           }


def fetch_similar( queries: list,                   
                   top_k: int=5,
                   threshold: int=0.75, ) -> list[dict]:

    if not isinstance(queries, list):
        raise ValueError(f'Expected type list for queries; got {type(queries)}')    
    elif not isinstance(top_k, int):
        raise ValueError(f'Expected type int for top_k; got {type(top_k)}')
    elif not isinstance(threshold, float):
        raise ValueError(f'Expected type float for threshold_; got {type(threshold)}')

    if not queries:
        return queries
    
    query_embeddings  = embedder.encode(queries, convert_to_tensor=True)
    hits              = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k)    # hits = list
    if not hits:
        return hits

    res_all = []
    for i in range(len(hits)):    
        for j in range(len(hits[i])):
            if hits[i][j]['score'] >= threshold:
                score = hits[i][j]['score']
                kb_phrase = corpus[hits[i][j]['corpus_id']]
                res = get_template()
                                
                res['score'] = round(score, 5)
                res['phrase'] = queries[i]
                res['kb_phrase'] = kb_phrase
                res['subindustry'] = sub_ind
                res['method'] = 'semantic'
                res_all.append(res)
                
    return res_all


def apply_fetch_similar( df_: pd.core.frame.DataFrame,
                         col: str,
                         knowledgebase: dict,
                         sub_ind_local: str,
                       ) -> pd.core.frame.DataFrame:
    '''
        Using pre-defined keyphrases for a specific subindustry, seach df_[col] for similar phrases
        Usage from a jupyter notebook:
            import semantic_search_ransformers as sst
            col_np = 'column_with_noun_phrases'
            df = sst.apply_fetch_similar(df, col_np, knowledgebase, sub_ind,)
                
        :param df_: pandas dataframe where sentence transformers must be applied,
        :param col: column in df_ containing parsed noun phrases,
        :param knowledgebase: a dict() containing manually selected keyphrases which are used
                              for semantic search in df_[col],
        :param sub_ind: specific sub-industry from knowledgebase for the semantic similarity search,
                
        :returns: df_ with additional column 'sim_transformers_' + col containing phrases from df_[col]
                  deemed to be semantically similar to keyphrases from knowledgebase[sub_ind]
        
    '''        
    global sub_ind, corpus, corpus_embeddings
    sub_ind           = sub_ind_local
    corpus            = list(knowledgebase[ sub_ind ]['semantic'])
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    
    start = time.time()
    df_['sim_transformers_'+col] = df_[col].apply( lambda x: fetch_similar(x) if isinstance(x, list) else x )   
    print( f"Time elapsed: {round( (time.time() - start)/60, 4 )} min" )
    
    return df_


if __name__ == '__main__':
    
    test_kb = { 'appliances': {'semantic': {'appliaces', 'household appliances'} } }
    sub_ind = 'appliances'
    corpus            = list(test_kb[ sub_ind ]['semantic'])
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    
    queries = [ 'electric appliances', 'home appliances' ]
    result  = fetch_similar(queries)
    print(result)
    
    
    # EXPECTED OUTPUT
    # [{'score': 0.89393, 'phrase': 'electric appliances', 'kb_phrase': 'household appliances', 'subindustry': 'appliances', 'method': 'semantic'},\
    # {'score': 0.97027, 'phrase': 'home appliances', 'kb_phrase': 'household appliances', 'subindustry': 'appliances', 'method': 'semantic'}]
   