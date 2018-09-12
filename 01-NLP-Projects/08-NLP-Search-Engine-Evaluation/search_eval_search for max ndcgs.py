import math
import sys
import string
import metapy
import pytoml


#class InL2Ranker(metapy.index.RankingFunction):
    #"""
    #Create a new ranking function in Python that can be used in MeTA.
    #"""
    #def __init__(self, some_param=1.0):
        #self.param = some_param
        # You *must* call the base class constructor here!
        #super(InL2Ranker, self).__init__()

    #def score_one(self, sd):
        #"""
        #You need to override this function to return a score for a single term.
        #For fields available in the score_data sd object,
        #@see https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html
        #"""
        #return (self.param + sd.doc_term_count) / (self.param * sd.doc_unique_terms + sd.doc_size)

        #tfn = sd.doc_term_count*math.log((1+sd.avg_dl/sd.doc_size), 2)
        #return (sd.query_term_weight*tfn /(tfn+self.param)*math.log((sd.num_docs+1)/(sd.corpus_term_count+0.5), 2))

#def load_ranker(cfg_file):
    #"""
    #Use this function to return the Ranker object to evaluate.
    #The parameter to this function, cfg_file, is the path to a
    #configuration file used to load the index. You can ignore this for MP2.
    #"""
    #You can set your new InL2Ranker here by: return InL2Ranker(some_param=1.0) 
    #Try to set the value between 0.9 and 1.0 and see what performs best

    #Best BM25 k1 value:
    #return metapy.index.OkapiBM25(k1=1.8583, b=0.75, k3=500)

    #Best InL2 c value:
    #return InL2Ranker(some_param=0.987)

    #return metapy.index.PivotedLength(s = 0.311)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    print('Building or loading index...')
    idx = metapy.index.make_inverted_index(cfg)
    #ranker = load_ranker(cfg)
    ev = metapy.index.IREval(cfg)

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d['query-runner']
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    top_k = 10
    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)

    query = metapy.index.Document()

    f = open ("maximum.txt", "w+")
    f.close
    mylist = []
    for i in range (10, 2500, 10):
        kay = i / 1000.0
        for j in range (10, 1000, 10):
            bee = j / 1000.0
            list_ndcg = []
            triple = []
            ranker = metapy.index.OkapiBM25(k1=kay, b=bee, k3=500)
            with open(query_path) as query_file:
                for query_num, line in enumerate(query_file):
                    query.content(line.strip())
                    results = ranker.score(idx, query, top_k)
                    avg_ndcg = float(ev.ndcg(results, query_start + query_num, top_k))
                    list_ndcg.append(avg_ndcg)
            mndcg = math.fsum (list_ndcg) / float(len(list_ndcg))
            if mndcg > 0.36:
                triple = [kay, bee, mndcg]
                mylist.append(triple)
                with open ("maximum.txt", "a") as myfile:
                    myfile.write(str(triple[0]) + "," + str(triple[1]) + "," + str(triple[2]) + "\n")
            print(str(i) + "." + str(j) + " nDCG {}".format(mndcg))
    for item in mylist:
        print(string.translate(str(item), None, "'") + "\n")
