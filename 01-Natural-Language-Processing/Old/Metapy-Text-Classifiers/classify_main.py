import math
import metapy
import sys
import time
import re
#from cucco import Cucco

def make_classifier(training, inv_idx, fwd_idx):
    """
    Use this function to train and return a Classifier object of your
    choice from the given training set. (You can ignore the inv_idx and
    fwd_idx parameters in almost all cases, but they are there if you need
    them.)

    **Make sure you update your config.toml to change your feature
    representation!** The data provided here will be generated according to
    your specified analyzer pipeline in your configuration file (which, by
    default, is going to be unigram words).

    Also, if you change your feature set and already have an existing
    index, **please make sure to delete it before running this script** to
    ensure your new features are properly indexed.
    """

    #return metapy.classify.DualPerceptron(training, kernel, alpha=0.1, gamma=0.05, bias=0.0, max_iter=100L)
    #return metapy.classify.Winnow (training, m=1.4, gamma=0.07, max_iter=10)
    return metapy.classify.NaiveBayes(training, alpha=0.0000000727, beta=0.19)
    #return metapy.classify.OneVsAll(training, metapy.classify.SGD, loss_id='hinge')
    #return metapy.classify.KNN(training, inv_idx, 5, metapy.index.OkapiBM25(k1=0.79, b=0.97, k3=500))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    metapy.log_to_stderr()

    cfg = sys.argv[1]
    print('Building or loading indexes...')
    inv_idx = metapy.index.make_inverted_index(cfg)
    fwd_idx = metapy.index.make_forward_index(cfg)

    dset = metapy.classify.MulticlassDataset(fwd_idx)

    print('Running cross-validation...')
    start_time = time.time()
    matrix = metapy.classify.cross_validate(lambda fold:
            make_classifier(fold, inv_idx, fwd_idx), dset, 10)

    print(matrix)
    matrix.print_stats()
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))