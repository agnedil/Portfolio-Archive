import math
import metapy
import sys
import time

#def make_classifier(training, inv_idx, fwd_idx):
    #"""
    #Use this function to train and return a Classifier object of your
    #choice from the given training set. (You can ignore the inv_idx and
    #fwd_idx parameters in almost all cases, but they are there if you need
    #them.)
#
#    **Make sure you update your config.toml to change your feature
#    representation!** The data provided here will be generated according to
#    your specified analyzer pipeline in your configuration file (which, by
#    default, is going to be unigram words).
#
#    Also, if you change your feature set and already have an existing
#    index, **please make sure to delete it before running this script** to
#    ensure your new features are properly indexed.
#    """
#    return metapy.classify.NaiveBayes(training, alpha=0.055, beta=0.1)
#    #return metapy.classify.OneVsAll(training, metapy.classify.SGD, loss_id='hinge')
#    #return metapy.classify.KNN(training, inv_idx, 5, metapy.index.OkapiBM25(k1=1.2, b=0.75, k3=500))

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

    f = open("maximum.txt", "w+")
    f.close
    f1 = open("maximum1.txt", "w+")
    f1.close
    for i in range(645, 777):
        iter = i / 10000000000.0
        matrix = metapy.classify.cross_validate(lambda fold:
            metapy.classify.NaiveBayes(fold, alpha=iter, beta=0.19), dset, 5)

        acc = matrix.accuracy()
        with open("maximum1.txt", "a") as myfile1:
            myfile1.write(str(iter) + "," + str(acc) + "\n")
        if acc > 0.95:
            with open ("maximum.txt", "a") as myfile:
                myfile.write(str(iter) + "," + str(acc) + "\n")
        #print(matrix)
        #matrix.print_stats()

    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))