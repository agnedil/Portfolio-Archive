# DM Capstone
# MCSDS UIUC
# Task 1
# File provided by teaching staff and modified by me

import logging
import glob
import argparse
from gensim import models
from gensim import matutils
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from nltk.tokenize import sent_tokenize
import math
import graph_tool.all as gt


class Visualization:
    """adjusting graph_tool"""

    color_map = {0: [1.0, 0.5, 0.5, 0.3], 1: [0.5, 1.0, 0.5, 0.3], 2: [0.5, 0.5, 1.0, 0.3],
                 3: [1.0, 0.5, 0.5, 0.3], 4: [0.5, 1.0, 0.5, 0.3], 5: [0.5, 0.5, 1.0, 0.3],
                 6: [1.0, 0.5, 0.5, 0.3], 7: [0.5, 1.0, 0.5, 0.3], 8: [0.5, 0.5, 1.0, 0.3],
                 9: [0.5, 1.0, 0.5, 0.3]}

    def __init__(self, root_label="Positive Reviews", term_count=100,
                 root_color=[0.5, 0.5, 1.0, 0.5], root_font_size=18):
        self.g = gt.Graph()

        self.vprop_label = self.g.new_vertex_property("string")
        self.text_position = self.g.new_vertex_property("float")
        self.text_rotation = self.g.new_vertex_property("float")
        self.fill_color = self.g.new_vertex_property("vector<float>")
        self.font_size = self.g.new_vertex_property("int")
        self.vertex_size = self.g.new_vertex_property("int")
        self.marker_size = self.g.new_edge_property("float")
        self.text_offset = self.g.new_vertex_property("vector<float>")

        self.root = self.g.add_vertex()
        self.vprop_label[self.root] = root_label
        self.text_position[self.root] = -1
        self.text_rotation[self.root] = 0
        self.fill_color[self.root] = root_color
        self.font_size[self.root] = root_font_size
        self.text_offset[self.root] = [0.0, 0.0]

        self.angle = 0.0
        self.position = 0.0
        self.first_turn = True
        self.cur_text_offset = [-0.07, 0.0]
        self.topic_idx = 0
        self.cur_term_color = Visualization.color_map[0]

        self.angle_step = 2 * math.pi / term_count

    def add_topic(self, label, root=None, vcolor=[0.0, 0.7, 0.5, 0.87], voffset=[0.0, 0.0],
                  emarker_size=2.0, vfont_size=16):
        vtopic = self.g.add_vertex()

        self.vprop_label[vtopic] = label
        self.text_position[vtopic] = -1
        self.text_rotation[vtopic] = 0
        self.fill_color[vtopic] = vcolor
        self.font_size[vtopic] = vfont_size
        self.text_offset[vtopic] = voffset

        if root == None:
            root = self.root

        e = self.g.add_edge(root, vtopic)
        self.marker_size[e] = emarker_size

        self.cur_term_color = Visualization.color_map[self.topic_idx % 10]
        self.topic_idx += 1

        return vtopic

    def add_term(self, vtopic, term, vfont_size=20, emarker_size=1.0, weight=1.0, vertex_size=50):
        vterm = self.g.add_vertex()
        self.vprop_label[vterm] = term.encode('utf8')
        self.text_position[vterm] = self.position
        self.text_rotation[vterm] = self.angle

        self.fill_color[vterm] = self.cur_term_color
        self.fill_color[vterm][3] *= weight

        self.font_size[vterm] = vfont_size
        self.text_offset[vterm] = self.cur_text_offset

        self.text_offset[vterm][0] *= vertex_size / 50.0
        self.text_offset[vterm][1] *= vertex_size / 50.0

        self.vertex_size[vterm] = vertex_size

        et = self.g.add_edge(vtopic, vterm)
        self.marker_size[et] = emarker_size

        self.increaseAngle(vertex_size)

    def increaseAngle(self, vertex_size):
        self.angle += self.angle_step

        if self.angle > math.pi / 2.0:
            if self.first_turn:
                self.angle *= -1.0
                self.position = math.pi
                self.first_turn = False
                self.cur_text_offset = [0.07, 0.0]
            else:
                self.angle *= -1
                self.position = 0
                self.cur_text_offset = [-0.07, 0.0]

    def draw(self, file_name, output_size=(1980, 1980)):
        # for straight edges, use only one line below instead of the next 5 lines before gt.graph_draw
        # tpos = pos = gt.radial_tree_layout(self.g, self.root)

        state = gt.minimize_nested_blockmodel_dl(self.g, deg_corr=True)
        t = gt.get_hierarchy_tree(state)[0]
        tpos = pos = gt.radial_tree_layout(self.g, self.root)
        cts = gt.get_hierarchy_control_points(self.g, t, tpos, beta=0.2)
        pos = self.g.own_property(tpos)

        gt.graph_draw(self.g, bg_color=[1,1,1,1], vertex_text=self.vprop_label, vertex_text_position=self.text_position, \
                      vertex_text_rotation=self.text_rotation, vertex_fill_color=self.fill_color, \
                      output=file_name, output_size=output_size, inline=True, vertex_font_size=self.font_size, \
                      edge_marker_size=self.marker_size, vertex_text_offset=self.text_offset, \
                      vertex_size=self.vertex_size, vertex_anchor = 0, pos=pos, edge_control_points=cts, fit_view=0.9)


def main(K, numfeatures, sample_file, num_display_words, outputfile):
    K_clusters = K
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=numfeatures,
                                     min_df=2, stop_words='english',
                                     use_idf=True)

    text = []
    with open (sample_file, 'r') as f:
        text = f.readlines()

    t0 = time()
    print("Extracting features from the training dataset using a sparse vectorizer")
    X = vectorizer.fit_transform(text)
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    
    # mapping from feature id to actual word
    id2words ={}
    for i,word in enumerate(vectorizer.get_feature_names()):
        id2words[i] = word

    t0 = time()
    print("Applying topic modeling, using LDA")
    print(str(K_clusters) + " topics")
    corpus = matutils.Sparse2Corpus(X,  documents_columns=False)
    lda = models.ldamodel.LdaModel(corpus, num_topics=K_clusters, id2word=id2words, passes=10, iterations=100)
    print("done in %fs" % (time() - t0))
        
    output_text = []

    allReviewsViz = Visualization()

    # i=topic #; element=list of tuples (term, weight) -> had to modify the below 4 lines from original code
    for i, item in enumerate(lda.show_topics(num_topics=K_clusters, num_words=num_display_words, log=False, formatted=False)):
        output_text.append("Topic: " + str(i))
        if i<10:
            vtopic = allReviewsViz.add_topic("Topic: " + str(i))

        maxWeight = 0.0
        for idx, element in enumerate(item[1]):
            if idx<10:
                maxWeight = max(maxWeight, element[1])

        for idx, element in enumerate(item[1]):
            output_text.append( element[0] + " : " + str(element[1]) )
            if idx<10:
                allReviewsViz.add_term(vtopic, element[0], weight=element[1] / maxWeight)

    allReviewsViz.draw("task1_1_all.png")
    print "writing topics to file:", outputfile
    with open ( outputfile, 'w' ) as f:
        f.write('\n'.join(output_text))

        
if __name__=="__main__":

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print "using input file:", "10review_positive_sample_150000.txt"
    main(10, 10000, "10review_positive_sample_150000.txt", 15, "sample_topics_positive.txt")

# Below is the complete original content in the if __name__=="__main__" section
# To restore: delete everything above and uncomment everything below

#    parser = argparse.ArgumentParser(description='This program takes in a file and some parameters and generates topic modeling from the file. This program assumes the file is a line corpus, e.g. list or reviews and outputs the topic with words and weights on the console.')
#
#    parser.add_argument('-f', dest='path2datafile', default="review_sample_100000.txt",
#                       help='Specifies the file which is used by to extract the topics. The default file is "review_sample_100000.txt"')
#
#    parser.add_argument('-o', dest='outputfile', default="sample_topics.txt",
#                       help='Specifies the output file for the topics, The format is as a topic number followed by a list of words with corresdponding weights of the words. The default output file is "sample_topics.txt"')
#
#    parser.add_argument('-K', default=100, type=int,
#                       help='K is the number of topics to use when running the LDA algorithm. Default 100.')
#    parser.add_argument('-featureNum', default=50000, type=int,
#                       help='feature is the number of features to keep when mapping the bag-of-words to tf-idf vectors, (eg. lenght of vectors). Default featureNum=50000')
#    parser.add_argument('-displayWN', default=15,type=int,
#                       help='This option specifies how many words to display for each topic. Default is 15 words for each topic.')
#    parser.add_argument('--logging', action='store_true',
#                       help='This option allows for logging of progress.')
    
    
#    args = parser.parse_args()
    #print args
#    if args.logging:
#       logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#    print "using input file:", args.path2datafile
#    main(args.K, args.featureNum, args.path2datafile, args.displayWN, args.outputfile)
