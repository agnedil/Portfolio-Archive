# based on http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
# http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
# in the last reference - see note about Kullback-Leibler divergence = PLSA
# So, thev below is an NMF-based of PLSA

from sklearn.feature_extraction.text import TfidfVectorizer             # add CountVectorizer for LDA
from sklearn.decomposition import NMF                                   # add LatentDirichletAllocation for LDA
#from sklearn.datasets import fetch_20newsgroups                        # using my own data
import logging
import glob
import argparse
from gensim import models
from gensim import matutils
from time import time
from nltk.tokenize import sent_tokenize
import math
import graph_tool.all as gt


class Visualization:
    # adjusting graph_tool

    color_map = {0: [1.0, 0.5, 0.5, 0.3], 1: [0.5, 1.0, 0.5, 0.3], 2: [0.5, 0.5, 1.0, 0.3],
                 3: [1.0, 0.5, 0.5, 0.3], 4: [0.5, 1.0, 0.5, 0.3], 5: [0.5, 0.5, 1.0, 0.3],
                 6: [1.0, 0.5, 0.5, 0.3], 7: [0.5, 1.0, 0.5, 0.3], 8: [0.5, 0.5, 1.0, 0.3],
                 9: [0.5, 1.0, 0.5, 0.3]}

    def __init__(self, root_label="All Reviews", term_count=100,
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


#dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
#documents = dataset.data

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

datafile = "review_sample_usedForPositive_150000.txt"
with open(datafile) as file:
    documents = file.readlines()
print("Data read! Running NMF")


# NMF uses tf-idf
num_features = 10000
tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=num_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts because it is a probabilistic graphical model
#tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
#tf = tf_vectorizer.fit_transform(documents)
#tf_feature_names = tf_vectorizer.get_feature_names()
#print("LDA features obtained")

# run NMF
num_topics = 10
nmf_model = NMF(n_components=num_topics, beta_loss='kullback-leibler', solver='mu', random_state=1, alpha=.1, l1_ratio=.5, init='nndsvda', max_iter=400).fit(tfidf)
print("NMF run")

# Run LDA
#lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
#print("LDA run")

output_text = []
allReviewsViz = Visualization()

numWords_toShow = 15
for topic_idx, topic in enumerate(nmf_model.components_):

    #crerate a list of words and a list of their weights in descending order
    words = []
    weights = []
    for i in topic.argsort()[:-numWords_toShow - 1:-1]:
        words.append(feature_names[i])
        weights.append(topic[i])

    # topic_idx is the topic number
    output_text.append("Topic: " + str(topic_idx))
    if topic_idx<10:
        vtopic = allReviewsViz.add_topic("Topic: " + str(topic_idx))        # add 10 topics to tree-diagram

    # find max weight
    #maxWeight = 0.0
    #for idx, item in enumerate(weights):
    #    if idx<10:
    #        maxWeight = max(maxWeight, item)

    # 15 top terms to list and 10 terms (optimal number) to tree diagram
    for idx, item in enumerate(words):
        output_text.append( words[idx] + " : " + str(weights[idx]) )
        if idx<10:
            allReviewsViz.add_term(vtopic, words[idx], weight=weights[idx])


allReviewsViz.draw("ztask1_1_nmf_all.png")
print "writing topics to file"
with open ("zsample_topics_nmf_all.txt", 'w' ) as f:
    f.write('\n'.join(output_text))
print "done!"
