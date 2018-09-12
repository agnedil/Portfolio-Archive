# Clustering and Vector Quantization

1. Dataset: European employment in 1979 at http://lib.stat.cmu.edu/DASL/Stories/EuropeanJobs.html. This dataset gives the percentage of people employed in each of a set of areas in 1979 for each of a set of European countries. Using it for visualization of clustering.

Agglomerative clustering of this data and producing a dendrogram for each of single link, complete link, and group average clustering using the hclust clustering function and turning the result into a phylogenetic tree as a fan plot: plot(as.phylo(hclustresult), type='fan'). The dendrograms "make sense" and have interesting differences. The clustergin of this dataset using k-means, making a good choice of k.


2. Dataset: https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerom from the UC Irvine machine learning

Building a classifier that classifies sequences into one of the 14 activities provided using vector quantization (with hierarchical k-means), a histogram of cluster centers. Improving the classifier by (a) modifying the number of cluster centers in the hierarchical k-means and (b) modifying the size of the fixed length samples used for vector quantization.
