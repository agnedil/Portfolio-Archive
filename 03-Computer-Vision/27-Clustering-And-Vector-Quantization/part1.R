# AML UIUC

# this code applies clustering to the "European Jobs" dataset located here http://lib.stat.cmu.edu/DASL/Datafiles/EuropeanJobs.html
# hint from Gabriel Martos at https://rpubs.com/gabrielmartos/ClusterAnalysis


rm(list = ls())
setwd('/home/andrew/Documents/')
library(ape)
rawData <- read.table("dataset.txt", header = TRUE, row.names = 1, sep = "\t")

# Part 1. Agglomerative clustering with hclust

# the three cases for which hclust clustering will be done; "methods" to be used in code, "mlabels" - in printed text
methods <- c("single", "complete", "average")
mlabels <- c("Single Link", "Complete Link", "Group Average")

# use hclust and plot results in a loop for each method
for (i in 1:length(methods))
    {
    # hclust for hierarchical clustering of the European Jobs dataset using methods from "methods"
    hclustresult <- hclust(dist(rawData), method = paste(methods[[i]]))
    
    # plot results and print automatically to 3 jpeg files
    jpeg(file=paste("dendogram_", methods[[i]], ".jpeg"), units="in", width=9, height=10, res=350)
    plot(as.phylo(hclustresult), type='fan', cex = 1.25, frame.plot = TRUE,
         main = paste("1979 European Employment Statistics by Country Dendrogram.\n Method - ", mlabels[[i]]),
         sub = "Dataset: European Jobs from http://lib.stat.cmu.edu/DASL/Datafiles/EuropeanJobs.html")
    dev.off()
    }

# Part 2. Kmeans clustering

# visualize k-means clustering results for several numbers of k
for (j in 2:7)
    {
    # k-means per se
    kmresult <- kmeans(rawData, j, nstart = 10)
    
    # distance matrix, multidimensional scaling to dim=2
    #scldata <- scale(rawData)
    dm <- dist(rawData)
    mds <- cmdscale(dm)
    
    #save plot for each k in a separate file
    jpeg(file=paste("kmeansClust_k", j, ".jpeg"), units="in", width=11, height=8.6, res=350)
    plot(mds, col = kmresult$cluster, main = paste("K-Means Clustering Results\n ( k = ", j, ")"),
         xlab="", ylab="", xlim=c(-18, 55), cex=1.5, pch = 19)
    text(mds, labels=row.names(rawData), pos = 4, col = kmresult$cluster, cex = 1.2)
    dev.off()
    }

# determine a good choice of k
# function to plot the # of clusters (nc) vs. within-cluster sums of squares (wcss) - visualizes the "elbow". See line 7
plotElbow <- function(data, nc=15, seed=1234)
    {
    wcss <- (nrow(data)-1)*sum(apply(data,2,var))
    for (i in 2:nc){
        set.seed(seed)
        wcss[i] <- sum(kmeans(data, centers=i)$withinss)}
    plot(1:nc, wcss, type="b", xlab="Number of clusters",
         ylab="Within-cluster sums of squares")
    }

# the resulting plot confirms that a good choice of k is 3
jpeg(file="goodK_elbowCriterion.jpeg", units="in", width=5.5, height=4.25, res=350)
plotElbow(rawData, nc=7)
dev.off()
