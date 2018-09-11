# AML UIUC
# this code applies hierarchical kmeans clustering as vector quantization and then random forest classification
# to the "ADL Recognition Dataset" dataset located at https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer

# hint from Tommy at https://stackoverflow.com/questions/7376499/how-can-i-read-multiple-files-from-multiple-directories-into-r-for-processing
# hint Harlan at https://stackoverflow.com/questions/3318333/split-a-vector-into-chunks-in-r

rm(list = ls())
setwd('/home/andrew/Documents/2_UIUC/')
library(randomForest)
library(factoextra)
library(rdist)

# funciton to read 1 txt file, flatten by row, split into length-96 chunks, remove last chunk if irregular size
processFile <- function(f)
    {
    df <- read.csv(f, header=FALSE, sep = " ")
    df <- c(t(df))
    seglist <- split(df, ceiling(seq_along(df)/segsize))
    if (length(unlist(tail(seglist, 1))) < segsize) {seglist[length(seglist)] <- NULL}
    return(seglist)
    }

# function to build a histogram for one file
toHist <- function(myfile)
{
    this.hist <- numeric(numk)
    mtxcenters <- matrix(clresult$centers, nrow = numk, ncol = segsize)
    for (i in 1:length(myfile))
    {
        mymtx <- matrix(myfile[[i]], nrow = 1, ncol = segsize)
        eucdm <- cdist(mymtx, mtxcenters)                           # Euclidean dist from segment to each row of kmeans centers mtx
        minidx <- which.min(eucdm)                                  # index of the shortest dist
        this.hist[[minidx]] <- this.hist[[minidx]] + 1              # increase hist at this index
    }# for i
    return(this.hist)
}# toHist

# list of 14 categories, to be used in a loop
categories <- c("Brush_teeth", "Climb_stairs", "Comb_hair", "Descend_stairs", "Drink_glass", "Eat_meat", "Eat_soup", "Getup_bed", "Liedown_bed", "Pour_water", "Sitdown_chair", "Standup_chair", "Use_telephone", "Walk")

numk <- 400                                                         # k for kmeans
numk1<- 401                                                         # k+1 for matrix operations
segsize <- 48                                                       # segment size

# data from all training files into one 3-tier list: 1) category level 2) file level 3) length-96 chunks
trainset <- list()
for (i in 1:length(categories))
    {
    path <- paste("train/", categories[[i]], sep="")
    files <- dir(path, recursive=TRUE, full.names=TRUE, all.files = TRUE)# pattern="*.txt")
    thiscat = lapply(files, processFile)
    trainset[[i]] <- thiscat
    }

# data from all testing files into one 3-tier list: 1) category level 2) file level 3) length-96 chunks
testset <- list()
for (j in 1:length(categories))
    {
    path <- paste("test/", categories[[j]], sep="")
    files <- dir(path, recursive=TRUE, full.names=TRUE, all.files = TRUE)
    thiscat = lapply(files, processFile)
    testset[[j]] <- thiscat
    }
# all training length-96 segment into one dataframe for hierarchical k-means clustering
df.kmeans <- list()
for (j in 1:length(trainset))
    {
    df.kmeans[[j]] <- as.data.frame(matrix(unlist(trainset[[j]]), ncol=length(unlist(trainset[[1]][[1]][[1]]))))
    }
df.kmeans <- as.data.frame(do.call(rbind, df.kmeans))

# hkmeans
clresult <- hkmeans(df.kmeans, numk)

# creat a matrix of all training histograms, labels in 1:14, create trainx and trainy sets
listHist <- numeric(numk1)
for (count in 1:14)
    {
    for (subcount in 1:length(trainset[[count]]))
        {
        temphist <- toHist(trainset[[count]][[subcount]])
        temphist <- c(temphist, count)
        listHist <- rbind(listHist, temphist)
        }# for subcount
    }# for count    
rownames(listHist) <- NULL                                      # remove unnecessar rownames
listHist <- listHist[-1,]                                       # not sure how to avoid the first row of zeros otherwise
trainx <- as.data.frame(listHist[,-numk1])
trainy <- as.factor(listHist[,numk1])                             # required for randomForest

# same mtx of hist for testing
listHistTest <- numeric(numk1)
for (count in 1:14)
{
    for (subcount in 1:length(testset[[count]]))
    {
        temphist <- toHist(testset[[count]][[subcount]])
        temphist <- c(temphist, count)
        listHistTest <- rbind(listHistTest, temphist)
    }# for subcount
}# for count    
rownames(listHistTest) <- NULL
listHistTest <- listHistTest[-1,]
testx <- as.data.frame(listHistTest[,-numk1])
testy <- as.factor(listHistTest[,numk1])

# randomForest stats
clf <- randomForest(trainx, y=trainy,  xtest=testx, ytest=testy, ntree=500)
