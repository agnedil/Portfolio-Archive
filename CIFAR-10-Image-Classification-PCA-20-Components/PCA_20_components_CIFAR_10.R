# AML 2018 UIUC
# hint from matt at https://stackoverflow.com/questions/32113942/importing-cifar-10-data-set-to-r
# hint from Ryan Yusko and "baptiste" at https://stackoverflow.com/a/11306342
# Read binary file and convert to integer vectors [otherwise if reading directly as integer(), the first bit is read as as signed]
# Original file format is 10000 records following the pattern: [label x 1][red x 1024][green x 1024][blue x 1024]
# NOT broken into rows, so need to be careful with "size" and "n": http://www.cs.toronto.edu/~kriz/cifar.html

rm(list = ls())
setwd('/home/andrew/Documents/2_UIUC/3_CS498_AML/Homeworks/HW3/inwork')
library(grid)
library(ape)
library(MASS)

txtlabels <- read.table("cifar-10-batches-bin/batches.meta.txt")                # list of label names (airplane, automobile, bird, cat, deer, dog, frog, horse,ship, truck)
labels <- list()                                                                # list of all labels ordered as images (60K)
images <- list()                                                                # list of all images (60K)
categories <- list()                                                            # list of image/label indices from images and labels above, grouped by each of the 10 categories
num.images = 10000                                                              # used to retrieve all images per each of 6 binary files

# function to print an image
drawImage <- function(img)
{
    # Convert each color layer into a color vector, combine into an rgb object, shape as matrix, and plot
    #img <- images[[index]]                                     #used when img index is passed as arg, not img itself
    r <- img[1:1024]
    g <- img[1025:2048]
    b <- img[2049:3072]
    img_matrix = rgb(r, g, b, maxColorValue=255)
    dim(img_matrix) = c(32,32)
    img_matrix = t(img_matrix)                                  # since we have vectors as rows
    grid.raster(img_matrix, interpolate=FALSE)                  # grid library
    remove(r, g, b, img_matrix)                                 #clean up
    #txtlabels[[1]][labels[[index]]]                            #print the image label for debugging when img index is passed as arg
}

# Cycle through all 6 binary files, place images into lists of vectors;
# IMPORTANT: "test_batch.bin" renamed to "data_batch_6.bin" for convenience of the cycle
for (f in 1:6)
    {
    to.read <- file(paste("cifar-10-batches-bin/data_batch_", f, ".bin", sep=""), "rb")
    for(i in 1:num.images)
        {
        label <- readBin(to.read, integer(), size=1, n=1, endian="big")
        imgvector <- as.integer(readBin(to.read, raw(), size=1, n=3072, endian="big"))
        index <- num.images * (f-1) + i
        images[[index]] = imgvector
        labels[[index]] = label+1                                # since the original labels are numbered 0 through 9
        }
    close(to.read)
    remove(label, imgvector, f, i, index, to.read)               # clean up   
    }
print("Done reading raw files")

#print random image for debugging
#drawImage(sample(1:(num.images*6), size=1))

# group indices (of images/labels) by each of the 10 categories
# grouping images themselves into a list is not plausible as it takes a whole lot more time (by a couple of orders of magnitude (several second vs. an hour))
for (categ in 1:10)
    {
    tempcat <- c()
    for (j in 1:length(labels))
        {
        if (labels[[j]] == categ) { tempcat <- c(tempcat, j) }
        }
    categories[[categ]] <- tempcat
    remove(tempcat, j, categ)
    }
print("Done grouping by category")

# calculate and save mean image for each category
meanimg <- list()
for (categ in 1:10)
    {
    index <- categories[[categ]]
    meancatimg <- Reduce(`+`, images[index])/length(categories[[categ]])              #calculates for each cat (group of 6000 images)
    meanimg[[categ]] <- meancatimg
    jpeg(file=paste("meanimg_cat", categ, ".jpeg", sep=""))                           # additional options: units="in", width=5.5, height=4.25, res=750)
    drawImage(meanimg[[categ]])
    dev.off()
    remove(meancatimg, index, categ)
    }
print("Done building mean images")

# run PCA for 20 PCs
eigvls <- list()                                    #list of eigenvalues from mypca$sdev^2
#eigvctrs <- list()                                 #list of eigenvectors from mypca$rotation
#means <- list()                                    #list of mean values from mypca$center (requires subindexing [[1]] to get a hold of)
pcomp20 <- list()                                   #list of parameters for 20 PCs in each category from mypca$x
abserror20 <- c()                                   #abs. error of representing each category with its first 20 PCs
relerror20 <- c()                                   #rel. error of representing each category with its first 20 PCs
for (categ in 1:10)
    {
    index <- categories[[categ]]
    x <- as.data.frame(images[index])
    mypca <- prcomp(x, retx = TRUE, center = TRUE, scale. = TRUE, rank. = 20)
    eigvls[[categ]]  <- mypca$sdev^2
    pcomp20[[categ]] <- mypca$x
    ae <- sum(eigvls[[categ]][21:3072])
    abserror20 <- c(abserror20, ae)
    re <- sum(eigvls[[categ]][21:3072])/sum(eigvls[[categ]])
    relerror20 <- c(relerror20, re)
    cat("Done PCA for category ", categ, "\n")
    remove(index, x, categ)
    }

# 1) plot absolute and relative error of representing each category with its first 20 PCs
#    (Part 1 of the assignment)

jpeg(file="abserror_20cat.jpeg", units="in", width=5.5, height=4.25, res=250)
barplot(abserror20, names.arg = c("airplane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"),
        col = 'light blue', main = "Absolute Error", cex.names = 0.65, log = "y",
        xlab = "Fig.2a Abs. error of representing each category with first 20 PCs", ylab = "Error",
        xlim = NULL, ylim = c(1400, 2300), axes = TRUE, axisnames = TRUE, axis.lty = 1)
dev.off()

jpeg(file="relerror_20cat.jpeg", units="in", width=5.5, height=4.25, res=250)
barplot(relerror20*100, names.arg = c("airplane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"),
        col = 'light green', main = "Relative Error", cex.names = 0.65, log = "y",
        xlab = "Fig.2b Rel. error of representing each category with first 20 PCs", ylab = "Error (%)",
        xlim = NULL, ylim = c(24, 40), axes = TRUE, axisnames = TRUE, axis.lty = 1)
dev.off()

# 2) principal coordinate analysis and 2D map based on distances between mean images of the classes
#    (Part 2 of the assignment)

x <- matrix(unlist(meanimg), ncol = 3072, byrow = TRUE)
dm <- dist(x, method = "euclidean", diag = TRUE)
mypcoa = pcoa(dm, rn = c("airplane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"))
jpeg(file="meanimg_dist_part2.jpeg", units="in", width=5.5, height=4.25, res=350)
biplot(mypcoa, main="Part 2. Similarity of classes (PCoA - mean images)", plot.axes = c(1, 2),
       xlim = c(-1500, 2000), ylim = c(-1000, 1000))
dev.off()
write.matrix(dm, file = "part2_distance_matrix.txt", sep = "  ")
remove(x, dm) #mypcao

# 3) principal coordinate analysis and 2D map based on average error
#    (1/2)(E(A | B) + E(B | A))
#    Part 3 of the assignment

# function to calculate a|b
averr <- function (a, b)
{
    a <- matrix(unlist(a), ncol = 3072, byrow = TRUE)           # [1:3072], mean image of a class
    b <- matrix(unlist(b), ncol = 3072, byrow = TRUE)           # [20:3072], 20 PCs of a class
    b <- t(b)
    res <- a%*%b                                                # [1:20], projecting to 20 PCs
    res <- res%*%t(b)                                           # [1:3072], projecting back to the size of a
    return(res)                                                 # [1:3072]
}

# initialize distance matrix based on average error between the classes
distmat <- matrix(0, 10, 10)

# populate the matrix
for (i in 1:10)
    {
    for (j in 1:10)
        {
        if (i!=j)
            {
            aa <- meanimg[[i]]                                      # [1:3072], mean image of a class
            bb <- pcomp20[[j]]                                      # [20:3072], 20 PCs of a class
            eab <- averr(aa, bb)                                    # E(A|B), [1:3072]
            bb <- meanimg[[j]]
            aa <- pcomp20[[i]]
            eba <- averr(bb, aa)                                    # E(B|A), [1:3072]
            similarity <- (eab + eba) / 2                           # (1/2)(E(A|B) + E(B|A)), [1:3072]
            similarity <- (meanimg[[i]] - similarity)^2             # squared error, [1:3072]; i OR j here don't matter - same result
            toMatrix <- sum(similarity)                             # 1 number, sum of squares
            distmat[[i, j]] <- toMatrix
            
            remove(aa, bb, eab, eba, similarity, toMatrix)
            }#if
        }#for j
    }#for i

# runn PCoA for this matrix and plot the results
#dm <- as.dist(distmat, diag = TRUE)
dm <- dist(distmat, method = "euclidean", diag = TRUE)
mypcoa2 <- pcoa(dm, rn = c("airplane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"))
jpeg(file="averr_dist_part3.jpeg", units="in", width=5.5, height=4.25, res=350)
biplot(mypcoa2, main="Part 3. Similarity of classes (PCoA - average error)", plot.axes = c(1, 2))
dev.off()
write.matrix(distmat, file = "part3_similarity_matrix.txt", sep = " ")
#remove(distmat, i, j, a, b, mypcao2)
