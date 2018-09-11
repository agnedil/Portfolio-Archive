# https://stackoverflow.com/questions/32113942/importing-cifar-10-data-set-to-r
# Read binary file and convert to integer vectors [otherwise if reading directly as integer(), the first bit is read as as signed]
# File format is 10000 records following the pattern: [label x 1][red x 1024][green x 1024][blue x 1024]
# NOT broken into rows, so need to be careful with "size" and "n"#
# (dataset and description: http://www.cs.toronto.edu/~kriz/cifar.html)

setwd('/home/andrew/Documents/2_UIUC/3_CS498_AML/Homeworks/HW3')

labels <- read.table("cifar-10-batches-bin/batches.meta.txt")
images.rgb <- list()
images.lab <- list()
num.images = 10000 # Set to 10000 to retrieve all images per file to memory

# Cycle through all 6 binary files, place images into list of dataframes;
# "test_batch.bin" renamed to "data_batch_6.bin" for convenience
for (f in 1:6)
    {
    to.read <- file(paste("cifar-10-batches-bin/data_batch_", f, ".bin", sep=""), "rb")
    for(i in 1:num.images)
        {
        label <- readBin(to.read, integer(), size=1, n=1, endian="big")
        red   <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
        green <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
        blue  <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
        index <- num.images * (f-1) + i
        images.rgb[[index]] = data.frame(red, green, blue)
        images.lab[[index]] = label+1
        }
    close(to.read)
    remove(label,red,green,blue,f,i,index, to.read)
    }

# function to print an image
drawImage <- function(index)
    {
    # Convert each color layer into a matrix,
    # combine into an rgb object, and display as a plot
    img <- images.rgb[[index]]
    img.r.mat <- matrix(img$red,   ncol=32, byrow = TRUE)
    img.g.mat <- matrix(img$green, ncol=32, byrow = TRUE)
    img.b.mat <- matrix(img$blue,  ncol=32, byrow = TRUE)
    img.col.mat <- rgb(img.r.mat, img.g.mat, img.b.mat, maxColorValue = 255)
    dim(img.col.mat) <- dim(img.r.mat)
    
    # Plot and output label
    library(grid)
    grid.raster(img.col.mat, interpolate=FALSE)
    
    # clean up
    remove(img, img.r.mat, img.g.mat, img.b.mat, img.col.mat)
    
    labels[[1]][images.lab[[index]]]
    }

drawImage(sample(1:(num.images*6), size=1))

library(grid)
drawImage2 <- function(img)
    {
    # hint from Ryan Yusko
    # and "baptiste" on StackOverflow https://stackoverflow.com/a/11306342
    r <- img[1:1024]
    g <- img[1025:2048]
    b <- img[2049:3072]
    img_matrix = rgb(r,g,b,maxColorValue=255)
    dim(img_matrix) = c(32,32)
    img_matrix = t(img_matrix) # fix to fill by columns
    grid.raster(img_matrix, interpolate=T)
    }
categories <- list()
for (idx in 1:10)
    {
    templist <- list()
    for (j in 1:length(images.lab))
        {
        if (images.lab[[j]] == idx) templist <- c(templist, j)
        }
    categories[[idx]] <- templist
    }