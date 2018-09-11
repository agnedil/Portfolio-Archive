# Read binary file and convert to integer vectors
# [Necessary because reading directly as integer() 
# reads first bit as signed otherwise]
#
# File format is 10000 records following the pattern:
# [label x 1][red x 1024][green x 1024][blue x 1024]
# NOT broken into rows, so need to be careful with "size" and "n"
#
# (See http://www.cs.toronto.edu/~kriz/cifar.html)
labels <- read.table("cifar-10-batches-bin/batches.meta.txt")
images.rgb <- list()
images.lab <- list()
num.images = 10000 # Set to 10000 to retrieve all images per file to memory

# Cycle through all 5 binary files
for (f in 1:5) {
    to.read <- file(paste("cifar-10-batches-bin/data_batch_", f, ".bin", sep=""), "rb")
    for(i in 1:num.images) {
        l <- readBin(to.read, integer(), size=1, n=1, endian="big")
        r <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
        g <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
        b <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
        index <- num.images * (f-1) + i
        images.rgb[[index]] = data.frame(r, g, b)
        images.lab[[index]] = l+1
    }
    close(to.read)
    remove(l,r,g,b,f,i,index, to.read)
}

# function to run sanity check on photos & labels import
drawImage <- function(index) {
    # Testing the parsing: Convert each color layer into a matrix,
    # combine into an rgb object, and display as a plot
    img <- images.rgb[[index]]
    img.r.mat <- matrix(img$r, ncol=32, byrow = TRUE)
    img.g.mat <- matrix(img$g, ncol=32, byrow = TRUE)
    img.b.mat <- matrix(img$b, ncol=32, byrow = TRUE)
    img.col.mat <- rgb(img.r.mat, img.g.mat, img.b.mat, maxColorValue = 255)
    dim(img.col.mat) <- dim(img.r.mat)
    
    # Plot and output label
    library(grid)
    grid.raster(img.col.mat, interpolate=FALSE)
    
    # clean up
    remove(img, img.r.mat, img.g.mat, img.b.mat, img.col.mat)
    
    labels[[1]][images.lab[[index]]]
}

drawImage(sample(1:(num.images*5), size=1))