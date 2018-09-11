#https://piazza.com/class/jchzguhsowz6n9?cid=564
library(grid)
disp_img <- function(img) {
  # hint from Ryan Yusko
  # and "baptiste" on StackOverflow https://stackoverflow.com/a/11306342
  r <- img[1:1024]
  g <- img[1025:2048]
  b <- img[2049:3072]
  img_matrix = rgb(r,g,b,maxColorValue=255)
  dim(img_matrix) = c(32,32)
  img_matrix = t(img_matrix) # fix to fill by columns
  grid.raster(img_matrix, interpolate=T)