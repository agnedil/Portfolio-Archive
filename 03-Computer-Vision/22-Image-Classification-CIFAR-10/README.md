# 20-Component PCA on CIFAR-10 Images

Principal component analysis using the first 20 components on the CIFAR-10 dataset (10 categories of images) to implement the following data manipulations:

1. Computing the mean image and the first 20 principal components for each category. Plotting the error resulting from representing the images of each category using the first 20 principal components against the category.

2. Computing the distances between mean images (as vectors) for each pair of classes. Using principal coordinate analysis to make a 2D map of the means of each categories.

3. Using a different measure of the similarity of two classes: for class A and class B, define E(A | B) to be the average error obtained by representing all the images of class A using the mean of class A and the first 20 principal components of class B. The similarity between classes is to be (1/2)(E(A | B) + E(B | A)). If A and B are very similar, then this error should be small because A's principal components should be good at representing B. Using principal coordinate analysis to make a 2D map of the classes.
