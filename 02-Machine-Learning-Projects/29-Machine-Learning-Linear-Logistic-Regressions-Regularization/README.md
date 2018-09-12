# Linear and Logistic Regressions With Various Regularizations

1. Linear regression with various regularizers using a dataset giving features of music, and the latitude and longitude from which that music originates (UCI Machine Learning dataset repository) - predicting latitude and longitude from these features; ignore dealing with outliers, regard latitude and longitude as entirely independent.

    a) Building a straightforward linear regression of latitude (resp. longitude) against features, retrieving R-squared and plotting a graph

    b) Using the Box-Cox transformation to improve the regressions

    c) Using glmnet to produce:

    - a regression regularized by L2 (ridge regression), estimating the regularization coefficient that produces the minimum error;

    - a regression regularized by L1 (lasso regression), estimating the regularization coefficient that produces the minimum error;

    - a regression regularized by elastic net (equivalently, a regression regularized by a convex combination of L1 and L2) trying three values of alpha, the weight setting how big L1 and L2 are, and estimating the regularization coefficient that produces the minimum error.


2. Logistic regression. Dataset: a dataset giving whether a Taiwanese credit card user defaults against a variety of features (the UCI Machine Learning dataset repository). Predicting whether the user defaults; ignore outliers, but try the above various regularization schemes.
