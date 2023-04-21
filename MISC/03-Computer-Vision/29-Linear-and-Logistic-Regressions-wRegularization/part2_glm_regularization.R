# AML UIUC

# Part 2. Logistic regression with various regularizers
# Logistic regression and plots based on a UCI ML dataset containing explanatory variables about customersused to predict 
# whether they will default on their payments: http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

library(glmnet)
library(gdata)
library(boot)

# read data
rm(list = ls())
setwd('/home/andrew/Documents/')
input <- read.xls('default of credit card clients.xls', skip=1, header = TRUE)

x <- as.matrix(input[,2:24])
y <- as.factor(input[,25])

# unregularized general logistic regression
myglm <- glm(y~x, family = "binomial")                                      # there is no R-squared for glm
cat ("Deviance of the general logistic regression:", myglm$deviance, "\n")
par(mfrow=c(2,2))                                                           # print plots as a 2 x 2 matrix
plot(myglm)

# cross-validation of glm to be able to compare with regularized models
par(mfrow=c(1,1))
mydata <- input[2:25]
cv.myglm <- suppressWarnings(cv.glm(mydata, myglm, K=100))         
misclasif.error.myglm <- cv.myglm$delta[2]                                  # misclassification error
cat("General logistic regression: misclassification error =", misclasif.error.myglm, "\n")
cat("General logistic regression: all 23 variables are used", "\n")


# regularized models
par(mfrow=c(3,2))

# ridge regression, alpha = 0
cv00 <- cv.glmnet(x, y, alpha=0,   family="binomial", type.measure = "class")
plot(cv00, main = "RIDGE REGRESSION, ALPHA=0",
     sub = paste("Min. error lam. = ", format(cv00$lambda.min, digits=4),
     " at misclassif. error = ", format(min(cv00$cvm), digits=4), ", # var = 23 (see on the plot)"))

# elnet regression, alpha = 0.25
cv25 <- cv.glmnet(x, y, alpha=.25, family="binomial", type.measure = "class")
cv25.coeffs <- coef(cv25, s = cv25$lambda.min)
numVari25 <- length(cv25.coeffs@x)-1                                        # to account for "intercept" according to Dinnames
plot(cv25, main = "ELNET REGRESSION, ALPHA=0.25",
     sub = paste("Min. error lam. = ", format(cv25$lambda.min, digits=4),
                 " at misclassif. error = ", min(cv25$cvm), ", # var =", numVari25))

# elnet regression, alpha = 0.5
cv50 <- cv.glmnet(x, y, alpha=.5,  family="binomial", type.measure = "class")
cv50.coeffs <- coef(cv50, s = cv50$lambda.min)
numVari50 <- length(cv50.coeffs@x)-1                                        # minus intercept
plot(cv50, main = "ELNET REGRESSION, ALPHA=0.50",
     sub = paste("Min. error lam. = ", format(cv50$lambda.min, digits=4),
                 " at misclassif. error = ", format(min(cv50$cvm), digits=4), ", # var =", numVari50))

# elnet regression, alpha = 0.75
cv75 <- cv.glmnet(x, y, alpha=.75, family="binomial", type.measure = "class")
cv75.coeffs <- coef(cv75, s = cv75$lambda.min)
numVari75 <- length(cv75.coeffs@x)-1                                        # minus intercept
plot(cv75, main = "ELNET REGRESSION, ALPHA=0.75",
     sub = paste("Min. error lam. = ", format(cv75$lambda.min, digits=4),
                 " at misclassif. error = ", min(cv75$cvm), ", # var =", numVari75))

# lasso regression, alpha = 1
cv10 <- cv.glmnet(x, y, alpha=1,   family="binomial", type.measure = "class")
cv10.coeffs <- coef(cv10, s = cv10$lambda.min)
numVari10 <- length(cv10.coeffs@x)-1                                        # minus intercept
plot(cv10, main = "LASSO REGRESSION, ALPHA=1",
     sub = paste("Min. error lam. = ", format(cv10$lambda.min, digits=4),
                 " at misclassif. error = ", format(min(cv10$cvm), digits=4), ", # var =", numVari10))

# binomial deviance as an alternative measure
# cv00_2 <- cv.glmnet(x, y, alpha=0,   family="binomial", type.measure = "deviance")
# cv25_2 <- cv.glmnet(x, y, alpha=.25, family="binomial", type.measure = "deviance")
# cv50_2 <- cv.glmnet(x, y, alpha=.5,  family="binomial", type.measure = "deviance")
# cv75_2 <- cv.glmnet(x, y, alpha=.75, family="binomial", type.measure = "deviance")
# cv10_2 <- cv.glmnet(x, y, alpha=1,   family="binomial", type.measure = "deviance")

# another altertive measure would be type.measure = "auc" for AUC (area under curve)
