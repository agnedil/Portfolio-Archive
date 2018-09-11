# AML UIUC

# Part 1. Linear regression with various regularizers
# Linear regression and plots based on a UCI ML dataset containing features of music, and the latitude and longitude from
# which that music originates: https://archive.ics.uci.edu/ml/datasets/Geographical+Original+of+Music

# used https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
# used https://www4.stat.ncsu.edu/~post/josh/LASSO_Ridge_Elastic_Net_-_Examples.html
# hint from fbt at https://stats.stackexchange.com/questions/107643/how-to-get-the-value-of-mean-squared-error-in-a-linear-regression-in-r
# hint from Jake Drew at https://stackoverflow.com/questions/21380236/cross-validation-for-glm-models
# hint from Jason Brownlee at https://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/


library(MASS)
library(glmnet)
library(caret)
library(lattice)
library(DAAG)

# read data
rm(list = ls())
setwd('/home/andrew/Documents/')
input <- read.table('default_plus_chromatic_features_1059_tracks.txt', header=FALSE, sep = ',')

#######################################################################
## 1 a)                                                              ##
## STRAIGHTFORWARD LINEAR REGRESSION                                 ##
#######################################################################
# linear regression of LATITUDE against features
x <- as.matrix(input[,1:116])                                       # this var is not really needed here, but will be used later
lat.y <- as.matrix(input[,117])                                     # same here
latData<-input[,-c(118)]                                            # remove longitude
latlm<-lm(lat.y~x, data=latData)                                    # another way: lm(y~., data=...) where "." means the rest of dataset (but y needs to be part of the dataset)

# 100-fold cross-validation to compare with regularized regressions, the more folds the better mse
# NOTE: either plotit or printit must be TRUE (default) because of a bug in the package
cv.latlm <- suppressWarnings(cv.lm(latData, V117~., m=100, printit=F))         
(latlm.cvmse <- attr(cv.latlm, "ms"))
#[1] 314

# using RMSE based on the caret package's train function as another option
model.latlm <- suppressWarnings(train(V117~., latData, method = "lm", trControl = trainControl(method = "cv", number = 100)))
(rmse.latlm <- model.latlm$results$RMSE)
#[1] 16.5

# plot the linear regression of latitude
plot(latlm$fitted.values, latlm$residuals, xlab = "Fitted Values", ylab = "Residual",
     main = "Straightforward regression of latitude",
     sub = paste("R2 = ", format(summary(latlm)$r.squared, digits=5), ", CV MSE = ", format(latlm.cvmse, digits=7)))
abline(h = 0, lty=2)

# linear regression of LONGITUDE against features
long.y <- as.matrix(input[,118])
longData<-input[,-c(117)]                                          # remove latitude
longlm<-lm(long.y~x, data=longData)

# 100-fold cross-validation to compare with regularized regressions, the more folds the better mse
# NOTE: either plotit or printit must be TRUE (default) because of a bug in the package
cv.longlm <- suppressWarnings(cv.lm(longData, V118~., m=100, printit=F))         
(longlm.cvmse <- attr(cv.longlm, "ms"))
#[1] 2090

# using RMSE based on the caret package's train function as another option
model.longlm <- suppressWarnings(train(V118~., longData, method = "lm", trControl = trainControl(method = "cv", number = 100)))
(rmse.longlm <- model.longlm$results$RMSE)
# [1] 42.8

plot(longlm$fitted.values, longlm$residuals, xlab = "Fitted Values", ylab = "Residual",
     main = "Straightforward regression of longitude",
     sub = paste("R2 = ", format(summary(longlm)$r.squared, digits=5), ", CV MSE = ", format(longlm.cvmse, digits=7)))
abline(h = 0, lty=2)

#######################################################################
## 1 b)                                                              ##
## BOX-COX TRANSFORMATION                                            ##
#######################################################################
# get rid of negative values by changing the origin
bcinput <- input
# as latitude ranges from -90 to 90
bcinput$V117 = input$V117 + 90                                        
# as longitude ranges from -180 to 180 (theoretically, although min(longitude) > -90 in our case, so I could have used 90)
bcinput$V118 = input$V118 + 180                                       

########################################################################
# boxcox transformation for LATITUDE
latbc.y <- as.matrix(bcinput$V117)
latData.bc <- bcinput[,-c(118)]                                         # remove longitude

mybc.lat <- boxcox(lm(latbc.y~x, data = latData.bc))
lambda1 <- mybc.lat$x[which.max(mybc.lat$y)]                            # best lambda                         

# the idea behind the below code for transformaion is based on:
# hint from mlegge at https://stackoverflow.com/questions/33999512/how-to-use-the-box-cox-power-transformation-in-r
# and from Arthur Charpentier at https://www.r-bloggers.com/onlatbc.y-box-cox-transform-in-regression-models/
if (lambda1 == 0) {
    latbc.y <- log(latbc.y)
} else {latbc.y <- (latbc.y^lambda1 - 1)/ lambda1}

# regression with transformed data
latlm.bc <- lm(latbc.y~x, data = latData.bc)
latlm.bc.r2 <- summary(latlm.bc)$r.squared                              # to use on the plot

# cross-validation with 100 folds (to be consistent with the straightforward lm)
# NOTE: either plotit or printit must be TRUE (default) because of a bug in the package
cv.latlm.bc <- suppressWarnings(cv.lm(latData.bc, V117~., m=100, printit=F))         
(latlmbc.cvmse <- attr(cv.latlm.bc, "ms"))
#[1] 541

# RMSE with caret's train as another option (RMSE)
model.latlm.bc <- suppressWarnings(train(V117~., latData.bc, method = "lm", trControl = trainControl(method = "cv", number = 100)))
(rmse.latlm.bc <- model.latlm.bc$results$RMSE)
# [1] 16.4

# reverse box-cox transformation to bring to the same coordinates
if (lambda1 == 0) {
    latlm.bc$fitted.values <- exp(latlm.bc$fitted.values)
} else {latlm.bc$fitted.values <- (latlm.bc$fitted.values*lambda1 + 1)^(1/lambda1)}

plot(latlm.bc$fitted.values, latlm.bc$residuals, xlab = "Fitted Values", ylab = "Residual",
     main = "Regression of latitude (with Box Cox)",
     sub = paste("R2 = ", format(latlm.bc.r2, digits=5), ", CV MSE = ", format(latlmbc.cvmse, digits=7)))
abline(h = 0, lty=2)

########################################################################
# boxcox transformation for LONGITUDE
longbc.y <- as.matrix(bcinput$V118)
longData.bc <- bcinput[,-c(117)]                                         # remove latitude

mybc.long <- boxcox(lm(longbc.y~x, data = longData.bc)) 
lambda2 <- mybc.long$x[which.max(mybc.long$y)]                          

if (lambda2 == 0) {
    longbc.y <- log(longbc.y)
} else {longbc.y <- ((longbc.y^lambda2 - 1)/ lambda2)}

# regression of transformed longitude
longlm.bc <- lm(longbc.y~x, data = longData.bc)
longlm.bc.r2 <- summary(longlm.bc)$r.squared                              # to use on the plot

# cross-validation with 100 folds (to be consistent with the straightforward lm)
cv.longlm.bc <- suppressWarnings(cv.lm(longData.bc, V118~., m=100, printit=F))         
(longlmbc.cvmse <- attr(cv.longlm.bc, "ms"))
#[1] 2832

# RMSE with caret's train as another option
model.longlm.bc <- suppressWarnings(train(V118~., longData.bc, method = "lm", trControl = trainControl(method = "cv", number = 100)))
(rmse.longlm.bc <- model.longlm.bc$results$RMSE)
# [1] 42.6

# reverse box-cox transformation to bring to the same coordinates
if (lambda2 == 0) {
    longlm.bc$fitted.values <- exp(longlm.bc$fitted.values)
} else {longlm.bc$fitted.values <- (longlm.bc$fitted.values*lambda2 + 1)^(1/lambda2)}

plot(longlm.bc$fitted.values, longlm.bc$residuals, xlab = "Fitted Values", ylab = "Residual",
     main = "Regression of lngitude (with Box Cox)",
     sub = paste("R2 = ", format(longlm.bc.r2, digits=5), ", CV MSE = ", format(longlmbc.cvmse, digits=7)))
abline(h = 0, lty=2)

# the box cox transformation does not improve the results; using original data hereinafter

#######################################################################
## 1 c)                                                              ##
## regularized regression using glmnet                               ##
#######################################################################

# LATITUDE
# some code tricks from course textbook, p. 202, Listing 8.2, etc.

x <- as.matrix(input[,1:116])
y <- as.matrix(input$V117)
foldid=sample(1:10,size=length(y),replace=TRUE)

# RIDGE regression; all requested parameters are listed on the plot
cvLatRidge <- cv.glmnet(x, y, foldid=foldid, alpha = 0)
r2Ridge <- cvLatRidge$glmnet.fit$dev.ratio[which(cvLatRidge$glmnet.fit$lambda == cvLatRidge$lambda.min)]
mseRidge <- cvLatRidge$cvm[cvLatRidge$lambda == cvLatRidge$lambda.min]

plot(cvLatRidge, main = "RIDGE REGRESSION, LATITUDE", sub = paste("R2 = ", format(r2Ridge, digits=3),
                    ", min. error lam. = ", format(cvLatRidge$lambda.min, digits=4), " at MSE = ",
                    format(mseRidge, digits=5)))

########################################################################
# LASSO regression; all requested parameters are listed on the plot
cvLatLasso <- cv.glmnet(x, y, foldid=foldid, alpha = 1)
r2Lasso <- cvLatLasso$glmnet.fit$dev.ratio[which(cvLatLasso$glmnet.fit$lambda == cvLatLasso$lambda.min)]
mseLasso <- cvLatLasso$cvm[cvLatLasso$lambda == cvLatLasso$lambda.min]

# number of variables used by the regression according to http://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
cvLatLasso.coeffs <- coef(cvLatLasso, s = cvLatLasso$lambda.min)
numVari = length(cvLatLasso.coeffs@i)                                   # "@i' - after examining 'str(cvLatLasso.coeffs)'

plot(cvLatLasso, main = "LASSO REGRESSION, LATITUDE",
     sub = paste("R2=", format(r2Lasso, digits=3), ", min. error lam.=", format(cvLatLasso$lambda.min, digits=4),
                 ", MSE=", format(mseLasso, digits=5), "# var=", numVari))

########################################################################
# ELASTIC NET regression; all requested parameters are listed on the plot

cvLatElnet25=cv.glmnet(x, y, foldid=foldid, alpha=.25)
cvLatElnet50=cv.glmnet(x, y, foldid=foldid, alpha=.5)
cvLatElnet75=cv.glmnet(x, y, foldid=foldid, alpha=.75)

# estimate R2s, best lambdas, MSE, number of variables for each of the three case
r2Elnet25 <- format(cvLatElnet25$glmnet.fit$dev.ratio[which(cvLatElnet25$glmnet.fit$lambda == cvLatElnet25$lambda.min)], digits=3)
r2Elnet50 <- format(cvLatElnet50$glmnet.fit$dev.ratio[which(cvLatElnet50$glmnet.fit$lambda == cvLatElnet50$lambda.min)], digits=3)
r2Elnet75 <- format(cvLatElnet75$glmnet.fit$dev.ratio[which(cvLatElnet75$glmnet.fit$lambda == cvLatElnet75$lambda.min)], digits=3)

lamElnet25 <- format(cvLatElnet25$lambda.min, digits=4)
lamElnet50 <- format(cvLatElnet50$lambda.min, digits=4)
lamElnet75 <- format(cvLatElnet75$lambda.min, digits=4)

mseElnet25 <- format(cvLatElnet25$cvm[cvLatElnet25$lambda == cvLatElnet25$lambda.min], digits=5)
mseElnet50 <- format(cvLatElnet50$cvm[cvLatElnet50$lambda == cvLatElnet50$lambda.min], digits=5)
mseElnet75 <- format(cvLatElnet75$cvm[cvLatElnet75$lambda == cvLatElnet75$lambda.min], digits=5)

cvLatElnet25.coeffs <- coef(cvLatElnet25, s = cvLatElnet25$lambda.min)
numVarElnet25 = length(cvLatElnet25.coeffs@i)
cvLatElnet50.coeffs <- coef(cvLatElnet50, s = cvLatElnet50$lambda.min)
numVarElnet50 = length(cvLatElnet50.coeffs@i)
cvLatElnet75.coeffs <- coef(cvLatElnet75, s = cvLatElnet75$lambda.min)
numVarElnet75 = length(cvLatElnet75.coeffs@i)

plot(log(cvLatElnet25$lambda), cvLatElnet25$cvm, pch=19, col="red",
     xlab="", ylab=cvLatElnet25$name, main = "ELASTIC NET, LATITUDE (log lambda)",
     sub = paste("R2s=", r2Elnet25,",", r2Elnet50, ",", r2Elnet75,
                ", min.error lam.=", lamElnet25, ",", lamElnet50, ",", lamElnet75, ",",
                "\n MSE=", mseElnet25, ",", mseElnet50, ",", mseElnet75,
                ", # var=", numVarElnet25, ",", numVarElnet50, ",", numVarElnet75))
points(log(cvLatElnet50$lambda), cvLatElnet50$cvm, pch=19, col="dark green")
points(log(cvLatElnet75$lambda), cvLatElnet75$cvm, pch=19, col="blue")
legend("topleft", legend=c("alpha=0.25","alpha=0.5","alpha=0.75"),
        pch=19, col=c("red","dark green","blue"))

# TODO: misclassification error using family = "multinomial", type.measure = "class"
# cvLatRidge2 <- cv.glmnet(x, y, foldid=foldid, alpha = 0, family = "multinomial", type.measure = "class")
# cvLatLasso2 <- cv.glmnet(x, y, foldid=foldid, alpha = 1, family = "multinomial", type.measure = "class")
# cvLatElnet25_2 <- cv.glmnet(x, y, foldid=foldid, alpha=.25, family = "multinomial", type.measure = "class")
# cvLatElnet50_2 <- cv.glmnet(x, y, foldid=foldid, alpha=.50, family = "multinomial", type.measure = "class")
# cvLatElnet75_2 <- cv.glmnet(x, y, foldid=foldid, alpha=.75, family = "multinomial", type.measure = "class")
# par (mfrow=c(1, 3))
# plot(cvLatRidge2)
# plot(cvLatLasso2)
# plot(cvLatElnet25_2, col="red")
# points(cvLatElnet50_2, col="dark green")
# points(cvLatElnet75_2, col="blue")
# legend("topleft", legend=c("alpha=0.25","alpha=0.5","alpha=0.75"),
#        pch=19, col=c("red","dark green","blue"))

################################################################################################################

# LONGITUDE
# some code tricks from course textbook, p. 202, Listing 8.2, etc.

x <- as.matrix(input[,1:116])
y <- as.matrix(input$V118)
foldid=sample(1:10,size=length(y),replace=TRUE)

# RIDGE regression; all requested parameters are listed on the plot
cvlongRidge <- cv.glmnet(x, y, foldid=foldid, alpha = 0)
r2Ridge <- cvlongRidge$glmnet.fit$dev.ratio[which(cvlongRidge$glmnet.fit$lambda == cvlongRidge$lambda.min)]
mseRidge <- cvlongRidge$cvm[cvlongRidge$lambda == cvlongRidge$lambda.min]

plot(cvlongRidge, main = "RIDGE REGRESSION, LONGITUDE",
     sub = paste("R2 = ", format(r2Ridge, digits=5), ", min. error at lambda = ",
                 format(cvlongRidge$lambda.min, digits=5), " at MSE = ", format(mseRidge, digits=5)))

########################################################################
# LASSO regression; all requested parameters are listed on the plot
cvlongLasso <- cv.glmnet(x, y, foldid=foldid, alpha = 1)
r2Lasso <- cvlongLasso$glmnet.fit$dev.ratio[which(cvlongLasso$glmnet.fit$lambda == cvlongLasso$lambda.min)]
mseLasso <- cvlongLasso$cvm[cvlongLasso$lambda == cvlongLasso$lambda.min]

# number of variables used by the regression according to http://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
cvlongLasso.coeffs <- coef(cvlongLasso, s = cvlongLasso$lambda.min)
numVari = length(cvlongLasso.coeffs@i)                                   # "@i' - after examining 'str(cvlongLasso.coeffs)'

plot(cvlongLasso, main = "LASSO REGRESSION, LONGITUDE",
     sub = paste("R2=", format(r2Lasso, digits=5), ", min.error lam.=", format(cvlongLasso$lambda.min, digits=5),
                 " at MSE=", format(mseLasso, digits=5), ", # var=", numVari))

########################################################################
# ELASTIC NET regression; all requested parameters are listed on the plot

cvlongElnet25=cv.glmnet(x,y,foldid=foldid,alpha=.25)
cvlongElnet50=cv.glmnet(x,y,foldid=foldid,alpha=.5)
cvlongElnet75=cv.glmnet(x,y,foldid=foldid,alpha=.75)

# estimate R2s, lambdas, MSE, number of variables for each of the three case
r2Elnet25 <- format(cvlongElnet25$glmnet.fit$dev.ratio[which(cvlongElnet25$glmnet.fit$lambda == cvlongElnet25$lambda.min)], digits=3)
r2Elnet50 <- format(cvlongElnet50$glmnet.fit$dev.ratio[which(cvlongElnet50$glmnet.fit$lambda == cvlongElnet50$lambda.min)], digits=3)
r2Elnet75 <- format(cvlongElnet75$glmnet.fit$dev.ratio[which(cvlongElnet75$glmnet.fit$lambda == cvlongElnet75$lambda.min)], digits=3)

lamElnet25 <- format(cvlongElnet25$lambda.min, digits=5)
lamElnet50 <- format(cvlongElnet50$lambda.min, digits=5)
lamElnet75 <- format(cvlongElnet75$lambda.min, digits=5)

mseElnet25 <- format(cvlongElnet25$cvm[cvlongElnet25$lambda == cvlongElnet25$lambda.min], digits=5)
mseElnet50 <- format(cvlongElnet50$cvm[cvlongElnet50$lambda == cvlongElnet50$lambda.min], digits=5)
mseElnet75 <- format(cvlongElnet75$cvm[cvlongElnet75$lambda == cvlongElnet75$lambda.min], digits=5)

cvlongElnet25.coeffs <- coef(cvlongElnet25, s = cvlongElnet25$lambda.min)
numVarElnet25 = length(cvlongElnet25.coeffs@i)
cvlongElnet50.coeffs <- coef(cvlongElnet50, s = cvlongElnet50$lambda.min)
numVarElnet50 = length(cvlongElnet50.coeffs@i)
cvlongElnet75.coeffs <- coef(cvlongElnet75, s = cvlongElnet75$lambda.min)
numVarElnet75 = length(cvlongElnet75.coeffs@i)

plot(log(cvlongElnet25$lambda), cvlongElnet25$cvm, pch=19, col="red",
     xlab="", ylab=cvlongElnet25$name, main = "ELASTIC NET, LONGITUDE (log lambda)",
     sub = paste("R2s=", r2Elnet25, ",", r2Elnet50, ",", r2Elnet75,
                 ", min.error lam.=", lamElnet25, ",", lamElnet50, ",", lamElnet75, ",",
                 "\n MSE=", mseElnet25, ",", mseElnet50, ",", mseElnet75,
                 ", # var=", numVarElnet25, ",", numVarElnet50, ",", numVarElnet75))
points(log(cvlongElnet50$lambda), cvlongElnet50$cvm, pch=19, col="dark green")
points(log(cvlongElnet75$lambda), cvlongElnet75$cvm, pch=19, col="blue")
legend("topleft", legend=c("alpha=0.25","alpha=0.5","alpha=0.75"),
       pch=19, col=c("red","dark green","blue"))

# TODO: misclassification error using family = "multinomial", type.measure = "class"
# cvLongRidge2 <- cv.glmnet(x, y, foldid=foldid, alpha = 0, family = "multinomial", type.measure = "class")
# cvLongLasso2 <- cv.glmnet(x, y, foldid=foldid, alpha = 1, family = "multinomial", type.measure = "class")
# cvLongElnet25_2 <- cv.glmnet(x, y, foldid=foldid, alpha=.25, family = "multinomial", type.measure = "class")
# cvLongElnet50_2 <- cv.glmnet(x, y, foldid=foldid, alpha=.50, family = "multinomial", type.measure = "class")
# cvLongElnet75_2 <- cv.glmnet(x, y, foldid=foldid, alpha=.75, family = "multinomial", type.measure = "class")
# par (mfrow=c(1, 3))
# plot(cvLongRidge2)
# plot(cvLongLasso2)
# plot(cvLongElnet25_2, col="red")
# points(cvLongElnet50_2, col="dark green")
# points(cvLongElnet75_2, col="blue")
# legend("topleft", legend=c("alpha=0.25","alpha=0.5","alpha=0.75"),
#        pch=19, col=c("red","dark green","blue"))


# TODO: a nice useful feature to get city from coordinates (can be used in conjunction
# for other types of similar data manipulation, e.g. when the latitude and longitude are not independent of each other):
# hint from https://stackoverflow.com/questions/42319993/listing-cities-from-coordinates-in-r

#library(ggmap)

#coords <- input[,117:118]
#coords <- coords[1:20,]
#colnames(coords) <- c('lat', 'lon')
#res <- lapply(with(coords, paste(lat, lon, sep = ",")), geocode, output = "more")
#transform(coords, city = sapply(res, "[[", "locality"))
