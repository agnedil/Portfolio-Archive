# system("wget http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data")
#reading data
library(devtools)
source_url("https://raw.githubusercontent.com/ggrothendieck/gsubfn/master/R/list.R")
d <- read.csv('./pima-indians-diabetes.data', sep=",", header=T)
dx <- d[, -c(9)]
dy <- d[, 9]

library('caret')
# preparing data with 80% traing and 20% testing
inTrain <- createDataPartition(dy, p=0.8, list=F)

trainData <- d[inTrain,]
testData <- d[-inTrain,]

trainX <- trainData[,-c(9)]
trainY <- trainData[,9]
testX <- testData[,-c(9)]
testY <- testData[,9]

# using gaussian distribution so that we know the best

trainGaussian <- function(trainX, trainY){
    posFilter <- trainY == 1
    trainMuP <- sapply(trainX[posFilter,], mean, na.rm=T)
    trainMuN <- sapply(trainX[!posFilter,], mean, na.rm=T)
    trainSigmaP <- sapply(trainX[posFilter,], sd, na.rm=T)
    trainSigmaN <- sapply(trainX[!posFilter,], sd, na.rm=T)
    return(list(trainMuP, trainMuN, trainSigmaP, trainSigmaN))
}

perdictGaussian <- function(x, muP, muN, sigmaP, sigmaN){
    offsetP <- t(t(x)-muP) #x-mu
    offsetN <- t(t(x)-muN)
    scaleP <- t(t(offsetP) / sigmaP) # (x-mu)/sigma
    scaleN <- t(t(offsetN) / sigmaN)
    logP <- -0.5*rowSums(apply(scaleP, c(1,2), square), na.rm=T)-sum(log(sigmaP))
    # (x-mu)^2/2sigma^2-log(simga)
    logN <- -0.5*rowSums(apply(scaleN, c(1,2), square), na.rm=T)-sum(log(sigmaN))
    return(logP>logN)
}

correctPercent <- function(x, y) sum(x==y)/length(x)

correctness1 <- array(dim=10)

# problem1
for (wi in 1:10){
    inTrain <- createDataPartition(y=dy, p=0.8, list=F)

    trainData <- d[inTrain,]
    testData <- d[-inTrain,]
    trainX <- trainData[,-c(9)]
    trainY <- trainData[,9]
    testX <- testData[,-c(9)]
    testY <- testData[,9]

    list[meanP, meanN, sdP, sdN] <- trainGaussian(trainX, trainY)
    correctness1[wi] <- correctPercent(perdictGaussian(testX, meanP, meanN, sdP, sdN), testY)
}

correctness2 <- array(dim=10)
# problem2
for (wi in 1:10){
    inTrain <- createDataPartition(y=dy, p=0.8, list=F)

    trainData <- d[inTrain,]
    testData <- d[-inTrain,]
    for (column in c(3,5,6,8)){
        trainData[trainData[,column]==0, column] <- NA
        testData[testData[,column]==0, column] <- NA
    }
    trainX <- trainData[,-c(9)]
    trainY <- trainData[,9]
    testX <- testData[,-c(9)]
    testY <- testData[,9]

    list[meanP, meanN, sdP, sdN] <- trainGaussian(trainX, trainY)
    correctness2[wi] <- correctPercent(perdictGaussian(testX, meanP, meanN, sdP, sdN), testY)
}

#problem3
library('klaR')

correctness3 <- array(dim=10)
for (wi in 1:10){

    inTrain <- createDataPartition(y=dy, p=.8, list=F)
    trainData <- d[inTrain,]
    testData <- d[-inTrain,]
    trainX <- trainData[,-c(9)]
    trainY <- as.factor(trainData[, 9])
    testX <- testData[,-c(9)]
    testY <- as.factor(testData[,9])

    model <- train(trainX, trainY, 'nb', trControl=trainControl(method='cv', number=10))
    prediction <- predict(model, newdata=testX)
    correctness3[wi] = correctPercent(prediction, testY)
    # confusionMatrix(data=prediction, testY)
}

#problem4
correctness4 <- array(dim=10)
for (wi in 1:10){

    inTrain <- createDataPartition(y=dy, p=.8, list=F)
    trainData <- d[inTrain,]
    testData <- d[-inTrain,]
    trainX <- trainData[,-c(9)]
    trainY <- trainData[, 9]
    testX <- testData[,-c(9)]
    testY <- testData[,9]

    svm <- svmlight(trainX, trainY, pathsvm='./svm_light_osx.8.4_i7')
    labels <- predict(svm, testX)
    correctness4[wi] <- correctPercent(labels$class, testY)
}

