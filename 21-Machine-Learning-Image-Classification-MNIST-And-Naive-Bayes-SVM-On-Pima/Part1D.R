#AML UIUC


library(klaR)
library(caret)

#read data
setwd('/home/andrew/Documents/')
sink(file = 'log.txt', append = TRUE, type = c("output"))                                  #write output to log.txt
rm(list=ls())                                                                              #cleaning all objects from R memory
rawdata<-read.csv('pima-indians-diabetes.data', header=FALSE)

#set features, labels (see 1A for description)
featuresx<-rawdata[,-c(9)]                                                                 #matrix of features
labelsy<-as.factor(rawdata[,9])                                                            #labels from column 9 (0=tested negative, 1=tested positive)

#create training data set
traindata<-createDataPartition(y=labelsy, p=.8, list=FALSE)                                #80%/20% train-test split

#train SVM using SVMLight
svmModel<-svmlight(featuresx[traindata,], labelsy[traindata], pathsvm='svm_light')

#apply trained model to testing data set
testSetResults<-predict(svmModel, featuresx[-traindata,])                                   #output of predict: classification - list with elements ‘class’ and ‘posterior’ (scaled, if scal = TRUE)
                                                                                           #regression - predicted values
predictLabels<-testSetResults$class                                                         #predicted labels
gotright<-predictLabels==labelsy[-traindata]                                               #compare with true labels
accuracy<-sum(gotright)/(sum(gotright)+sum(!gotright))                                     #SVM accuracy on testing set

cat("HW1. Part 1D. SVMLight (simple)\n")
cat(sprintf("SVMLight accuracy: %s\n\n", format(accuracy, digits=5)))
