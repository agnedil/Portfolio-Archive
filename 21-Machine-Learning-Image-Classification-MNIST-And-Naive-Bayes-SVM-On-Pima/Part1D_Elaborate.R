# AML UIUC

library(klaR)
library(caret)

#read data
setwd('/home/andrew/Documents/')
sink(file = 'log.txt', append = TRUE, type = c("output"))                                                         #write output to log.txt
rm(list=ls())                                                                                                     #cleaning all objects from R memory
rawdata<-read.csv('pima-indians-diabetes.data', header=FALSE)

#set features, labels (see 1A for description) + squared features
featuresx<-rawdata[,-c(9)]                                                                                        #matrix of features
featuresx2<-apply(featuresx, c(1, 2), function(x)x^2)                                                             #matrix of squared features
featuresx<-cbind(featuresx, featuresx2)                                                                           #combine both matrices by column (sequence = side by side, left to right)
labelsy<-as.factor(rawdata[,9])                                                                                   #labels from column 9 (0=tested negative, 1=tested positive)

#initialize array for 3 accuracies associated with 3 different error penalties
accuracy<-array(dim=3)

#set 3 values for error penalties (trade-offs between maximum margin and classification error during training)
cvs<-c(0.005, 0.01, 0.1)

#run SVMLight for 3 different error penalties
for (wi in c(1, 2, 3))
  {
  traindata<-createDataPartition(y=labelsy, p=.8, list=FALSE)                                                     #80%/20% train-test split
  
  #train SVM using SVMLight
  wstring<-paste("-c", sprintf('%f', cvs[wi]), sep=" ")                                                           #text of parameter for svm.options ("-c float")
  svmmodel<-svmlight(featuresx[traindata,], labelsy[traindata], pathsvm='svm_light', svm.options=wstring)         #wstring = error penalty "-c float" (How to use section at http://svmlight.joachims.org/)
  
  #apply trained model to testing data set
  testSetResults<-predict(svmmodel, featuresx[-traindata,])                                                       #output of predict: classification - list with elements ‘class’ and ‘posterior’ (scaled, if scal = TRUE)
                                                                                                                  #regression - predicted values
  predictLabels<-testSetResults$class                                                                             #predicted labels
  gotright<-predictLabels==labelsy[-traindata]                                                                    #compare with true labels
  accuracy[wi]<-sum(gotright)/(sum(gotright)+sum(!gotright))                                                      #SVM accuracy on testing set
  }
cat("HW1. Part 1D. Naive Bayes using SVMLight (more elaborate with squared features and different error penalties)\n")
cat(sprintf("SVMLight accuracies for error penalties of 0.005, 0.01, 0.1 are %s, %s, %s, respectively\n\n", format(accuracy[1], digits=5), format(accuracy[2], digits=5), format(accuracy[3], digits=5)))
