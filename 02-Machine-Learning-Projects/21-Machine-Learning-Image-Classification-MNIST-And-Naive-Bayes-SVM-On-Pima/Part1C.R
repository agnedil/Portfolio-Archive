#AML UIUC

library(klaR)
library(caret)


#read data
setwd('/home/andrew/Documents/')
sink(file = 'log.txt', append = TRUE, type = c("output"))                        #write output to log.txt
rawdata<-read.csv('pima-indians-diabetes.data', header=FALSE)

#set features, labels (see 1A for description)
featuresx<-rawdata[,-c(9)]                                                       #matrix of features
labelsy<-as.factor(rawdata[,9])                                                  #labels from column 9 (0=tested negative, 1=tested positive)

#create training and testing sets
traindata<-createDataPartition(y=labelsy, p=.8, list=FALSE)                      #80%/20% train-test split
trainfeaturesx<-featuresx[traindata,]                                            #training features
trainlabelsy<-labelsy[traindata]                                                 #training labels
testfeaturesx  <- featuresx[-traindata, ]                                        #testing features
testlabelsy    <-   labelsy[-traindata]                                          #testing labels (true)

#training Naive Bayes with 10-fold cross-validation on training data
model<-train(trainfeaturesx, trainlabelsy, 'nb', trControl=trainControl(method='cv', number=10))

#apply trained model to testing data set
testClasses<-predict(model, newdata=testfeaturesx)
matrix<-confusionMatrix(data=testClasses, testlabelsy)

#output confusion matrix
cat("HW1. Part 1C. Naive Bayes using caret and klaR packages. NAs Ignored\n")
print(matrix)
cat("\n")
