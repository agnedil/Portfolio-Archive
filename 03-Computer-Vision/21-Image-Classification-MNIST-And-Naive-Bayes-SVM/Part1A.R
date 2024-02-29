#AML UIUC

library(caret)

#read data
setwd('/home/andrew/Documents/')
sink(file = 'log.txt', append = TRUE, type = c("output"))                                  #write output to log.txt
rawdata<-read.csv('pima-indians-diabetes.data', header=FALSE)

#set features, labels
featuresx<-rawdata[,-c(9)]                            #matrix of features (columns 1=#pregnancies 2=2-hr glucose tolerance
                                                      #3=Diastolic blood pressure (mmHg) 4=triceps skin thickness(mm)
                                                      #5=2-hr serum insulin(mu U/ml) 6=BMI(kg/m^2
                                                      #7=diabetes pedigree f(x) 8=age (yr))
labelsy<-rawdata[,9]                                  #labels from column 9 (0=tested negative, 1=tested positive)

#initialize arrays to store training and testing accuracies for each of the 10 folds
trainScore<-array(dim=10)
testScore<-array(dim=10)

#Naive Bayes with averaging over 10 test-train splits and ignoring examples with NAs
for (i in 1:10) {
  
  #create training and testing sets
  traindata<-createDataPartition(y=labelsy, p=.8, list=FALSE)           #80%/20% split
  #nbx<-featuresx                                                       #rename matrix of features and use nbx further instead of features
  trainfeaturesx <- featuresx[traindata, ]                              #training features
  trainlabelsy   <-   labelsy[traindata]                                #training labels (true)
  testfeaturesx  <- featuresx[-traindata, ]                             #testing features
  testlabelsy    <-   labelsy[-traindata]                               #testing labels (true)
  
  #calculate priors for each class
  negPrior<- length(which(trainlabelsy == 0))/length(trainlabelsy)
  posPrior<- length(which(trainlabelsy == 1))/length(trainlabelsy)
  
  #split + and - cases
  trainPosFlag<-trainlabelsy>0                                          #diabetes-positive training labels
  trainPosRows<-trainfeaturesx[trainPosFlag, ]                          #diabetes-positive training rows
  trainNegRows<-trainfeaturesx[!trainPosFlag,]                          #diabetes-negative training rows
  
  #means and sds
  trainPosMeans<-sapply(trainPosRows, mean, na.rm=TRUE)                 #vector of means for training, diabetes-positive
  trainNegMeans<-sapply(trainNegRows, mean, na.rm=TRUE)                 #vector of means for training, diabetes-negative
  trainPosSd  <-sapply(trainPosRows, sd, na.rm=TRUE)                    #vector of sd for training, diabetes-positive
  trainNegSd  <-sapply(trainNegRows, sd, na.rm=TRUE)                    #vector of sd for training, diabetes-negative
  
  #scaling and log likelihood for training sets
  trainPosOffsets<-t(t(trainfeaturesx)-trainPosMeans)                   #first step normalize training diab-pos, subtract mean
  trainPosScales<-t(t(trainPosOffsets)/trainPosSd)                      #second step normalize training diab-pos, divide by sd
  trainPosLogs<-(-(1/2)*rowSums(apply(trainPosScales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(trainPosSd))) + log(posPrior)    #log likelihoods based on normal distrib. for diab-pos (each class separately)
  trainNegOffsets<-t(t(trainfeaturesx)-trainNegMeans)
  trainNegScales<-t(t(trainNegOffsets)/trainNegSd)
  trainNegLogs<-(-(1/2)*rowSums(apply(trainNegScales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(trainNegSd))) + log(negPrior)    #log likelihoods based on normal distrib. for diab-neg (both log-likel. account for priors)
  lvwTrain<-trainPosLogs>trainNegLogs                                   #rows classified as diab-pos by classifier
  gotrightTrain<-lvwTrain==trainlabelsy                                 #compare with true labels
  trainScore[i]<-sum(gotrightTrain)/(sum(gotrightTrain)+sum(!gotrightTrain))                                      #classifier accuracy on training set
  
  #scaling and log likelihood for test set with parameters from training for diab-pos class
  testPosOffsets<-t(t(testfeaturesx)-trainPosMeans) 
  testPosScales<-t(t(testPosOffsets)/trainPosSd)
  testPosLogs<-(-(1/2)*rowSums(apply(testPosScales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(trainPosSd))) + log(posPrior)
  testNegOffsets<-t(t(testfeaturesx)-trainNegMeans)                     #same for diab-neg class
  testNegScales<-t(t(testNegOffsets)/trainNegSd)
  testNegLogs<-(-(1/2)*rowSums(apply(testNegScales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(trainNegSd))) + log(negPrior)
  lvwTest<-testPosLogs>testNegLogs
  gotright<-lvwTest==testlabelsy
  testScore[i]<-sum(gotright)/(sum(gotright)+sum(!gotright))                                                      #classifier accuracy on testing set
}

#output accuracies
cat("HW1. Part 1A. Naive Bayes Results. NAs Ignored\n")
trainAccuracy <- sum(trainScore) / length(trainScore)
testAccuracy  <- sum(testScore) /  length(testScore)
cat("Training accuracy in each loop: ", format(trainScore, digits=5), "\n")
cat(sprintf("Average 10-loop training accuracy: %s\n", format(trainAccuracy, digits=5), "\n"))
cat("Testing accuracy in each loop: ", format(testScore, digits=5), "\n")
cat(sprintf("Average 10-loop testing accuracy: %s\n\n", format(testAccuracy, digits=5), "\n"))
