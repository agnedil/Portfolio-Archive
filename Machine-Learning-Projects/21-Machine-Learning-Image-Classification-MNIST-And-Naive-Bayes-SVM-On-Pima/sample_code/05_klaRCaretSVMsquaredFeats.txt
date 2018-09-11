setwd('~/Current/Courses/LearningCourse/Pima')
rm(list=ls())
wdat<-read.csv('data.txt', header=FALSE)
library(klaR)
library(caret)
bigx<-wdat[,-c(9)]
bigx2<-apply(bigx, c(1, 2), function(x)x^2)
bigx<-cbind(bigx, bigx2)
errs<-array(dim=3)
cvs<-c(0.005, 0.01, 0.1)
for (wi in c(1, 2, 3))
  {bigy<-as.factor(wdat[,9])
  wtd<-createDataPartition(y=bigy, p=.8, list=FALSE)
  wstring<-paste("-c", sprintf('%f', cvs[wi]), sep=" ")
  svm<-svmlight(bigx[wtd,], bigy[wtd], pathsvm='/Users/daf/Downloads/svm_light_osx.8.4_i7/', svm.options=wstring)
  labels<-predict(svm, bigx[-wtd,])
foo<-labels$class
errs[wi]<-sum(foo==bigy[-wtd])/(sum(foo==bigy[-wtd])+sum(!(foo==bigy[-wtd])))
}
