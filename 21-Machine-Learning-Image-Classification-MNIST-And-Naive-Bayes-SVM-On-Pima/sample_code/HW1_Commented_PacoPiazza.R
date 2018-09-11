#setwd('~/Current / / / /')
wdat <- read.csv('pima-indians-diabetes.data', header = FALSE)

library(klaR)
library(caret)

bigx <- wdat[, -c(9)] # matrix of features
bigy <- wdat[, 9]     # labels; class value 1 means "tested positive for diabetes"

trscore <- array(dim = 10)
tescore <- array(dim = 10)

for (wi in 1:10) {
  wtd <- createDataPartition(y = bigy, p = 0.8, list = FALSE) # 80% of the data into training
  nbx <- bigx                                 # matrix of features
  ntrbx <- nbx[wtd, ]                         # training features
  ntrby <- bigy[wtd]                          # training labels
  
  trposflag <- ntrby > 0                      # training labels for diabetes positive
  ptregs <- ntrbx[trposflag, ]                # training rows features with diabetes positive
  ntregs <- ntrbx[!trposflag, ]               # training rows features with diabetes negative
  
  ntebx <- nbx[-wtd, ]                        # test rows - features
  nteby <- bigy[-wtd]                         # test rows - labels
  
  ptrmean <- sapply(ptregs, mean, na.rm = T)  # vector of means for training, diabetes positive
  ntrmean <- sapply(ntregs, mean, na.rm = T)  # vector of means for training, diabetes negative
  ptrsd   <- sapply(ptregs, sd, na.rm = T)    # vector of sd for training, diabetes positive
  ntrsd   <- sapply(ntregs, sd, na.rm = T)    # vector of sd for training, diabetes negative
  
  ptroffsets <- t(t(ntrbx) - ptrmean)         # first step normalize training diabetes pos, subtract mean
  ptrscales  <- t(t(ptroffsets) / ptrsd)      # second step normalize training diabetes pos, divide by sd
  ptrlogs    <- -(1/2) * rowSums(apply(ptrscales, c(1,2),
                function(x) x^2), na.rm = T) - sum(log(ptrsd))  # Log likelihoods based on 
								# normal distr. for diabetes positive
  ntroffsets <- t(t(ntrbx) - ntrmean)
  ntrscales  <- t(t(ntroffsets) / ntrsd)
  ntrlogs    <- -(1/2) * rowSums(apply(ntrscales, c(1,2) 
                                       , function(x) x^2), na.rm = T) - sum(log(ntrsd))
                                                                # Log likelihoods based on 
								# normal distr for diabetes negative
                                                                # (It is done separately on each class)
  
  lvwtr      <- ptrlogs > ntrlogs              # Rows classified as diabetes positive by classifier 
  gotrighttr <- lvwtr == ntrby                 # compare with true labels
  trscore[wi]<- sum(gotrighttr)/(sum(gotrighttr)+sum(!gotrighttr)) # Accuracy with training set
  
  pteoffsets <- t(t(ntebx)-ptrmean)            # Normalize test dataset with parameters from training
  ptescales  <- t(t(pteoffsets)/ptrsd)
  ptelogs    <- -(1/2)*rowSums(apply(ptescales,c(1, 2)
                                     , function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
  
  nteoffsets <- t(t(ntebx)-ntrmean)            # Normalize again for diabetes negative class
  ntescales  <- t(t(nteoffsets)/ntrsd)
  ntelogs    <- -(1/2)*rowSums(apply(ntescales,c(1, 2)
                                     , function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
  
  lvwte<-ptelogs>ntelogs
  gotright<-lvwte==nteby
  tescore[wi]<-sum(gotright)/(sum(gotright)+sum(!gotright))  # Accuracy on the test set
}
