# AML UIUC
# used this post https://stackoverflow.com/questions/14860078/plot-multiple-lines-data-series-each-with-unique-color-in-r
# used this post https://stackoverflow.com/questions/8166931/plots-with-good-resolution-for-printing-and-screen-display

setwd('/home/andrew/Documents/2_UIUC/3_CS498_AML/Homeworks/HW2/MyHW2')
options(warn=-1)                                                            #neg-ignore, 0-store, 1-print, 2-turn into errors
library(caret)

#create one dataset out of the given two (to make my own random split of the data)
rawdata1 <- read.csv('adult.data', header=FALSE, na.strings = "?", strip.white = TRUE)                #strips spaces from values,
rawdata2 <- read.csv('adult.test', header=FALSE, na.strings = "?", strip.white = TRUE, skip = 1)      #skips the useless first line "|1x3 Cross validator" 
rawdata <- rbind(rawdata1, rawdata2, make.row.names=FALSE)                                            

#set features and labels
#use only continuous attributes as features: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
features <- rawdata[,c(1,3,5,11,12,13)]
labels <- rawdata[,15]

#pre-process features: scale for unit variance
for (i in 1:6)
    {
    features[i] <- scale(as.numeric(as.matrix(features[i])), center = TRUE, scale = TRUE)           #center deducts mean, scale divides by std (by default)
    }

#pre-process labels: set their value to -1 if income is <=50K and to 1 otherwise
labels <- as.character(labels)
for (i in 1:length(labels))
{
    if (labels[i]=="<=50K" | labels[i]=="<=50K.") {labels[i] <- as.numeric(-1)}                     #labels have different format in adult.data and adult.test
    else if (labels[i]==">50K"  | labels[i]==">50K." ) {labels[i] <- as.numeric(1)}
}
labels <- as.factor(labels)

#split data into training, testing, and validation sets
trainData <- createDataPartition(y=labels, p=.8, list=FALSE)
trainx <- features[trainData,]
trainy <- labels[trainData]
intermedx <- features[-trainData,]
intermedy <- labels[-trainData]
testData <- createDataPartition(y=othery, p=.5, list=FALSE)
testx <- intermedx[testData,]
testy <- intermedy[testData]
validatex <- intermedx[-testData,]
validatey <- intermedy[-testData]

#hinge loss function
hingeloss <- function(predicted, true)
    {
    return (max(0, 1 - (predicted * true) ))
    }

#function to classify specific case (x, a, and b)
classify <- function(x, a, b)
    {
    x <- as.numeric(as.matrix(x))
    return (t(a) %*% x + b) 
    }

#function to calculate specific accuracy
accuracy <- function(x,y,a,b)
    {
    gotright <- 0
    for (i in 1:length(y))
        {
        if (classify(x[i,], a, b) >= 0) {predicted <- 1} else {predicted <- -1}
        if (predicted == y[i]) {gotright <- gotright + 1}
        }
    return (gotright/length(y))
    }

#to store accuracies during looping (needed inside loops as a and b will be changing)
validationAccuracies <- c()
testAccuracies <- c()

#store aT*a magnitudes and accuracies on 50 examples held out for the epoch for all lambdas (needed later for plotting) 
accuracies50_all <- data.frame()
magnitudes_all <- data.frame()

#predetermined regularization weights further called lambdas
mylambdas <- c(.001, .01, .1, 1)

#SVM with SGD for each lambda
for (lambda in mylambdas){
  
    #initialize a and b
    a <- c(0,0,0,0,0,0)
    b <- 0
  
    #to store accuracies on the 50 random examples held out for the epoch and magnitudes for each lambda
    accuracies50 <- c()
    magnitudes <- c()
  
    #50 epochs
    for (epoch in 1:50){
    
        #hold out 50 random examples for the epoch to check accuracy every 30 steps
        randomExamples <- sample(1:dim(trainx)[1], 50)
        features50HeldOut <- trainx[randomExamples, ]
        labels50HeldOut <- trainy[randomExamples]
        train_data <- trainx[-randomExamples,]
        train_labels <- trainy[-randomExamples]
    
        #to keep track of the number of steps in each epoch
        stepCounter <- 0
    
        #300 steps in each epoch
        for (step in 1:300){
            
            #Setting up stochastic gradient descent - see Course Textbook, Section 3.1, p. 35
            #random batch Nb, size = 1
            batch <- sample(1:length(train_labels), 1)
            while(is.na( train_labels[batch] ) )
                {
                batch <- sample(1:length(train_labels), 1)
                }
            xk <- as.numeric (as.matrix (train_data[batch,]) )
            if ((train_labels[batch]) == 1) {yk <- 1} else {yk <- -1}                               #TODO: why direct assigment doesn't work here
            
            predictY <- classify(xk, a, b)
            steplength <- 1 / ((.01 * epoch) + 50)                                                  #values for m and n are taken from the Course Textbook
      
            #choosing the gradient - see Course Textbook, Section 3.1, p. 35
            if(yk * predictY >= 1)
                {
                a_n_plus_1 <- lambda * a
                b_n_plus_1 <- 0
                }
            else
                {
                a_n_plus_1 <- (lambda * a) - (yk * xk)
                b_n_plus_1 <- -(yk)
                }
      
            #update estimates of a and b using gradient and steplength (eta) - see Course Textbook, Section 3.1, p. 35
            a <- a - (steplength * a_n_plus_1)
            b <- b - (steplength * b_n_plus_1)
      
            
            if(stepCounter %% 30 == 0)
                {
                #calculate accuracy on the 50 random examples held out for the epoch; done every 30 steps
                accuracy50 <- accuracy(features50HeldOut, labels50HeldOut, a, b)
                accuracies50 <- c(accuracies50, accuracy50)
                
                #calculate magnitude of the coefficient vector aT*a, done every 30 steps
                magnitude <- t(a) %*% a
                magnitudes <- c(magnitudes, magnitude)
                }
            
            stepCounter <- stepCounter + 1
            
        }#for (step in 1:300)
        if (epoch%%10==0) print(epoch)                                                          #debugging
    }#for (epoch in 1:50)
  
    #accuracy on the validation dataset for a particular lambda
    lambdaValidationAccu <- accuracy (validatex,     validatey, a, b)
    validationAccuracies <- c(validationAccuracies, lambdaValidationAccu)
  
    #accuracy on the testing dataset
    testAccuracy <- accuracy(testx, testy, a, b)
    testAccuracies <- c(testAccuracies, testAccuracy)
    
    #accumulate accuracies on the 50 held out examples and magnitudes into one dataframe each
    accuracies50_all <- rbind(accuracies50_all, accuracies50)
    magnitudes_all   <- rbind(magnitudes_all, magnitudes)
    
    cat("Process finished for lambda = ", lambda, "\n")                                         #debugging
}#for (lambda in mylambdas)

#The code below provides answers to the 4 sub-tasks required for this assignment
#A plot of the accuracy on the 50 held out examples every 30 steps, for each value of the regularization constant

jpeg(file="accuracies.jpeg", units="in", width=5.5, height=4.25, res=750)
#empty plot
plot(1, 1, xlim = c(1, length(accuracies50)), ylim = c(.5, 1), xlab = "Steps", ylab = "Held Out Accuracy", main = "Held Out Accuracy for Each Lambda", type = 'n')

#list of 4 colors to use for the lines
cl <- rainbow(4)

#plot accuracy for each lambda with its own color + a legend in the end
plotcol <- c()
for (i in 1:length(mylambdas))
    {
    lines(1:length(accuracies50), accuracies50_all[i,], type = 'l', col = cl[i], lwd = 1, lty = 1)
    plotcol[i] <- cl[i]
    }
legend("bottomright", legend = c(mylambdas), col = plotcol, lwd = 2, cex = .5, title="Lambda Values")
dev.off()

#A plot of the magnitude of the coefficient vector every 30 steps, for each value of the regularization constant

jpeg(file="magnitudes.jpeg", units="in", width=5.5, height=4.25, res=750)
plot(1, 1, xlim = c(1, length(magnitudes)), ylim = c(0, 5), xlab = "Steps", ylab = "Size of aT*a", main = "Magnitude of Coefficient Vector for Each Lambda", type = 'n')
cl <- rainbow(4)
plotcol <- c()
for (i in 1:length(mylambdas))
    {
    lines(1:length(magnitudes), magnitudes_all[i,], type = 'l', col = cl[i], lwd = 1, lty = 1)
    plotcol[i] <- cl[i]
    }
legend("topright", legend = c(mylambdas), col = plotcol, lwd = 2, cex = .5, title="Lambda Values")
dev.off()

#The best value of the regularization parameter lambda: the one that showed the best accuracy on the validation dataset
maxindex <- 1
for(i in 1:length(validationAccuracies))
    {
    if (validationAccuracies[i] >= validationAccuracies[maxindex]) {maxindex <- i}
    }
maxlambda <- mylambdas[maxindex]
cat("The best value of the regularization parameter lambda: ", maxlambda, "\n")

#Accuracy of the best classifier on the 10% test dataset
cat(sprintf("The accuracy of the best classifier on the 10 percent test dataset: %s\n", format(testAccuracies[maxindex], digits=5)))
