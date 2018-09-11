# AML UIUC

# Linear regression and plots based on a dataset describing the concentration of a sulfate in the blood of
# a baboon named Brunhilda as a function of time: http://www.statsci.org/data/general/brunhild.html

# multiple hints at https://stackoverflow.com/questions/15274039/using-lm-and-predict-on-data-in-matrices and the like 
# hints at https://stackoverflow.com/questions/29283205/graphing-several-lines-of-same-function-in-r

# read data
rm(list = ls())
setwd('/home/andrew/Documents/')
input <- read.table("data/brunhild.txt", header = TRUE)

# logs of input data
logData <- log(input)

# linear regression: log of concentration against log of time
logLM <- lm(Sulfate ~ Hours, data = logData)                                # y~x = y modelled by a linear predictor x

# a) plot with data points and regression line in log-log coordinates
plot(logData, main = "Sulfate Level vs. Time\n (log-log coordinates)")
abline(logLM)                                                               # add 1 or more straight lines to current plot

# b) plot with data points and regression curve in the original coordinates
plot(input, main = "Sulfate Level vs. Time\n (original coordinates)")
lines(exp(predict(logLM))~input$Hours)                                      # lines (x, y, ...) - joins points with line segments

# c) plot residual vs. fitted values in log-log / original coordinates

# residual vs. fitted values in log-log coordinates
plot(logLM$fitted.values, logLM$residuals, xlab = "Fitted Values", ylab = "Residual",
     main = "Residual vs. Fitted Values\n (log-log coordinates)")
abline(h = 0)                                                               # 0 residual line

# residual vs. fitted values in original coordinates
newResid <- input$Sulfate - exp(logLM$fitted.values)
plot(exp(logLM$fitted.values), newResid, xlab = "Fitted Values", ylab = "Residual",
     main = "Residual vs. Fitted Values\n (original coordinates)")
abline(h = 0)                                                               # 0 residual line

# d) see the pdf report
