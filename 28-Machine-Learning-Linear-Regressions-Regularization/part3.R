# AML UIUC

# Linear regression and plots based on a dataset used for predicting the age of abalone (Haliotis rubra)
# from physical measurements: https://archive.ics.uci.edu/ml/datasets/Abalone

# data manipulation hints from http://scg.sdsu.edu/linear-regression-in-r-abalone-dataset/

library(glmnet)

# read data
rm(list = ls())
setwd('/home/andrew/Documents/')

names = c('sex','length','diameter','height','weight_whole','weight_shuck','weight_visc','weight_shell','age')
input <- read.table("data/abalone.data", header = FALSE, sep = ',', col.names = names)

# transform gender (M = -1, F = 0, I = 1, as suggested in the text of the excecise)
#          and age (+1.5 gives the age in years - from dataset's website)
input$sex = as.character(input$sex)
input$sex[input$sex=="I"] <- 1
input$sex[input$sex=="F"] <- 0
input$sex[input$sex=="M"] <- -1
input$sex = as.factor(input$sex)
input$age <- input$age + 1.5

# a) Linear regression predicting the age from measurements, ignoring gender.
# Plot of residual vs. fitted values:
mylm_a <- lm(age ~ length + diameter + height + weight_whole + weight_shuck + weight_visc + weight_shell, data = input)
plot(mylm_a$fitted.values, mylm_a$residuals, xlab = "Fitted Values", ylab = "Residual",
     main = "7.11 a) Residual vs. Fitted Values\n (ignoring gender)",
     sub = paste("R2 = ", format(summary(mylm_a)$r.squared, digits=5)))
abline(h = 0)                                                               # 0 residual line
    
# (b) Linear regression predicting the age from measurements, including gender.
# Three levels for gender (1, 0, -1). Plot of residual vs. fitted values.
mylm_b <- lm(age ~ sex + length + diameter + height + weight_whole + weight_shuck + weight_visc + weight_shell, data = input)
plot(mylm_b$fitted.values, mylm_b$residuals, xlab = "Fitted Values", ylab = "Residual",
     main = "7.11 b) Residual vs. Fitted Values\n (including gender)",
     sub = paste("R2 = ", format(summary(mylm_b)$r.squared, digits=5)))
abline(h = 0)                                                               # 0 residual line

# (c) Linear regression predicting the log of age from measurements, ignoring gender.
# Plot of residual vs. fitted values.
mylm_c <- lm(log(age) ~ length + diameter + height + weight_whole + weight_shuck + weight_visc + weight_shell, data = input)
newResid_c <- input$age - exp(mylm_c$fitted.values)
plot(exp(mylm_c$fitted.values), newResid_c, xlab = "Fitted Values", ylab = "Residual",
     main = "7.11 c) Residual vs. Fitted Values (ignoring gender)\n (log of age in orig. coord.)",
     sub = paste("R2 = ", format(summary(mylm_c)$r.squared, digits=5)))
abline(h = 0)                                                               # 0 residual line               

# (d) Linear regression predicting the log age from the measurements, including gender (same three levels).
# Plot of residual vs. fitted values.
mylm_d <- lm(log(age) ~ sex + length + diameter + height + weight_whole + weight_shuck + weight_visc + weight_shell, data = input)
newResid_d <- input$age - exp(mylm_d$fitted.values)
plot(exp(mylm_d$fitted.values), newResid_d, xlab = "Fitted Values", ylab = "Residual",
     main = "7.11 d) Residual vs. Fitted Values (including gender)\n (log of age in orig. coord.)",
     sub = paste("R2 = ", format(summary(mylm_d)$r.squared, digits=5)))
abline(h = 0)                                                               # 0 residual line             

# e) see the pdf report


# f) use cv.glmnet to build plots of cross-validated prediction error to see if each of the regressions
#    can be improved through the use of a regularizer (4 plots)
fit_a <- cv.glmnet(as.matrix(sapply(input[,-1], as.numeric)), mylm_a$fitted.values, alpha = 0)                # matrix of doubles
plot(fit_a, sub = "CV prediction error for 7.11 a) plot")
fit_b <- cv.glmnet(as.matrix(sapply(input[,-1], as.numeric)), mylm_b$fitted.values, alpha = 0)
plot(fit_b, sub = "CV prediction error for 7.11 b) plot")
fit_c <- cv.glmnet(as.matrix(sapply(input[,-1], as.numeric)), mylm_c$fitted.values, alpha = 0)
plot(fit_c, sub = "CV prediction error for 7.11 c) plot")
fit_d <- cv.glmnet(as.matrix(sapply(input[,-1], as.numeric)), mylm_d$fitted.values, alpha = 0)
plot(fit_d, sub = "CV prediction error for 7.11 d) plot")
