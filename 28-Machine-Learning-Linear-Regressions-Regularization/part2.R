# AML UIUC

# Linear regression and plots based on a dataset containing weight and other physical measurements
# for 22 male subjects aged 16 - 30: http://www.statsci.org/data/oz/physical.html

# read data
rm(list = ls())
setwd('/home/andrew/Documents/')
input <- read.table("data/physical.txt", header = TRUE)

# linear regression: predicting body mass from diameters
mylm <- lm(Mass~Fore+Bicep+Chest+Neck+Shoulder+Waist+Height+Calf+Thigh+Head, data = input)    


# a) plot residual against fitted values
plot(mylm$fitted.values, mylm$residuals, xlab = "Fitted Values", ylab = "Residual",
     main = "Residual vs. Fitted Values\n (original - mass from data)")
abline(h = 0)                                                               # 0 residual line 

# b) regress the cube root of mass against these diameters

crlm <- lm(((Mass)^(1/3))~Fore+Bicep+Chest+Neck+Shoulder+Waist+Height+Calf+Thigh+Head, data = input)

# residual vs. fitted values in cube root coordinates
plot(crlm$fitted.values, crlm$residuals, xlab = "Fitted Values", ylab = "Residual",
     main = "Residual vs. Fitted Values\n (cube-root coordinates)")
abline(h = 0)                                                               # 0 residual line

# residual vs. fitted values in original coordinates
newResid <- input$Mass - (crlm$fitted.values)^3
plot((crlm$fitted.values)^3, newResid, xlab = "Fitted Values", ylab = "Residual",
     main = "Residual vs. Fitted Values\n (orig. coord. from cube root)")
abline(h = 0)                                                               # 0 residual line

# c) see the pdf report
