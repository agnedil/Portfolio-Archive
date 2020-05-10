library(readr)
setwd('/home/andrew/Documents/2_UIUC/CS598_Data_Mining_Capstone/task2/task2_pycharms_project')
mydata <- read_csv("none.csv", col_names=TRUE, progress = FALSE)

#To avoid alphabetical order of labels on axis X and enforce the exact order from the csv file
#turn your 'Cuisine' column into a character vector
mydata$Cuisine <- as.character(mydata$Cuisine)
#Then turn it back into an ordered factor
mydata$Cuisine <- factor(mydata$Cuisine, levels=unique(mydata$Cuisine))

#View below lets you see the matrix
#View(mydata)
library(reshape2)
melted_cormat <- melt(mydata)
#head below prints the correlation matrix
#head(melted_cormat)

library(ggplot2)
ggplot(data = melted_cormat, aes(x=Cuisine, y=variable, fill=value)) +
  scale_x_discrete(position = "top") +                                #this is to move axis x to the top
  scale_y_discrete(limits = rev(levels(melted_cormat$variable))) +    #this is to reverse the order of labels on axis y
  geom_tile(color = "blue")+
  scale_fill_gradient2(high = "red", low = "light blue", midpoint=0.5,
                         limit = c(0,1), space = "Lab", name="Similarity",
                         breaks = c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 90, vjust = .5,
                                   size = 7, hjust = 0),              #font angle, justification, size on axis x
        axis.text.y = element_text(angle = 0, vjust = .5, 
                                   size = 7, hjust = 1),              #font angle, justification, size on axis y
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.grid.major = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        axis.ticks = element_blank())+
  coord_fixed()
#see the rest of the code needed here:
#http://www.sthda.com/english/wiki/ggplot2-quick-correlation-matrix-heatmap-r-software-and-data-visualization