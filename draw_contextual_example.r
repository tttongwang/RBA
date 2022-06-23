
library(ggplot2)
library(dplyr)


theme_update(panel.grid.major=element_blank(),
             panel.grid.minor=element_blank(),
             panel.background=element_blank(),
             axis.line=element_line(colour='black'),
             plot.title = element_text(hjust = 0.5,size=7,face = "bold"),
             legend.text = element_text(size=7,face = "bold"),
             axis.title=element_text(size=7,face = "bold"),
             axis.text=element_text(size=7,face = "bold")
)

update_geom_defaults("line", list(size=0.7))




setwd('C:\\Users\\tongw\\OneDrive - Temple University\\Research\\MAB\\code\\R')


png("Contextual_example_trend.png", units="in", width=4, height=3, res=300)

data        <- contextual_example_gd
names(data) <- c('beta','Bias')

ggplot(data=data, mapping=aes(x=beta,y=Bias))           + 
  geom_line()                                           + 
  geom_point()                                          + 
  ylim(-0.3,0)                                          + 
  geom_hline(yintercept=0, linetype = "dashed")         +
  labs(x=expression(beta[1]), y='Bias of OLS Estimate')



dev.off()
 




