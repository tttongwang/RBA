
library(readxl)

setwd('C:\\Users\\tongw\\OneDrive - Temple University\\Research\\MAB\\code\\R')

source('theme_setting.r')

png("uneq_lingd_c2.png", units="in", width=5, height=3, res=300)


uneq_lingd_bias        <- read_excel("uneq_lingd_bias_c2.xlsx")
names(uneq_lingd_bias) <- c('beta', 
                           'contextual_bandits', 
                           'RBA', 
                           'suggested_W', 
                           'controlled_std_W')








p_gd_bias <- ggplot(data=uneq_lingd_bias, aes(x=beta))                       + 
  geom_line(aes(y=contextual_bandits, 
            color='contextual_bandits'))                                   +                                       
  geom_line(aes(y=RBA, 
            color='RBA'))                                                 + 
  geom_line(aes(y=suggested_W, 
            color='suggested_W'))                                          + 
  geom_line(aes(y=controlled_std_W,
            color='controlled_std_W'))                                     + 
  ylim(-0.1,0.1)                                                           + 
  geom_hline(yintercept=0, 
             linetype = "dashed")                                          +
  labs(x='c', 
       y= expression(paste('Bias of estimating',beta[1](1))) )                       +
  scale_colour_manual("",
                      breaks = names(uneq_lingd_bias)[-1],
                      values = colors)






uneq_lingd_std         <- read_excel("uneq_lingd_std_c2.xlsx")
names(uneq_lingd_std) <- c('beta', 
                           'contextual_bandits', 
                           'RBA', 
                           'suggested_W', 
                           'controlled_std_W')



p_gd_std <- ggplot(data=uneq_lingd_std, aes(x=beta))                             + 
  geom_line(aes(y=contextual_bandits, 
                color='contextual_bandits'))                                   +                                       
  geom_line(aes(y=RBA, 
                color='RBA'))                                                 + 
  geom_line(aes(y=suggested_W, 
                color='suggested_W'))                                          + 
  geom_line(aes(y=controlled_std_W,
                color='controlled_std_W'))                                     + 
  ylim(0,1.5)                                                                  + 
  labs(x='c', 
       y=expression(paste('Std of estimating',beta[1](1))) )                   +
  scale_colour_manual("",
                      breaks = names(uneq_lingd_std)[-1],
                      values = colors)





grid_arrange_shared_legend(p_gd_bias,
                           p_gd_std,
                           nrow=1, ncol=2)


dev.off()
