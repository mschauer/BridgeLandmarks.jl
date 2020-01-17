PARESTIMATION <- TRUE

# get directory of source script
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

widthfig <- 7.5

library(tidyverse)
library(stringr)
library(gridExtra)
library(ggforce)
library(GGally)
theme_set(theme_light())
########  read observations


  
v0 <- read_delim("obs0.csv", ";", escape_double = FALSE, trim_ws = TRUE) %>% 
  mutate(landmarkid=as.factor(landmark)) %>% mutate(time=0.0)
vT <- read_delim("obsT.csv", ";", escape_double = FALSE, trim_ws = TRUE) %>% mutate(landmarkid=as.factor(landmark)) %>%
    spread(key=pos,value=value)  %>% dplyr::select(-shape) %>% mutate(time=1.0)
n <- max(v0$landmark)
  
#######  read noisefields
nfsdf <- read_delim("noisefields.csv",";", escape_double = FALSE, trim_ws = TRUE)
  
######## read and analyse bridge iterates
d <- read_table2("iterates.csv") %>%
        mutate(landmarkid=as.factor(landmarkid)) %>%
        gather(key="iteratenr",value="value",contains("iter")) 
d <- d %>% mutate(i = rep(1:(nrow(d)/2),each=2)) # add column to identify a particular combination (pos1,pos2,mom1,mom2) of one landmark at a specific time of a specific shape
d <- d %>% spread(key=pqtype, value=value) %>% dplyr::select(-i) %>%  # can drop identifier column
   mutate(iteratenr=str_replace(iteratenr,"iter",""),iterate=as.numeric(str_replace(iteratenr,"iter","")))
d

####### read parameter updates
parsdf <- read_delim("parameters.csv", ";", escape_double = FALSE, trim_ws = TRUE) 


# plot paths of landmarks positions and bridges over various iterations
p1 <- d %>% dplyr::filter(iteratenr %in% c("0","10","20")) %>%
      ggplot(aes(x=time,y=pos1)) + 
      geom_path(aes(group=interaction(landmarkid,iteratenr),colour=time)) + facet_wrap(~iteratenr) +
      geom_point(data=v0, aes(x=time,y=pos1), colour='black',size=0.5)+
      geom_point(data=vT, aes(x=time,y=pos1), colour='orange',size=0.5) +
      theme(axis.title.x=element_blank(), axis.title.y=element_blank()) 
p1
pdf("bridges_faceted_fewiterates.pdf",width=widthfig,height=4)  
show(p1)
dev.off()



p2 <- d %>% ggplot() + geom_path(aes(x=time,y=pos1, group=iterate,colour=iterate)) +
      facet_wrap(~landmarkid) +
      geom_point(data=v0, aes(x=time,y=pos1), colour='black',size=0.8)+
      geom_point(data=vT, aes(x=time,y=pos1), colour='orange',size=0.8) 
p2
pdf("bridges_overlaid.pdf",width=widthfig,height=4)  
show(p2)
dev.off()



  
if (FALSE) {
# plot parameter updates
ppar1 <- parsdf %>% mutate(cdivgamma2=c/gamma^2) %>% gather(key=par, value=value, a, c, gamma,cdivgamma2) 
ppar1$par <- factor(ppar1$par, levels=c('a', 'c', 'gamma','cdivgamma2'), labels=c("a","c",expression(gamma),expression(c/gamma^2)))
tracepars <- ppar1 %>% ggplot(aes(x=iterate, y=value)) + geom_path() + facet_wrap(~par, scales="free_y",labeller = label_parsed) +
 xlab("iterate") + ylab("") +  theme(strip.text.x = element_text(size = 12))
pdf("trace-pars.pdf",width=6,height=4)  
  show(tracepars)
dev.off()
  
# pairwise scatter plots for parameter updates  
ppar2 <- parsdf %>% ggplot(aes(x=a,y=c,colour=iterate)) + geom_point() + theme(legend.position = 'none')  +scale_colour_gradient(low="orange",high="darkblue")
ppar3 <- parsdf %>% ggplot(aes(x=a,y=gamma,colour=iterate)) + geom_point() + theme(legend.position = 'none') +scale_colour_gradient(low="orange",high="darkblue")
ppar4 <- parsdf %>% ggplot(aes(x=c,y=gamma,colour=iterate)) + geom_point()+ theme(legend.position = 'none') +scale_colour_gradient(low="orange",high="darkblue")
pdf("scatter-pars.pdf",width=6,height=2)  
  grid.arrange(ppar2,ppar3,ppar4,ncol=3)
dev.off()
}  
  

