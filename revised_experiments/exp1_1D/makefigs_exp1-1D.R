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
dlabel0 <- v0 %>% mutate(iteratenr=factor(rep("50",n), levels=c('0','50','100')))
dlabelT <- vT %>% mutate(iteratenr=factor(rep("50",n), levels=c('0','50','100')))

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
dd <-d %>% dplyr::filter(iteratenr %in% c("0","50","100")) %>% 
  mutate(iteratenr=factor(iteratenr,levels=c("0","50","100")))
p1 <-  dd %>%
            ggplot(aes(x=time,y=pos1)) + 
      geom_path(aes(group=interaction(landmarkid,iteratenr),colour=time))  +
      theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  geom_point(data=v0, aes(x=time,y=pos1), colour='black',size=1.0)+
  geom_point(data=vT, aes(x=time,y=pos1), colour='orange',size=1.0) +
  geom_label(data=dlabel0, aes(x=time,y=pos1,label=landmarkid))+
  geom_label(data=dlabelT, aes(x=time,y=pos1,label=landmarkid),colour="orange")+
  facet_wrap(~iteratenr) 

p1
pdf("bridges_faceted_fewiterates.pdf",width=widthfig,height=2.5)  
show(p1)
dev.off()



p2 <- d %>% ggplot() + geom_path(aes(x=time,y=pos1, group=iterate,colour=iterate)) +
      facet_wrap(~landmarkid) +
  scale_colour_gradient(low="grey",high="darkblue")+ 
      geom_point(data=v0, aes(x=time,y=pos1), colour='black',size=2)+
      geom_point(data=vT, aes(x=time,y=pos1), colour='orange',size=2) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) 
p2
pdf("bridges_overlaid.pdf",width=widthfig,height=2.5)  
show(p2)
dev.off()


accdf <- read_delim("accdf.csv", ";", escape_double = FALSE, trim_ws = TRUE)

accfig <- accdf %>% gather(key='kernel',value='acc',-iteration) %>%
  ggplot(aes(x=iteration,y=acc)) +
  geom_point(shape=124)+ 
  facet_wrap(~kernel)+xlab("iteration nr")+ylab("accept")

accfig

pdf("acceptance.pdf",width=widthfig,height=4)  
show(accfig)
dev.off()
