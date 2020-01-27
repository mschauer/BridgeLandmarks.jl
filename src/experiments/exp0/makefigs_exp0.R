# get directory of source script
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

widthfig <- 7.5

library(ggpubr)
library(tidyverse)
library(stringr)
library(gridExtra)
library(ggforce)
library(GGally)
theme_set(theme_light())
########  read observations
obs0df <- read_delim("obs0.csv", ";", escape_double = FALSE, trim_ws = TRUE)
obsTdf <- read_delim("obsT.csv", ";", escape_double = FALSE, trim_ws = TRUE) %>% spread(key=pos,value=value) %>%
  mutate(shape=as.factor(shape))


v0 <- bind_rows(obs0df, obs0df)
vT <- bind_rows(obsTdf, obsTdf)
vT1 <- vT %>% dplyr::filter(shape=="1")
n <- max(v0$landmark)

landmarkid_subset <- c(1,3)#seq(1,6,by=1)
landmarkid_subset2 <- seq(1,3, by=2)

d1 <- d %>% dplyr::filter(shapes==1)

# dsub <- d1 %>% dplyr::filter(iterate %in% c(0,50,100)) %>%
#   mutate(iteratenr = fct_relevel(iteratenr, c("0","50","100")))
dlabel0 <- obs0df %>% mutate(landmarkid=as.factor(landmark))
dlabelT <- obsTdf %>% mutate(landmarkid=as.factor(landmark))

#######  read noisefields
nfsdf <- read_delim("noisefields.csv",";", escape_double = FALSE, trim_ws = TRUE)

dforw <- read_table2("forward/iterates.csv") %>%
  mutate(landmarkid=as.factor(landmarkid)) %>%
  gather(key="iteratenr",value="value",contains("iter")) 
dforw <- dforw %>% mutate(i = rep(1:(nrow(dforw)/4),each=4)) # add column to identify a particular combination (pos1,pos2,mom1,mom2) of one landmark at a specific time of a specific shape
dforw <- dforw %>% spread(key=pqtype, value=value) %>% dplyr::select(-i) %>%  # can drop identifier column
  mutate(iterate=as.numeric(str_replace(iteratenr,"iter","")),iteratenr=str_replace(iteratenr,"iter1","forw")) 
  



d <- read_table2("iterates.csv") %>%
  mutate(landmarkid=as.factor(landmarkid)) %>%
  gather(key="iteratenr",value="value",contains("iter")) 
d <- d %>% mutate(i = rep(1:(nrow(d)/4),each=4)) # add column to identify a particular combination (pos1,pos2,mom1,mom2) of one landmark at a specific time of a specific shape
d <- d %>% spread(key=pqtype, value=value) %>% dplyr::select(-i) %>%  # can drop identifier column
  mutate(iteratenr=str_replace(iteratenr,"iter",""),iterate=as.numeric(str_replace(iteratenr,"iter","")))


ds <-  d %>% filter(iterate %in% c(0, 200))
dcomb <- bind_rows(ds, dforw) %>% mutate(iteratenr=fct_relevel(iteratenr,"forw", "0","25", "50", "100", "150","200"))

p1 <- dcomb %>% 
  ggplot() + 
  geom_path(aes(pos1,y=pos2,group=interaction(landmarkid,iteratenr),colour=time),size=1.0) +
    geom_point(data=v0, aes(x=pos1,y=pos2), colour='black')+geom_point(data=vT, aes(x=pos1,y=pos2), colour='orange')+
  geom_path(data=v0, aes(x=pos1,y=pos2), colour='black',size=1.0)+geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange',size=1.0) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  geom_label(data=filter(dlabel0, landmarkid %in% landmarkid_subset), aes(x=pos1,y=pos2,label=landmarkid,hjust="outward",vjust="outward"))+
  geom_label(data=filter(dlabelT, landmarkid %in% landmarkid_subset), aes(x=pos1,y=pos2,label=landmarkid,hjust="outward",vjust="outward"),colour='orange')+
  geom_point(data=nfsdf, aes(x=locx, y=locy), color="Grey")+
  geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = nfsdf,color="Grey",linetype="dotted")+ 
  coord_fixed(xlim = c(-2.3,2.5), ylim = c(-2.1,2))+
  theme(legend.position='none')+ facet_wrap(~iteratenr,ncol=3) 
  
p1

p2 <- dcomb %>% dplyr::filter(landmarkid %in% landmarkid_subset2) %>%
  dplyr::select(-iterate) %>% mutate(iterate=iteratenr) %>% ggplot() + 
  geom_path(aes(mom1,y=mom2,group=interaction(landmarkid,iterate),colour=iterate),size=0.5) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  facet_wrap(~landmarkid)
p2


p3 <-  d %>% dplyr::filter(landmarkid %in% landmarkid_subset2) %>%
  dplyr::filter(time==0) %>% ggplot() + geom_point(aes(mom1,y=mom2,colour=iterate)) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  facet_wrap(~landmarkid) +geom_hline(yintercept=0, linetype="dashed")+geom_vline(xintercept=0, linetype="dashed")+
  scale_colour_gradient(low="grey",high="darkblue")
p3

pdf('landmarks.pdf',width=widthfig,height=4)
 show(p1)
dev.off()
pdf("momenta.pdf",width=widthfig,height=4)  
  grid.arrange(p2,p3,ncol=1)
dev.off()


accdf <- read_delim("accdf.csv", ";", escape_double = FALSE, trim_ws = TRUE)
accfig <- accdf %>% mutate(acc=as.factor(acc)) %>% 
  mutate(kernel=recode(kernel, mala_mom="MALA momenta" ))  %>%
  ggplot(aes(x=iter,y=acc)) +geom_point(shape=124)+ 
  facet_wrap(~kernel)+xlab("iteration nr")+ylab("accept")
pdf("acceptance.pdf",width=widthfig,height=1.5)  
show(accfig)
dev.off()
accfig

################ shape evolution plots

p3 <- dcomb %>% 
  ggplot() + 
  geom_path(aes(pos1,y=pos2,colour=time),size=1.0) +
  geom_point(data=vT, aes(x=pos1,y=pos2), colour='orange') +
  geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange',size=1.0) +
  geom_point(data=v0, aes(x=pos1,y=pos2), colour='black') +
  geom_path(data=v0, aes(x=pos1,y=pos2), colour='black',size=1.0)+
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  geom_label(data=filter(dlabel0, landmarkid %in% landmarkid_subset), aes(x=pos1,y=pos2,label=landmarkid,hjust="outward",vjust="outward"))+
  geom_label(data=filter(dlabelT, landmarkid %in% landmarkid_subset), aes(x=pos1,y=pos2,label=landmarkid,hjust="outward",vjust="outward"),colour='orange')+
  geom_point(data=nfsdf, aes(x=locx, y=locy), color="Grey")+
  geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = nfsdf,color="Grey",linetype="dotted")+ 
  coord_fixed(xlim = c(-2.3,2.5), ylim = c(-2.1,2))+
  theme(legend.position='none')+ facet_wrap(~iteratenr,ncol=3) 

pdf('landmarks_evolution.pdf',width=widthfig,height=4)
show(p3)
dev.off()


### animation

panim <- dcomb %>% 
  ggplot() + 
  geom_path(aes(pos1,y=pos2,colour=time),size=1.0) +
  geom_point(data=vT, aes(x=pos1,y=pos2), colour='orange') +
  geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange',size=1.0) +
  geom_point(data=v0, aes(x=pos1,y=pos2), colour='black') +
  geom_path(data=v0, aes(x=pos1,y=pos2), colour='black',size=1.0)+
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
    #geom_point(data=nfsdf, aes(x=locx, y=locy), color="Grey")+
  #geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = nfsdf,color="Grey",linetype="dotted")+ 
  coord_fixed(xlim = c(-2.3,2.5), ylim = c(-2.1,2))+
  theme(legend.position='none')+ facet_wrap(~iteratenr,ncol=3) 

theme_set(theme_bw())
u <- panim + transition_reveal(time)
animate(u, renderer = ffmpeg_renderer())
anim_save("animate_introexample.mp4", u, renderer = ffmpeg_renderer())
