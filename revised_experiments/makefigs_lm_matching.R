# get directory of source script
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

widthfig <- 7.5
fracwidthfig = 0.8 * widthfig

library(tidyverse)
library(stringr)
library(gridExtra)
library(ggforce)
library(GGally)
#theme_set(theme_light())
theme_set(theme_bw())

########  read observations
obs0df <- read_delim("obs0.csv", ";", escape_double = FALSE, trim_ws = TRUE)
obsTdf <- read_delim("obsT.csv", ";", escape_double = FALSE, trim_ws = TRUE) %>% spread(key=pos,value=value) %>%
    mutate(shape=as.factor(shape))

v0 <- bind_rows(obs0df, obs0df)
vT <- bind_rows(obsTdf, obsTdf)
vT1 <- vT %>% dplyr::filter(shape=="1")
n <- max(v0$landmark)
  
#######  read noisefields
nfsdf <- read_delim("noisefields.csv",";", escape_double = FALSE, trim_ws = TRUE)

####### read parameter updates
parsdf <- read_delim("parameters.csv", ";", escape_double = FALSE, trim_ws = TRUE) 

###### read acceptance info
accdf <- read_delim("accdf.csv", ";", escape_double = FALSE, trim_ws = TRUE)

######## read and analyse bridge iterates
d <- read_table2("iterates.csv") %>%
        mutate(landmarkid=as.factor(landmarkid)) %>%
        gather(key="iteratenr",value="value",contains("iter")) 
d <- d %>% mutate(i = rep(1:(nrow(d)/4),each=4)) # add column to identify a particular combination (pos1,pos2,mom1,mom2) of one landmark at a specific time of a specific shape
d <- d %>% spread(key=pqtype, value=value) %>% dplyr::select(-i) %>%  # can drop identifier column
   mutate(iteratenr=str_replace(iteratenr,"iter",""),iterate=as.numeric(str_replace(iteratenr,"iter","")))
d

# select all data for shape 1
d1 <- d %>% dplyr::filter(shapes==1)
  
dlabel0 <- obs0df; dlabel0$landmarkid <- unique(d$landmarkid)
dlabelT <- obsTdf; dlabelT$landmarkid <- unique(d$landmarkid)

  
#------------------ figures --------------------------------------------------------------

# plots shapes and noisefields  
shapes <- ggplot() +
    geom_path(data=v0, aes(x=pos1,y=pos2), colour='black')+ #coord_cartesian(xlim = c(-3,3), ylim = c(-2,2))+
    geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange') +
    geom_point(data=nfsdf, aes(x=locx, y=locy), color="Grey")+
    geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = nfsdf,color="Grey",linetype="dotted")+ 
      theme(axis.title.x=element_blank(), axis.title.y=element_blank()) + 
    geom_label(data=dlabel0, aes(x=pos1,y=pos2,label=landmarkid))+
    geom_label(data=dlabelT, aes(x=pos1,y=pos2,label=landmarkid),colour="orange") 
shapes
  
pdf("shapes-noisefields.pdf",width=widthfig,height=4)  
  show(shapes)
dev.off()

  
# plot paths of landmarks positions and bridges over various iterations
dsub <- d1 %>% dplyr::filter(iterate %in% c(0,50,100)) %>%
  mutate(iteratenr = fct_relevel(iteratenr, c("0","50","100")))

p4 <-     dsub %>% ggplot(aes(x=pos1,y=pos2)) +# coord_cartesian(xlim = c(-3,3), ylim = c(-2,2))+
    geom_path(aes(group=interaction(landmarkid,iteratenr),colour=time)) + facet_wrap(~iteratenr) +
    geom_point(data=v0, aes(x=pos1,y=pos2), colour='black')+geom_point(data=vT, aes(x=pos1,y=pos2), colour='orange')+
    geom_path(data=v0, aes(x=pos1,y=pos2), colour='black')+geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange') +
    theme(axis.title.x=element_blank(), axis.title.y=element_blank()) 
p4
    
pdf("bridges-faceted.pdf",width=widthfig,height=2)  
  show(p4)
dev.off()



# plot subset of overlaid landmark bridges
landmarkid_subset <- as.character(seq(1,n,by=4)[1:4]) # only plot these paths and momenta

p1 <- d1 %>% 
  dplyr::filter(landmarkid %in% landmarkid_subset  )%>%
 ggplot() + 
    geom_path(aes(pos1,y=pos2,group=interaction(landmarkid,iteratenr),colour=iterate),size=0.5) +
    scale_colour_gradient(low="grey",high="darkblue")+ 
    geom_point(data=v0, aes(x=pos1,y=pos2), colour='black')+geom_point(data=vT, aes(x=pos1,y=pos2), colour='orange')+
    geom_path(data=v0, aes(x=pos1,y=pos2), colour='black',size=1.0)+geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange',size=1.0) +
    theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
    geom_label(data=filter(dlabel0, landmarkid %in% landmarkid_subset), aes(x=pos1,y=pos2,label=landmarkid,hjust="outward",vjust="outward"))+
  geom_label(data=filter(dlabelT, landmarkid %in% landmarkid_subset), aes(x=pos1,y=pos2,label=landmarkid,hjust="outward",vjust="outward"),colour='orange')+
  geom_point(data=nfsdf, aes(x=locx, y=locy), color="Grey")+
  geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = nfsdf,color="Grey",linetype="dotted")+ # coord_fixed(xlim = c(-2.5,2.5), ylim = c(-2.5,2.5))+
  theme(legend.position='none')#+ coord_fixed()
p1  
pdf("bridges-overlaid.pdf",width=widthfig,height=4)  
  show(p1)
dev.off()



# plot all overlaid landmark bridges
p1all <- d1 %>% 
    ggplot() + 
  geom_path(aes(pos1,y=pos2,group=interaction(landmarkid,iteratenr),colour=iterate),size=0.5) +
  scale_colour_gradient(low="grey",high="darkblue")+ 
  geom_point(data=v0, aes(x=pos1,y=pos2), colour='black')+geom_point(data=vT, aes(x=pos1,y=pos2), colour='orange')+
  geom_path(data=v0, aes(x=pos1,y=pos2), colour='black',size=1.1)+geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange',size=1.0) +
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
    geom_point(data=nfsdf, aes(x=locx, y=locy), color="Grey")+
  geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = nfsdf,color="Grey",linetype="dotted")#+ 
#coord_cartesian(xlim = c(-2.5,2.5), ylim = c(-2.5,2.5))
p1all  
pdf("bridges-overlaid-all.pdf",width=fracwidthfig,height=4)  
p1all
dev.off()


# plot shapes at four different times
d1halfend <- bind_rows(d1,d1) %>% dplyr::filter(time %in% c(0.2604,0.51,0.75,1))

phalf <-
  ggplot() +   
  geom_path(data=d1halfend, aes(x=pos1,y=pos2,colour=iterate,group=iterate),alpha=0.5,size=0.5)+
  #geom_path(data=d1end, aes(x=pos1,y=pos2,colour=iterate,group=iterate),alpha=0.5,size=0.5)+
  geom_point(data=v0, aes(x=pos1,y=pos2), colour='black')+
  geom_point(data=vT, aes(x=pos1,y=pos2), colour='orange')+
  geom_path(data=v0, aes(x=pos1,y=pos2), colour='black',size=0.6)+
  geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange',size=0.6)+
  facet_wrap(~time)+theme(axis.title.x=element_blank(), axis.title.y=element_blank())+scale_colour_gradient(low="grey",high="darkblue")+
  coord_fixed()
phalf


pdf("shapeshalfway.pdf", width = fracwidthfig, height=3.5)
  show(phalf)
dev.off()

  

# plot parameter updates
# ppar1 <- parsdf %>% mutate(cdivgamma2=c/gamma^2) %>% gather(key=par, value=value, a, c, gamma,cdivgamma2) 
# ppar1$par <- factor(ppar1$par, levels=c('a', 'c', 'gamma','cdivgamma2'), labels=c("a","c",expression(gamma),expression(c/gamma^2)))

ppar1 <- parsdf %>% gather(key=par, value=value, a, c, gamma) 
ppar1$par <- factor(ppar1$par, levels=c('a', 'c', 'gamma'), labels=c("a","c",expression(gamma)))
tracepars <- ppar1  %>%  ggplot(aes(x=iterate, y=value)) + geom_path() + geom_point(size=0.5) + facet_wrap(~par, scales="free_y",ncol=1,labeller = label_parsed) +
  xlab("iterate") + ylab("")
tracepars
pdf("trace-pars.pdf",width=widthfig,height=4)
show(tracepars)
dev.off()  

# plot paths of landmarks momenta at time 0 for a subset of landmarks
pmom <-  d1 %>% dplyr::filter(time==0) %>%
  dplyr::filter(landmarkid %in% landmarkid_subset  )%>%
  ggplot(aes(x=mom1,y=mom2,colour=iterate)) + geom_point(size=0.5) +
  #  geom_path(aes(group=interaction(landmarkid,iteratenr),colour=iterate)) +
    facet_wrap(~landmarkid)  +scale_colour_gradient(low="grey",high="darkblue")+
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  geom_hline(yintercept=0, linetype="dashed")+geom_vline(xintercept=0, linetype="dashed")+theme(legend.position='bottom')
pmom

pdf("momenta-faceted.pdf",width=7.5,height=5)  
  grid.arrange(p1,pmom,ncol=2)#show(pmom)
dev.off()

grid.arrange(p1,pmom,ncol=2)



# plot acceptance rates
accfig <- accdf %>% gather(key='kernel',value='acc',-iteration) %>%
  ggplot(aes(x=iteration,y=acc)) +
  geom_point(shape=124)+
  facet_wrap(~kernel)+xlab("iteration nr")+ylab("accept")
accfig
pdf("acceptance.pdf",width=widthfig,height=1.5)
show(accfig)
dev.off()


## plot landmarks evolution for 2 iterates
p3 <- d1 %>% filter(iteratenr %in% c(10,100)) %>%
  ggplot() + 
  geom_path(aes(pos1,y=pos2,colour=time),size=1.0) +
  geom_point(data=vT, aes(x=pos1,y=pos2), colour='orange') +
  geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange',size=1.0) +
  geom_point(data=v0, aes(x=pos1,y=pos2), colour='black') +
  geom_path(data=v0, aes(x=pos1,y=pos2), colour='black',size=1.0)+
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  #geom_label(data=filter(dlabel0, landmarkid %in% landmarkid_subset), aes(x=pos1,y=pos2,label=landmarkid,hjust="outward",vjust="outward"))+
  #geom_label(data=filter(dlabelT, landmarkid %in% landmarkid_subset), aes(x=pos1,y=pos2,label=landmarkid,hjust="outward",vjust="outward"),colour='orange')+
  geom_point(data=nfsdf, aes(x=locx, y=locy), color="Grey")+
  geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = nfsdf,color="Grey",linetype="dotted")+ 
  #coord_fixed(xlim = c(-2.3,2.5), ylim = c(-2.1,2))+ 
  theme(legend.position='none')+ facet_wrap(~iteratenr,ncol=3) 

pdf('landmarks_evolution.pdf',width=widthfig,height=4)
show(p3)
dev.off()





# landamrk momenta at time T for a subset of landmarks
pmomT <-  d %>% dplyr::filter(time==1) %>%
  dplyr::filter(landmarkid %in% landmarkid_subset  )%>%
  ggplot(aes(x=mom1,y=mom2,colour=iterate)) + geom_point(size=0.5) +
  #  geom_path(aes(group=interaction(landmarkid,iteratenr),colour=iterate)) +
  facet_wrap(~landmarkid)  +scale_colour_gradient(low="grey",high="darkblue")+
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  geom_hline(yintercept=0, linetype="dashed")+geom_vline(xintercept=0, linetype="dashed")+theme(legend.position='bottom')
pmomT

#accdf %>% group_by(kernel) %>% count(acc)


ptracemom1 <-  d1 %>% dplyr::filter(time==0) %>%
  dplyr::filter(landmarkid %in% landmarkid_subset  )%>%
  ggplot(aes(x=iterate,y=mom1,colour=iterate)) + geom_point(size=0.5) +
  #  geom_path(aes(group=interaction(landmarkid,iteratenr),colour=iterate)) +
  facet_wrap(~landmarkid)  +scale_colour_gradient(low="grey",high="darkblue")+
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  geom_hline(yintercept=0, linetype="dashed")+geom_vline(xintercept=0, linetype="dashed")+theme(legend.position='bottom')
ptracemom1


ptracemom2 <-  d1 %>% dplyr::filter(time==0) %>%
  dplyr::filter(landmarkid %in% landmarkid_subset  )%>%
  ggplot(aes(x=iterate,y=mom2,colour=iterate)) + geom_point(size=0.5) +
  #  geom_path(aes(group=interaction(landmarkid,iteratenr),colour=iterate)) +
  facet_wrap(~landmarkid)  +scale_colour_gradient(low="grey",high="darkblue")+
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  geom_hline(yintercept=0, linetype="dashed")+geom_vline(xintercept=0, linetype="dashed")+theme(legend.position='bottom')
ptracemom2



