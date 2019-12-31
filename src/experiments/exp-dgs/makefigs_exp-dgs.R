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

ALLPLOTS <- FALSE
  
obs0df <- read_delim("obs0.csv", ";", escape_double = FALSE, trim_ws = TRUE)
obsTdf <- read_delim("obsT.csv", ";", escape_double = FALSE, trim_ws = TRUE) %>% spread(key=pos,value=value) %>%
    mutate(shape=as.factor(shape))


v0 <- bind_rows(obs0df, obs0df)
vT <- bind_rows(obsTdf, obsTdf)
vT1 <- vT %>% dplyr::filter(shape=="1")
n <- max(v0$landmark)
  
#######  read noisefields
nfsdf <- read_delim("noisefields.csv",";", escape_double = FALSE, trim_ws = TRUE)
  
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
  
dsub <- d1 %>% dplyr::filter(iterate %in% c(0,50,100)) %>%
    mutate(iteratenr = fct_relevel(iteratenr, c("0","50","100")))
dlabel0 <- obs0df; dlabel0$landmarkid <- unique(d$landmarkid)
dlabelT <- obsTdf; dlabelT$landmarkid <- unique(d$landmarkid)

####### read parameter updates
parsdf <- read_delim("parameters.csv", ";", escape_double = FALSE, trim_ws = TRUE) 
  
  #------------------ figures --------------------------------------------------------------
if (ALLPLOTS) {
# plots shapes and noisefields  
shapes <- ggplot() +
    geom_path(data=v0, aes(x=pos1,y=pos2), colour='black')+ coord_cartesian(xlim = c(-3,3), ylim = c(-2,2))+
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
}
  
# plot paths of landmarks positions and bridges over various iterations
p4 <-     dsub %>% ggplot(aes(x=pos1,y=pos2)) + coord_cartesian(xlim = c(-2,2), ylim = 0.8*c(-1,1))+
    geom_path(aes(group=interaction(landmarkid,iteratenr),colour=time)) + facet_wrap(~iteratenr) +
    geom_point(data=v0, aes(x=pos1,y=pos2), colour='black')+geom_point(data=vT, aes(x=pos1,y=pos2), colour='orange')+
    geom_path(data=v0, aes(x=pos1,y=pos2), colour='black')+geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange') +
    theme(axis.title.x=element_blank(), axis.title.y=element_blank()) 
p4
    
pdf("bridges-faceted.pdf",width=widthfig,height=2)  
  show(p4)
dev.off()

landmarkid_subset <- as.character(seq(1,n,by=4)[1:4]) # only plot these paths and momenta

subsamplenr <-  10 # only show every subsamplenr-th iterate

# plot overlaid landmark bridges
p1 <- d1 %>% dplyr::filter(iterate %in% seq(0,max(d$iterate),by=subsamplenr))%>%
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
  geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = nfsdf,color="Grey",linetype="dotted")+ 
   coord_fixed(xlim = c(-2,2), ylim = 0.8*c(-1,1))+theme(legend.position='none')#+ coord_fixed()
p1  
pdf("bridges-overlaid.pdf",width=widthfig,height=4)  
  show(p1)
dev.off()

# plot overlaid landmark bridges
p1all <- d1 %>% dplyr::filter(iterate %in% seq(0,max(d$iterate),by=subsamplenr))%>%
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

fracwidhtfig = 0.8 * widthfig
pdf("bridges-overlaid.pdf",width=fracwidhtfig,height=4)  
p1all
dev.off()

  
if (ALLPLOTS) {
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
  
# plot paths of landmarks momenta
pmom <-  d %>% dplyr::filter(time==0) %>%
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

grid.arrange(p1,pmom,ncol=2)#show(pmom)

  accdf <- read_delim("accdf.csv", ";", escape_double = FALSE, trim_ws = TRUE)
  accfig <- accdf %>% mutate(acc=as.factor(acc)) %>% 
      mutate(kernel=recode(kernel, mala_mom="MALA momenta" ))  %>%
    ggplot(aes(x=iter,y=acc)) +geom_point(shape=124)+ facet_wrap(~kernel)+xlab("iteration nr")+ylab("accept")
pdf("acceptance.pdf",width=widthfig,height=1.5)  
  show(accfig)
dev.off()
accfig


pmomT <-  d %>% dplyr::filter(time==1) %>%
  dplyr::filter(landmarkid %in% landmarkid_subset  )%>%
  ggplot(aes(x=mom1,y=mom2,colour=iterate)) + geom_point(size=0.5) +
  #  geom_path(aes(group=interaction(landmarkid,iteratenr),colour=iterate)) +
  facet_wrap(~landmarkid)  +scale_colour_gradient(low="grey",high="darkblue")+
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  geom_hline(yintercept=0, linetype="dashed")+geom_vline(xintercept=0, linetype="dashed")+theme(legend.position='bottom')
pmomT


if (PARESTIMATION == TRUE)
{
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