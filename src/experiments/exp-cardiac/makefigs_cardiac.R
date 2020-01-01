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


landmarkid_subset <- as.character(seq(1,n,by=4)[1:4]) # only plot these paths and momenta

subsamplenr <-  10 # only show every subsamplenr-th iterate






# plot acc probs
accdf <- read_delim("accdf.csv", ";", escape_double = FALSE, trim_ws = TRUE)
accfig <- accdf %>% mutate(acc=as.factor(acc)) %>% 
      mutate(kernel=recode(kernel, mala_mom="MALA momenta" ))  %>%
    ggplot(aes(x=iter,y=acc)) +geom_point(shape=124)+ facet_wrap(~kernel)+xlab("iteration nr")+ylab("accept")
pdf("acceptance.pdf",width=widthfig,height=2.5)  
  show(accfig)
dev.off()
accfig


pmomT <-  d %>% dplyr::filter(time==1) %>%
  dplyr::filter(landmarkid %in% landmarkid_subset  )%>%
  ggplot(aes(x=mom1,y=mom2,colour=iterate)) + geom_point(size=0.5) +
  facet_wrap(~landmarkid)  +scale_colour_gradient(low="grey",high="darkblue")+
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  geom_hline(yintercept=0, linetype="dashed")+geom_vline(xintercept=0, linetype="dashed")+theme(legend.position='bottom')
pmomT

ppos0 <-  d %>% dplyr::filter(time==0) %>% mutate(shapes=as.factor(shapes)) %>%
  dplyr::filter(landmarkid %in% landmarkid_subset  )%>%
    ggplot(aes(x=pos1,y=pos2,colour=iterate)) + geom_point(size=0.5) +
  facet_wrap(~landmarkid)  +scale_colour_gradient(low="grey",high="darkblue")+
  theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
  geom_hline(yintercept=0, linetype="dashed")+geom_vline(xintercept=0, linetype="dashed")+theme(legend.position='bottom')
ppos0


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



# plot initial positions for all shapes (only interesting in case initial shape is unobserved)
dtime0 <- d %>% dplyr::filter(time==0)

# add factor for determining which phase of sampling  
dtime0 <-  dtime0%>% dplyr::filter(iterate %in% seq(0,max(d$iterate),by=subsamplenr))%>% mutate(phase = 
                               case_when(iterate < quantile(iterate,1/2) ~ "initial",
                                 iterate >= quantile(iterate,1/2) ~ "final")  ) %>% # reorder factor levels
                          mutate(phase = fct_relevel(phase, "initial"))
dtime0double <- bind_rows(dtime0,dtime0) %>% mutate(landmark=as.numeric(landmarkid))


vTinner <- vT %>% dplyr::filter(landmark<=7)
vTouter <- vT %>% dplyr::filter(landmark>7)

dtime0doubleinner <- dtime0double %>% dplyr::filter(landmark<=7)
dtime0doubleouter <- dtime0double %>% dplyr::filter(landmark>7)
initshapes0 <- ggplot()  + 
  geom_point(data=vTinner, aes(x=pos1,y=pos2), colour='orange',size=0.4)+
  geom_path(data=vTinner, aes(x=pos1,y=pos2,group=shape), colour='orange', linetype="dashed",size=0.4) +
  geom_point(data=vTouter, aes(x=pos1,y=pos2), colour='orange',size=0.4)+
  geom_path(data=vTouter, aes(x=pos1,y=pos2,group=shape), colour='orange', linetype="dashed",size=0.4) +
    geom_path(data=dtime0doubleinner,aes(x=pos1,y=pos2,colour=iterate),size=0.4) +
  geom_path(data=dtime0doubleouter,aes(x=pos1,y=pos2,colour=iterate),size=0.4) +
 # scale_colour_gradient(low="grey",high="darkblue")+
  facet_wrap(~phase,ncol=2)+ xlab("")+ylab("")#coord_fixed()+
initshapes0


pdf("initial-shapes.pdf",width=widthfig,height=3)  
initshapes0
dev.off()

# tabulate acc probs 
accdf %>% group_by(kernel)%>%summarise(a=mean(acc))
