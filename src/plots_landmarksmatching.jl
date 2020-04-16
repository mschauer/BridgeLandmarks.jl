"""
  plotlandmarksmatching(outdir)

Script for some plots after calling landmarksmatching.
"""
function plotlandmarksmatching(outdir)
  @warn "R packages tidyverse and ggforce should be installed."

  @rput outdir
  R"""
  setwd(outdir)
  widthfig <- 7.5
  library(tidyverse)
  library(ggforce)
  theme_set(theme_light())

  ########  read observations and process
  obs0df <- read_delim("obs0.csv", ";", escape_double = FALSE, trim_ws = TRUE)
  obsTdf <- read_delim("obsT.csv", ";", escape_double = FALSE, trim_ws = TRUE) %>% spread(key=pos,value=value) %>%
      mutate(shape=as.factor(shape))
  v0 <- bind_rows(obs0df, obs0df)
  vT <- bind_rows(obsTdf, obsTdf)

  #######  read noisefields
  nfsdf <- read_delim("noisefields.csv",";", escape_double = FALSE, trim_ws = TRUE)
  ######## read and analyse bridge iterates
  d <- read_table2("iterates.csv") %>%
    mutate(landmarkid=as.factor(landmarkid)) %>%
    gather(key="iteratenr",value="value",contains("iter"))
  d <- d %>% mutate(i = rep(1:(nrow(d)/4),each=4)) # add column to identify a particular combination (pos1,pos2,mom1,mom2) of one landmark at a specific time of a specific shape
  d <- d %>% spread(key=pqtype, value=value) %>% dplyr::select(-i) %>%  # can drop identifier column
    mutate(iteratenr=str_replace(iteratenr,"iter",""),iterate=as.numeric(str_replace(iteratenr,"iter","")))
  ####### read parameter updates
  parsdf <- read_delim("parameters.csv", ";", escape_double = FALSE, trim_ws = TRUE)
  ####### read acceptance information in mcmc-steps
  accdf <- read_delim("accdf.csv", ";", escape_double = FALSE, trim_ws = TRUE)

  # plots shapes and noisefields
  shapes <- ggplot() +
      geom_path(data=v0, aes(x=pos1,y=pos2), colour='black')+
      geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange') +
      geom_point(data=nfsdf, aes(x=locx, y=locy), color="Grey")+
      geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = nfsdf,color="Grey",linetype="dotted")+
        theme(axis.title.x=element_blank(), axis.title.y=element_blank())
  pdf("shapes-noisefields.pdf",width=widthfig,height=4)
    show(shapes)
  dev.off()

  # plot overlaid landmark bridges
  bridges <- d %>%
   ggplot() +
      geom_path(aes(pos1,y=pos2,group=interaction(landmarkid,iteratenr),colour=iterate),size=0.5) +
      scale_colour_gradient(low="grey",high="darkblue")+
      geom_point(data=v0, aes(x=pos1,y=pos2), colour='black')+geom_point(data=vT, aes(x=pos1,y=pos2), colour='orange')+
      geom_path(data=v0, aes(x=pos1,y=pos2), colour='black',size=1.0)+geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange',size=1.0) +
      theme(axis.title.x=element_blank(), axis.title.y=element_blank()) +
      geom_point(data=nfsdf, aes(x=locx, y=locy), color="Grey")+
    geom_circle(aes(x0 = locx, y0 = locy, r = nfstd), data = nfsdf,color="Grey",linetype="dotted")+
     theme(legend.position='none')#+ coord_fixed()
  pdf("bridges-overlaid.pdf",width=widthfig,height=4)
    show(bridges)
  dev.off()

  # traceplot of parameters
  tracepars <- parsdf %>% gather(key=par, value=value, a, c, gamma)  %>%
    ggplot(aes(x=iterate, y=value)) + geom_path() + facet_wrap(~par, scales="free_y",ncol=1) +
   xlab("iterate") + ylab("")
  pdf("trace-pars.pdf",width=widthfig,height=4)
    show(tracepars)
  dev.off()

  # plot shape evolution
  ut <- unique(d$time)
  ind <- round(quantile(1:length(ut), (1:4)/4))
  times <- ut[ind]
  dhalfend <- bind_rows(d,d) %>%  dplyr::filter(time %in% times)

  shapes_evolution <-
    ggplot() +
    geom_path(data=dhalfend, aes(x=pos1,y=pos2,colour=iterate,group=iterate),alpha=0.5,size=0.5)+
      geom_point(data=v0, aes(x=pos1,y=pos2), colour='black')+
    geom_point(data=vT, aes(x=pos1,y=pos2), colour='orange')+
    geom_path(data=v0, aes(x=pos1,y=pos2), colour='black',size=0.6)+
    geom_path(data=vT, aes(x=pos1,y=pos2,group=shape), colour='orange',size=0.6)+
    facet_wrap(~time,ncol=2)+
    theme(axis.title.x=element_blank(), axis.title.y=element_blank())+
    scale_colour_gradient(low="grey",high="darkblue")+
  pdf("shapes_evolution.pdf", width = widthfig, height=4)
  show(shapes_evolution)
  dev.off()

  # plot acceptance rates
  accfig <- accdf %>% gather(key='kernel',value='acc',-iteration) %>%
        ggplot(aes(x=iteration,y=acc)) +
      geom_point(shape=124)+
      facet_wrap(~kernel)+xlab("iteration nr")+ylab("accept")
  pdf("acceptance.pdf",width=widthfig,height=4)
    show(accfig)
  dev.off()
  """
end
