
###################################################################################################
# INPUT
###################################################################################################

neg = "GM12878"
pos = "K562"

# Import
library(lattice)
library(reshape)
library(plotrix)
w2l <- function(xx) melt(xx, measure.vars = c(neg,pos))

###################################################################################################
# FUNCTIONS
###################################################################################################

# BOXPLOT
createBoxplot <- function(dataFrame, outFileName){

  postscript(outFileName,width=5.0,height=5.0,horizontal=FALSE,paper='special')
  par(cex.axis=1.0)
  test_df2 <- w2l(dataFrame)
  p = bwplot(value ~ variable, data = test_df2, 
         scales=list(tck=0, x=list(rot=90,cex=1.0), y=list(cex=1.0)), col = "black",
         main = "", xlab = "", ylab = list("Y LABEL",cex=1.8),
         par.settings = list( plot.symbol=list(col = "black"),box.rectangle = list(col = "black"),
         box.dot = list(col = "black"), box.umbrella= list(lty=1, col = "black")),
         panel=function(...) {
           panel.bwplot(...)
         })
  print(p)
  dev.off()
  system(paste("epstopdf",outFileName,sep=" "))
  system(paste("rm",outFileName,sep=" "))

}

remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

###################################################################################################
# EXECUTION
###################################################################################################

# Input
iLoc = "./distribution/"
oLoc = "./boxplot/"

# Parameters
tfVec = c("ETS1", "REL", "SOX5", "GATA1")
methodVec = c("HINTBC")
metricVec = c("flr")

# TF Loop
for(t in tfVec){

  inFileName = paste(iLoc,t,"_filt.txt",sep="")
  data = read.table(inFileName, header=TRUE)

  # Method Loop
  for(method in methodVec){

    # Metric Loop
    for(metric in metricVec){

      vec1 = remove_outliers(data[,paste(method,neg,metric,sep="_")])
      vec2 = remove_outliers(data[,paste(method,pos,metric,sep="_")])

      print(paste(t,method,metric,sep="_"))
      print(c(mean(vec1,na.rm=TRUE), mean(vec2,na.rm=TRUE)))
      print(c(median(vec1,na.rm=TRUE), median(vec2,na.rm=TRUE)))
      data.sampled = data.frame(GM12878=vec1, K562=vec2)
      outFileName = paste(oLoc,paste(t,method,metric,sep="_"),'.eps',sep='')

      createBoxplot(data.sampled, outFileName)

    }

  }
  
}


