
###################################################################################################
# INPUT
###################################################################################################

neg = "H1hesc"
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

  postscript(outFileName,width=2.5,height=5.0,horizontal=FALSE,paper='special')
  par(cex.axis=1.0)
  test_df2 <- w2l(dataFrame)
  p = bwplot(value ~ variable, data = test_df2,
         scales=list(tck=1, x=list(rot=90,cex=1.0), y=list(cex=1.0)), 
         #col = c("#CC0000FF","#0000CCFF"),
         main = "", xlab = "", ylab = list("",cex=1.8),
         #par.settings = list( plot.symbol=list(col = "black"),
         #box.rectangle = list(col = c("#CC0000FF","#0000CCFF")),
         #box.dot = list(col = "black"), 
         #box.umbrella= list(lty=1, col = c("#CC0000FF","#0000CCFF"))),
         panel=function(...) {
           panel.grid(h=-1,v=0,lty=2,col="lightgray")
           panel.bwplot(..., fill = c("#CC0000FF","#0033FFFF"))
         })
  print(p)
  dev.off()
  system(paste("epstopdf",outFileName,sep=" "))
  #system(paste("rm",outFileName,sep=" "))

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
tfVec = c("MYCN", "FOXH1", "NR5A2", "CDX2", "BHLHE40")
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
      data.sampled = data.frame(H1hesc=vec1, K562=vec2)
      outFileName = paste(oLoc,paste(t,method,metric,sep="_"),'.eps',sep='')

      createBoxplot(data.sampled, outFileName)

      write(c(t,vec1[!is.na(vec1)]),file=paste(oLoc,t,"_vec1.txt",sep=""),ncolumns=1,append=FALSE,sep='\t')
      write(c(t,vec2[!is.na(vec2)]),file=paste(oLoc,t,"_vec2.txt",sep=""),ncolumns=1,append=FALSE,sep='\t')

    }

  }
  
}


