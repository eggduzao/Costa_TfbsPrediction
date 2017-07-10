
###################################################################################################
# INPUT
###################################################################################################

# Import
library(lattice)
library(reshape)
library(plotrix)
w2l <- function(xx) melt(xx, measure.vars = colnames(xx))

###################################################################################################
# FUNCTIONS
###################################################################################################

# BOXPLOT
createBoxplot <- function(dataFrame, outFileName){

  postscript(outFileName,width=12.0,height=6.0,horizontal=FALSE,paper='special')
  par(cex.axis=1.0)
  test_df2 <- w2l(dataFrame)
  p = bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=90,cex=1.0), 
         y=list(cex=1.0)), col = "black",
         main = "", xlab = "", ylab = list("K-mer Bias Distribution",cex=1.2),
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

hypTest <- function(dataFrame, methodNameVec, outFileName){

  toWrite = c("XXXXX",methodNameVec)
  for (i in (1:ncol(dataFrame))){
    toWrite = c(toWrite,methodNameVec[i])
    for (j in (1:ncol(dataFrame))){
      if(i == j){
        toWrite = c(toWrite,"XXXXX")
      }
      else{
        teste = wilcox.test(dataFrame[,i],dataFrame[,j],paired=TRUE)
        toWrite = c(toWrite,format(teste$p.value,digits=5))
      }
    }
  }
  write(toWrite,file=outFileName,ncolumns=ncol(dataFrame)+1,append=FALSE,sep='\t')

}

###################################################################################################
# EXECUTION
###################################################################################################

# Input
iLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBiasFull/bias/table/"
oLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBiasFull/bias_distribution/"

cellList = c(
"DUHS_A549","DUHS_K562","DUHS_Mcf7","UWHS_A549","UWHS_K562","UWHS_Mcf7","DUHS_Hepg2",
"DUHS_Huvec","DUHS_Imr90","DUHS_LNCaP","DUHS_Sknsh","UWDGF_A549","UWDGF_K562","UWHS_Hepg2","UWHS_Huvec",
"DUHS_H1hesc","DUHS_H7hesc","DUHS_Helas3","UWDGF_Hepg2","UWDGF_Huvec","UWDGF_m3134","UWHS_H1hesc","UWHS_Helas3",
"UWHS_Lhcnm2","DUHS_Gm12878","UWDGF_H7hesc","UWDGF_Lhcnm2","UWHS_Gm12878","DUHS_Monocd14","UWHS_Monocd14",
"DUHS_K562Nabut","DUHS_NakedK562","DUHS_NakedMcf7","UWHS_K562Znfp5","UWDGF_K562Znfp5","DUHS_Cd20ro01794",
"DUHS_K562G1phase","UWDGF_NakedIMR90","UWHS_Cd20ro01778","DUHS_Helas3Ifna4h","DUHS_K562G2mphase",
"DUHS_K562Sahactrl","DUHS_Mcf7Hypoxlac","UWDGF_Cd20ro01778","UWHS_K562Znf4g7d3","UWHS_K562Znfa41c6",
"UWHS_K562Znfb34a8","UWHS_K562Znff41b2","UWHS_Lhcnm2Diff4d","DUHS_Mcf7Ctcfshrna","DUHS_Mcf7Randshrna",
"UWDGF_K562Znfa41c6","UWDGF_Lhcnm2Diff4d","UWHS_K562Znf2c10c5","UWHS_K562Znf4c50c4","UWHS_K562Znfe103c6",
"UWHS_K562Znfg54a11","UWHS_Mcf7Estctrl0h","DUHS_K562Saha1u72hr","UWHS_Mcf7Est100nm1h","UWHS_Monocd14ro1746")

plotMat = c()
for(cell in cellList){
  inFileName = paste(iLoc,"hs_",cell,"_6_All.txt",sep="")
  data = read.table(inFileName, header=TRUE)
  vec = data[,2]
  plotMat = cbind(plotMat,vec)
}
medVec = apply(plotMat,2,median)
sortInd = sort(medVec, decreasing = TRUE, index.return = TRUE, na.last=NA)$ix
plotMat = plotMat[,sortInd]
colnames(plotMat) = cellList[sortInd]
plotMat = as.data.frame(plotMat)
outGraphName = paste(oLoc,"spearman_bias_boxplot.eps",sep="")
createBoxplot(plotMat,paste(outGraphName,sep=""))
outTestName = paste(oLoc,"spearman_bias_boxplot.txt",sep="")
hypTest(plotMat,colnames(plotMat),outTestName)


