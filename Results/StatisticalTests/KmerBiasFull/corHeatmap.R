
#################################################
# Import
#################################################

library(ggplot2)
library(reshape)
library(gplots)
library(gtools)
library(RColorBrewer)

#################################################
# Parameters
#################################################

cellData = read.table('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBiasFull/bias/CellFull.txt', sep='\t', header=F)
cellTranslate = as.character(cellData[,2])
names(cellTranslate) = as.character(cellData[,1])
colorTranslate = as.character(cellData[,3])
names(colorTranslate) = as.character(cellData[,1])
myDist = function(p1) dist(p1, method="euclidean")
myHclust = function(p2) hclust(p2, method="ward.D")
source('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/newHeatmap2S.R')

#################################################
# Function
#################################################

createCorHeatmapDendrogram <- function(inFileName,outFileName){

  # Fetching data
  hmbreaks = c(seq(0.0,0.8,length=15),seq(0.8,1.0,length=15))
  hmcol = colorRampPalette(c("black", "green"))(n = 29)
  data = read.table(inFileName, sep='\t', header=T, row.names = 1)
  data = as.matrix(data)
  rscol = c()
  cscol = c()

  for (i in 1:length(colnames(data))) {
    cscol = c(cscol,colorTranslate[colnames(data)[i]])
    colnames(data)[i] = cellTranslate[colnames(data)[i]]
  }
  for (i in 1:length(rownames(data))) {
    rscol = c(rscol,colorTranslate[rownames(data)[i]])
    rownames(data)[i] = cellTranslate[rownames(data)[i]]
  }

  lmat=rbind(c(5,4,0), c(3,2,1))
  lhei=c(0.5, 4.5)
  #lwid=c(1,4,0.25)
  lwid=c(1.5,3.5,0.25)

  # Plot
  postscript(outFileName,width=16.0,height=15.0,horizontal=FALSE,paper='special')
  par(cex.axis=1.0, cex.main=1.0, mar=c(12,3,3,12))
  heatmap.2(data, col=hmcol, breaks=hmbreaks, main="", cexCol=1.0, cexRow=1.0, trace='none',
            sepwidth=c(2,2), sepcolor='black', Rowv=TRUE, Colv=TRUE, density.info = 'none',
            lmat = lmat, lhei = lhei, lwid = lwid, RowSideColors=rscol,
            distfun=myDist, hclustfun=myHclust, keysize = 1, margins=c(12,12))
  dev.off()
  system(paste('epstopdf',outFileName,sep=' '))

}

createSmallerCorHeatmapDendrogram <- function(inFileName,outFileName){

  # Fetching data
  hmbreaks = c(seq(0.0,0.8,length=15),seq(0.8,1.0,length=15))
  hmcol = colorRampPalette(c("black", "green"))(n = 29)
  data = read.table(inFileName, sep='\t', header=T, row.names = 1)
  data = as.matrix(data)
  rscol = c()
  cscol = c()

  # Selecting
  data = data[c("DUHS_Gm12878","DUHS_H1hesc","DUHS_K562","DUHS_NakedK562","DUHS_NakedMcf7",     
              "UWDGF_NakedIMR90","UWHS_Gm12878","UWHS_H1hesc","UWHS_K562"),
              c("DUHS_Gm12878","DUHS_H1hesc","DUHS_K562","DUHS_NakedK562","DUHS_NakedMcf7",     
              "UWDGF_NakedIMR90","UWHS_Gm12878","UWHS_H1hesc","UWHS_K562")]

  for (i in 1:length(colnames(data))) {
    cscol = c(cscol,colorTranslate[colnames(data)[i]])
    colnames(data)[i] = cellTranslate[colnames(data)[i]]
  }
  for (i in 1:length(rownames(data))) {
    rscol = c(rscol,colorTranslate[rownames(data)[i]])
    rownames(data)[i] = cellTranslate[rownames(data)[i]]
  }

  lmat=rbind(c(5,4,0), c(3,2,1))
  lhei=c(2.0, 4.0)
  #lwid=c(1,4,0.25)
  lwid=c(2.5,3.5,0.25)

  # Plot
  postscript(outFileName,width=16.0,height=15.0,horizontal=FALSE,paper='special')
  par(cex.axis=1.0, cex.main=1.0, mar=c(20,3,3,20))
  heatmap.2(data, col=hmcol, breaks=hmbreaks, main="", cexCol=3.0, cexRow=3.0, trace='none',
            sepwidth=c(2,2), sepcolor='black', Rowv=TRUE, Colv=TRUE, density.info = 'none',
            lmat = lmat, lhei = lhei, lwid = lwid, RowSideColors=rscol,
            distfun=myDist, hclustfun=myHclust, keysize = 1, margins=c(20,20))
  dev.off()
  system(paste('epstopdf',outFileName,sep=' '))

}

#################################################
# Execution
#################################################

# Input
loc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBiasFull/correlation/"
inList = c("spearman_All_corr")

# Strand Loop
for (i in 1:length(inList)) {

  # Parameters
  inFileName = paste(loc,inList[i],".txt",sep="")
  #outFileName = paste(loc,inList[i],".eps",sep="")
  outFileName = paste(loc,inList[i],"_small.eps",sep="")

  # Creating heatmap
  #createCorHeatmapDendrogram(inFileName,outFileName)
  createSmallerCorHeatmapDendrogram(inFileName,outFileName)

}


