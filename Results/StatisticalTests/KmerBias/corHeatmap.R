
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

flagDatasetCorrelation = TRUE
cellTranslate = c("H1-hESC(DU)","HeLa-S3(DU)","HepG2(DU)","Huvec(DU)","K562(DU)","MCF-7(DU)",
    "DP-K562(DU)","DP-MCF-7(DU)","H7-hESC(UW)","HepG2(UW)","HepG2-HS(UW)","Huvec(UW)",
    "DP-IMR90(UW)","K562(UW)","K562-HS(UW)","K562-RAW(UW)","LNCaP(DU)","m3134(UW)" )
names(cellTranslate) = c("DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","DU_Mcf7",
    "DU_NakedK562","DU_NakedMcf7","UW_H7hesc","UW_HepG2","UW_HepG2_HS","UW_Huvec",
    "UW_IMR90","UW_K562","UW_K562_HS","UW_K562_RAW","UW_LnCaP","UW_m3134" )

myDist = function(p1) dist(p1, method="euclidean")
myHclust = function(p2) hclust(p2, method="ward.D")
source('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/newHeatmap2.R')

#################################################
# Function
#################################################

createCorHeatmap <- function(inFileName,outFileName){

  # Reading correlation into data matrix
  data = read.table(inFileName, sep='\t', header=T, row.names = 1)

  # Changing colnames and rownames
  #for (i in 1:length(colnames(data))) {
  #  label = colnames(data)[i]
  #  ls = strsplit(label, "_")[[1]]
  #  newlabel = paste(cellTranslate[ls[2]],"(",ls[1],")",sep="")
  #  colnames(data)[i] = newlabel
  #}
  #for (i in 1:length(rownames(data))) {
  #  label = rownames(data)[i]
  #  ls = strsplit(label, "_")[[1]]
  #  newlabel = paste(cellTranslate[ls[2]],"(",ls[1],")",sep="")
  #  rownames(data)[i] = newlabel
  #}

  for (i in 1:length(colnames(data))) {
    colnames(data)[i] = cellTranslate[colnames(data)[i]]
  }
  for (i in 1:length(rownames(data))) {
    rownames(data)[i] = cellTranslate[rownames(data)[i]]
  }

  # Melting
  dataMatrix = as.matrix(data)
  dataMelt <- melt(dataMatrix)
  dataMelt$X2 <- factor(dataMelt$X2, levels = as.character(colnames(data))) # Needed to keep the order of the labels
  dataMelt$X1 <- factor(dataMelt$X1, levels = as.character(rownames(data))) # Needed to keep the order of the labels

  # Plotting graph
  postscript(outFileName, width=9.0, height=6.0, horizontal=FALSE, paper="special", onefile=FALSE)
  par(mar=c(6,5,4,3))
  p = ggplot(dataMelt, aes(X1, X2, fill = value)) 
  p = p + theme_bw()
  p = p + geom_tile()
  p = p + scale_fill_gradient2("k-mer bias Spearman\ncorrelation (R)", low = "red", mid = "black",  high = "green", limits=c(-1.0, 1.0), na.value="white") 
  p = p + ylab("") + xlab("")

  p = p + theme(axis.text.x = element_text(angle=90, size=16, vjust=0.5, face="bold", color="black"),
                axis.text.y = element_text(size=16, face="bold", color="black"),
                legend.text = element_text(colour="black", size = 14))
                #legend.position=c(-0.09,-0.145))
                #legend.position=c(.85, .8))

  #p = p + scale_y_discrete(limits = rev(levels(dataMelt$X2)) )

  p = p + theme(plot.background = element_blank(), panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(), panel.border = element_blank())

  #p = p + annotate("text", x = -0.5, y = 0.0, label = "correlation", angle = 90)

  print(p)
  dev.off()
  system(paste("epstopdf",outFileName,sep=" "))
  #system(paste("rm",outFileName,sep=" "))

}

createCorHeatmapDendrogram <- function(inFileName,outFileName){

  # Fetching data
  hmbreaks = c(seq(0.0,0.5,length=10),seq(0.5,1.0,length=10))
  hmcol = colorRampPalette(c("black", "green"))(n = 19)
  #hmcol = brewer.pal(11,"RdBu")
  data = read.table(inFileName, sep='\t', header=T, row.names = 1)
  data = as.matrix(data)

  # Changing colnames and rownames
  #for (i in 1:length(colnames(data))) {
  #  label = colnames(data)[i]
  #  ls = strsplit(label, "_")[[1]]
  #  newlabel = paste(cellTranslate[ls[2]],"(",ls[1],")",sep="")
  #  colnames(data)[i] = newlabel
  #}
  #for (i in 1:length(rownames(data))) {
  #  label = rownames(data)[i]
  #  ls = strsplit(label, "_")[[1]]
  #  newlabel = paste(cellTranslate[ls[2]],"(",ls[1],")",sep="")
  #  rownames(data)[i] = newlabel
  #}

  for (i in 1:length(colnames(data))) {
    colnames(data)[i] = cellTranslate[colnames(data)[i]]
  }
  for (i in 1:length(rownames(data))) {
    rownames(data)[i] = cellTranslate[rownames(data)[i]]
  }

  # Plot
  postscript(outFileName,width=8.0,height=7.0,horizontal=FALSE,paper='special')
  par(cex.axis=1.2, cex.main=1.0, mar=c(10,5,4,10))
  heatmap.2(data, col=hmcol, breaks=hmbreaks, main="", cexCol=1.2, cexRow=1.2, trace='none',
            sepwidth=c(2,2), sepcolor='black', Rowv=TRUE, Colv=TRUE, density.info = 'none', lhei = c(1, 4),
            distfun=myDist, hclustfun=myHclust, keysize = 2, margins=c(9,9) )
  dev.off()
  system(paste('epstopdf',outFileName,sep=' '))

}


#################################################
# Execution
#################################################

if(flagDatasetCorrelation){

# Input
loc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBias/CorrelationGraphs/Dataset_spearman/"
inList = c("Sampled_All_corr")

# Strand Loop
for (i in 1:length(inList)) {

  # Parameters
  inFileName = paste(loc,inList[i],".txt",sep="")
  outFileName = paste(loc,inList[i],".eps",sep="")

  # Creating heatmap
  createCorHeatmapDendrogram(inFileName,outFileName)

}

}


