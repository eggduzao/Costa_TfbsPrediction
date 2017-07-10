
###########################################################
# Import
###########################################################

# Import
library(lattice)
library(reshape)
library(plotrix)
library(ggplot2)
library(scales)

###########################################################
# Correlation Function
###########################################################

createGroupedCorrelation <- function(matrix1, matrix2, legendLab, colorList, legPos, txtWid, mainText,
                                     outFileName, outFileNameTable, outFileNamePvalue){

  # Calculating correlation coefficients and writing the graph in tabular form
  pList = c()
  cList = c()
  finalLegendLabels = c()
  toWrite = c()
  toWriteP = c()
  for (col in 1:ncol(matrix1)){
    corrTest = cor.test(matrix1[,col], matrix2[,col], alternative = "two.sided", method = "spearman", conf.level = 0.95) # Correlation
    pList = c(pList,corrTest$p.value)
    cList = c(cList,corrTest$estimate)
    finalLegendLabels = c(finalLegendLabels,paste(legendLab[col],", R=",round(corrTest$estimate,3),sep=""))#,
                          #", p-value",format.pval(corrTest$p.value),sep=""))
    toWrite = c(toWrite,paste(legendLab[col],"_AVERAGE_BIAS",sep=""),as.numeric(matrix1[,col]),
                        paste(legendLab[col],"_CG_CONTENT",sep=""),as.numeric(matrix2[,col]))
    toWriteP = c(toWriteP,legendLab[col],corrTest$estimate,corrTest$p.value)
  }
  toWrite = as.character(t(matrix(toWrite, nrow=33, ncol = 2*ncol(matrix1), byrow = FALSE)))
  write(toWrite,file=outFileNameTable,ncolumns=2*ncol(matrix1),append=FALSE,sep='\t')
  write(toWriteP,file=outFileNamePvalue,ncolumns=3,append=FALSE,sep='\t')   

  # Initializing graph
  postscript(outFileName, width=6.0, height=6.0, horizontal=FALSE, paper='special')
  par(mar=c(4.2,5,6,1))

  # Calculating first correlation line
  dataFr = data.frame(X=matrix1[,1], Y=matrix2[,1])
  regLine = lm(Y~X, data=dataFr) # Regression line (y,x)

  # Creating plot with first correlation
  plot(matrix1[,1], matrix2[,1], xlab="Average Predicted Bias (log)", ylab="CG Content", 
       cex.lab=1.3, cex.axis=1.3, cex.main=1.5, cex.sub=1.3,
       col = colorList[1], pch = 19)
  abline(regLine, lty=1, lwd=3.0, col=colorList[1])

  # Ploting rest of data
  for (col in 2:ncol(matrix1)){
    dataFr = data.frame(X=matrix1[,col], Y=matrix2[,col])
    regLine = lm(Y~X, data=dataFr) # Regression line (y,x)
    points(matrix1[,col], matrix2[,col], col = colorList[col], pch = 19)
    abline(regLine, lty=1, lwd=3.0, col=colorList[col])
  }

  # Legend
  legend(legPos[1], legPos[2], legend=finalLegendLabels, lty=NA, col=colorList, 
         xpd = TRUE, ncol=2, bty = "n", lwd=5.0, cex=1.0, pch = 19, text.width=txtWid)

  # Main text
  text(as.numeric(mainText[1]), as.numeric(mainText[2]), mainText[3], xpd = TRUE, cex=1.5)

  # End of graph
  grid()
  dev.off()
  system(paste("epstopdf",outFileName,sep=" "))

}

###########################################################
# Execution
###########################################################

# Parameters
inLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MotifDinucl/table/"
ol = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MotifDinucl/kmer_bias_correlation/"
avgbiasData = read.table(paste(inLoc,'kmer_table_avgbias.txt',sep=''), sep='\t', header=T)
kmerCgData = read.table(paste(inLoc,'kmer_table_CG.txt',sep=''), sep='\t', header=T)
avgbiasData = log(avgbiasData)

# DU datasets
du_datasets = c("DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","DU_Mcf7","UW_LnCaP")
du_labels = c("H1-hESC","HeLa-S3","HepG2","Huvec","K562","MCF-7","LNCaP")
matrix1 = avgbiasData[,du_datasets]
matrix2 = kmerCgData[,du_datasets]
colorList = c("#000000FF", "#00FFFFFF", "#00FF40FF", "#FFBF00FF", "#FF0000FF", "#336611FF", "#0040FFFF")
outFileName = paste(ol,"CgBiasCorr_DU.eps",sep="")
outFileNameTable = paste(ol,"CgBiasCorr_DU.txt",sep="")
outFileNamePvalue = paste(ol,"CgBiasCorr_DU_pvalue.txt",sep="")
legPos = c(-2.5,0.89)
mainText = c(-1,0.91,"DU Datasets")
txtWid = 1.5
createGroupedCorrelation(matrix1,matrix2,du_labels,colorList,legPos,txtWid,mainText,outFileName,outFileNameTable,outFileNamePvalue)

# Naked datasets
nk_datasets = c("DU_NakedK562","DU_NakedMcf7","UW_IMR90")
nk_labels = c("K562(DU)","MCF-7(DU)","IMR90(UW)")
matrix1 = avgbiasData[,nk_datasets]
matrix2 = kmerCgData[,nk_datasets]
colorList = c("#000000FF", "#00FFFFFF", "#00FF40FF", "#FFBF00FF", "#FF0000FF", "#336611FF", "#0040FFFF")
outFileName = paste(ol,"CgBiasCorr_Naked.eps",sep="")
outFileNameTable = paste(ol,"CgBiasCorr_Naked.txt",sep="")
outFileNamePvalue = paste(ol,"CgBiasCorr_Naked_pvalue.txt",sep="")
legPos = c(-3,.585)
mainText = c(-0.8,.595,"Naked DNA Datasets")
txtWid = 1.8
createGroupedCorrelation(matrix1,matrix2,nk_labels,colorList,legPos,txtWid,mainText,outFileName,outFileNameTable,outFileNamePvalue)

# UW datasets
uw_datasets = c("UW_H7hesc","UW_HepG2","UW_Huvec","UW_K562","UW_m3134_with_DEX")
uw_labels = c("H7-hESC","HepG2","Huvec","K562","m3134")
matrix1 = avgbiasData[,uw_datasets]
matrix2 = kmerCgData[,uw_datasets]
colorList = c("#000000FF", "#00FFFFFF", "#00FF40FF", "#FFBF00FF", "#FF0000FF", "#336611FF", "#0040FFFF")
outFileName = paste(ol,"CgBiasCorr_UW.eps",sep="")
outFileNameTable = paste(ol,"CgBiasCorr_UW.txt",sep="")
outFileNamePvalue = paste(ol,"CgBiasCorr_UW_pvalue.txt",sep="")
legPos = c(-3.2,.593)
mainText = c(-1,.6,"UW Datasets")
txtWid = 2.0
createGroupedCorrelation(matrix1,matrix2,uw_labels,colorList,legPos,txtWid,mainText,outFileName,outFileNameTable,outFileNamePvalue)


