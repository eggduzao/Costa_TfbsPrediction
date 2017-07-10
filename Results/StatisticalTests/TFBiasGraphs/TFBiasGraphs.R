
###########################################################
# Import
###########################################################

# Import
library(lattice)
library(reshape)
library(plotrix)
library(ggplot2)
library(scales)

# Correlation method
#corMethod = "spearman" 
corMethod = "pearson"

###########################################################
# Correlation Function
###########################################################

createGroupedCorrelation <- function(matrix1, matrix2, labelMatrix, outFileName, xLab, yLab, legendLab, xLim, yLim,colorList,myMar,legY,ntfs,outFileNamePvalue){

  # Calculating correlation coefficients
  pList = c()
  cList = c()
  for (col in 1:ncol(matrix1)){
    corrTest = cor.test(matrix1[,col], matrix2[,col], alternative = "two.sided", method = corMethod, conf.level = 0.95) # Correlation
    pList = c(pList,corrTest$p.value)
    cList = c(cList,corrTest$estimate)
  }

  # Writing correlation and p-value
  newIndexList = sort(cList,index.return=TRUE)$ix
  toWrite = c("METHOD","CORRELATION","P-VALUE")
  for (i in newIndexList){
    toWrite = c(toWrite,legendLab[i],cList[i],pList[i])
  }
  write(toWrite,file=outFileNamePvalue,ncolumns=3,append=FALSE,sep='\t')

  # Initializing graph
  postscript(outFileName, width=7.0, height=6.0, horizontal=FALSE, paper='special')
  par(mar=myMar)

  # Calculating first correlation line
  dataFr = data.frame(X=matrix1[,1], Y=matrix2[,1])
  regLine = lm(Y~X, data=dataFr) # Regression line (y,x)

  # Creating plot with first correlation
  plot(matrix1[,1], matrix2[,1], xlab=xLab, ylab=yLab, cex.lab=2.0, cex.axis=1.5, cex.main=1.7, cex.sub=2.0,
       col = colorList[1], pch = 19, xlim=xLim, ylim=yLim)
  #text(matrix1[,1], matrix2[,1], labels=labelMatrix[,1], cex= 0.3, pos = 3, col = colorList[1])
  abline(regLine, lty=1, lwd=3.0, col=colorList[1])

  # Ploting rest of data
  for (col in 2:ncol(matrix1)){
    dataFr = data.frame(X=matrix1[,col], Y=matrix2[,col])
    regLine = lm(Y~X, data=dataFr) # Regression line (y,x)
    points(matrix1[,col], matrix2[,col], col = colorList[col], pch = 19)
    #text(matrix1[,col], matrix2[,col], labels=labelMatrix[,col], cex= 0.3, pos = 3, col = colorList[col])
    abline(regLine, lty=1, lwd=3.0, col=colorList[col])
  }

  # Legend
  legendLabel = c()
  for (ll in 1:length(legendLab)){
    legendLabel = c(legendLabel,paste(legendLab[ll],", R=",round(cList[ll],digits=4),sep=""))
  }
  legendLabel = c(legendLabel[newIndexList],paste("Number of TFs = ",ntfs,sep=""))
  colorList = c(colorList[newIndexList],"NA")
  legend(-0.25,legY, legend=legendLabel, lty=1, col=colorList, xpd = TRUE, ncol=3, bty = "n", lwd=5.0, cex=1.0,
         text.width = 0.32)

  # End of graph
  grid()
  dev.off()
  system(paste("epstopdf",outFileName,sep=" "))
  system(paste("rm",outFileName,sep=" "))

}

correlationGraph <- function(aucType,cellList,methodList,legendList,methodIndexList,colorList,inLoc,outFileName,myMar,legY,ntfs,outFileNamePvalue){

  # Initialization
  bigMatrixAll1 = matrix(ncol=length(methodList))
  bigMatrixAll2 = matrix(ncol=length(methodList))
  bigTFLabelList = matrix(ncol=length(methodList))
  countNameList = c("TC")
  countIndexList = c(3)

  # Input Loop
  for (i in 1:length(cellList)){

    # Parameters
    cell = cellList[i]
    inFile = paste(inLoc,cell,"_",aucType,".txt",sep="")
  
    # Reading data
    data = read.table(inFile, sep='\t', header=T)
    tfLabelList = data[1:nrow(data),1]

    # Count AUC Loop
    for (j in 1:length(countNameList)){

      # Method parameters
      countName = countNameList[j]
      countIndex = countIndexList[j]

      # Parameters for grouped graph
      matrix1 = c()
      matrix2 = c()
      labelMatrix = c()

      # Method Loop
      for (k in 1:length(methodList)){

        # Parameters
        methodName = methodList[k]
        methodIndex = methodIndexList[k]

        # Fetch input
        vec1 = data[,2] # Pearson Correlation
        vec2 = data[,methodIndex]/data[,countIndex] # Method AUC / Count AUC

        # Parameters for grouped graph
        matrix1 = cbind(matrix1,vec1)
        matrix2 = cbind(matrix2,vec2)
        labelMatrix = cbind(labelMatrix,as.character(tfLabelList))

      }

      # Parameters for big grouped graph
      bigMatrixAll1 = rbind(bigMatrixAll1,matrix1)
      bigMatrixAll2 = rbind(bigMatrixAll2,matrix2)
      bigTFLabelList = rbind(bigTFLabelList,labelMatrix)

    }

  }

  # For some reason the first row is only NAs
  bigTFLabelList = bigTFLabelList[2:nrow(bigTFLabelList),]

  # Creating big grouped graph
  createGroupedCorrelation(bigMatrixAll1, bigMatrixAll2, bigTFLabelList, outFileName, 
                           "Observed vs. Bias Signal (OBS)", paste("Method AUC / TC AUC",sep=""), 
                           legendList, c(0.0,1.0), c(0.0,1.6),colorList,myMar,legY,ntfs,outFileNamePvalue)

}

###########################################################
# Execution
###########################################################

# Parameters
inLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson_AUC_Table/"
ol = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFBiasGraphs/GraphsPearson/"
#inLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Spearman_AUC_Table/"
#ol = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFBiasGraphs/GraphsSpearman/"

# H1hesc (DU)
aucType = "10AUC"
cellList = c("DU_H1hesc")
methodList = c("FS","PWM","CUELLAR","FLR","PIQ","PIQUE","BINDNASE","NEPH","WELLINGTON",
               "DNASE2TF","BOYLE","HINT","HINT-BC","HINT-BCN")
legendList = c("FS","PWM","Cuellar","FLR","PIQ","Centipede","BinDNase","Neph","Wellington",
               "DNase2TF","Boyle","HINT","HINT-BC","HINT-BCN")
methodIndexList = c(4,5,11,14,15,12,17,10,16,13,9,6,7,8)
colorList = c("#FF6600FF", "#000000FF", "#00FFFFFF", "#662288FF", "#336611FF", "#FF00BFFF", "#CCCC99FF",
              "#0040FFFF", "#661111FF", "#FFFF00FF", "#FFBF00FF", "#FF0000FF", "#00FF40FF", "#888888FF")
outFileName = paste(ol,"BiasCorr_H1hesc_DU.eps",sep="")
outFileNamePvalue = paste(ol,"BiasCorr_H1hesc_DU.txt",sep="")
myMar = c(5,5,5,2) # bottom, left, top and right
legY = 2.14
ntfs = "29"
correlationGraph(aucType,cellList,methodList,legendList,methodIndexList,colorList,inLoc,outFileName,myMar,legY,ntfs,outFileNamePvalue)

# K562 (DU)
aucType = "10AUC"
cellList = c("DU_K562")
methodList = c("FS","PWM","CUELLAR","FLR","PIQ","PIQUE","BINDNASE","NEPH","WELLINGTON",
               "DNASE2TF","BOYLE","HINT","HINT-BC","HINT-BCN")
legendList = c("FS","PWM","Cuellar","FLR","PIQ","Centipede","BinDNase","Neph","Wellington",
               "DNase2TF","Boyle","HINT","HINT-BC","HINT-BCN")
methodIndexList = c(4,5,11,14,15,12,17,10,16,13,9,6,7,8)
colorList = c("#FF6600FF", "#000000FF", "#00FFFFFF", "#662288FF", "#336611FF", "#FF00BFFF", "#CCCC99FF",
              "#0040FFFF", "#661111FF", "#FFFF00FF", "#FFBF00FF", "#FF0000FF", "#00FF40FF", "#888888FF")
outFileName = paste(ol,"BiasCorr_K562_DU.eps",sep="")
outFileNamePvalue = paste(ol,"BiasCorr_K562_DU.txt",sep="")
myMar = c(5,5,5,2) # bottom, left, top and right
legY = 2.14
ntfs = "59"
correlationGraph(aucType,cellList,methodList,legendList,methodIndexList,colorList,inLoc,outFileName,myMar,legY,ntfs,outFileNamePvalue)

# K562 (UW)
aucType = "10AUC"
cellList = c("UW_K562")
methodList = c("FS","PWM","HINT","HINT-BC","HINT-BCN")
legendList = c("FS","PWM","HINT","HINT-BC","HINT-BCN")
methodIndexList = c(4,5,6,7,8)
colorList = c("#FF6600FF", "#000000FF", "#FF0000FF", "#00FF40FF", "#888888FF")
outFileName = paste(ol,"BiasCorr_K562_UW.eps",sep="")
outFileNamePvalue = paste(ol,"BiasCorr_K562_UW.txt",sep="")
myMar = c(5,5,3,2) # bottom, left, top and right
legY = 1.9
ntfs = "59"
correlationGraph(aucType,cellList,methodList,legendList,methodIndexList,colorList,inLoc,outFileName,myMar,legY,ntfs,outFileNamePvalue)

# All Methods
aucType = "10AUC"
cellList = c("DU_H1hesc","DU_K562")
methodList = c("FS","PWM","CUELLAR","FLR","PIQ","PIQUE","BINDNASE","NEPH","WELLINGTON",
               "DNASE2TF","BOYLE","HINT","HINT-BC","HINT-BCN")
legendList = c("FS","PWM","Cuellar","FLR","PIQ","Centipede","BinDNase","Neph","Wellington",
               "DNase2TF","Boyle","HINT","HINT-BC","HINT-BCN")
methodIndexList = c(4,5,11,14,15,12,17,10,16,13,9,6,7,8)
colorList = c("#FF6600FF", "#000000FF", "#00FFFFFF", "#662288FF", "#336611FF", "#FF00BFFF", "#CCCC99FF",
              "#0040FFFF", "#661111FF", "#FFFF00FF", "#FFBF00FF", "#FF0000FF", "#00FF40FF", "#888888FF")
outFileName = paste(ol,"BiasCorr_AllMethods.eps",sep="")
outFileNamePvalue = paste(ol,"BiasCorr_AllMethods.txt",sep="")
myMar = c(5,5,5,2) # bottom, left, top and right
legY = 2.14
ntfs = "88"
correlationGraph(aucType,cellList,methodList,legendList,methodIndexList,colorList,inLoc,outFileName,myMar,legY,ntfs,outFileNamePvalue)

# All Cells
aucType = "10AUC"
cellList = c("DU_H1hesc", "DU_HeLaS3", "DU_Huvec", "DU_K562", "DU_Mcf7",
             "UW_HepG2", "UW_Huvec", "UW_K562", "UW_LnCaP", "UW_m3134_with_DEX")
methodList = c("FS","PWM","HINT","HINT-BC","HINT-BCN")
legendList = c("FS","PWM","HINT","HINT-BC","HINT-BCN")
methodIndexList = c(4,5,6,7,8)
colorList = c("#FF6600FF", "#000000FF", "#FF0000FF", "#00FF40FF", "#888888FF")
outFileName = paste(ol,"BiasCorr_AllCells.eps",sep="")
outFileNamePvalue = paste(ol,"BiasCorr_AllCells.txt",sep="")
myMar = c(5,5,3,2) # bottom, left, top and right
legY = 1.9
ntfs = "209"
correlationGraph(aucType,cellList,methodList,legendList,methodIndexList,colorList,inLoc,outFileName,myMar,legY,ntfs,outFileNamePvalue)

# He et al
aucType = "10AUC"
cellList = c("HE")
methodList = c("FS","PWM","HINT","HINT-BC","HINT-BCN")
legendList = c("FS","PWM","HINT","HINT-BC","HINT-BCN")
methodIndexList = c(4,5,6,7,8)
colorList = c("#FF6600FF", "#000000FF", "#FF0000FF", "#00FF40FF", "#888888FF")
outFileName = paste(ol,"BiasCorr_HeReplication.eps",sep="")
outFileNamePvalue = paste(ol,"BiasCorr_HeReplication.txt",sep="")
myMar = c(5,5,3,2) # bottom, left, top and right
legY = 1.9
ntfs = "36"
correlationGraph(aucType,cellList,methodList,legendList,methodIndexList,colorList,inLoc,outFileName,myMar,legY,ntfs,outFileNamePvalue)


