
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

createGroupedCorrelation <- function(matrix1, matrix2, labelMatrix, outFileName, xLab, yLab, legendLab, xLim, yLim,colorList,myMar,legY,ntfs,outFileNameTable,outFileNamePvalue){

  # Calculating correlation coefficients
  pList = c()
  cList = c()
  for (col in 1:ncol(matrix1)){
    corrTest = cor.test(matrix1[,col], matrix2[,col], alternative = "two.sided", method = "pearson", conf.level = 0.95) # Correlation
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
  postscript(outFileName, width=8.0, height=6.0, horizontal=FALSE, paper='special')
  par(mar=myMar)

  # Calculating first correlation line
  dataFr = data.frame(X=matrix1[,1], Y=matrix2[,1])
  regLine = lm(Y~X, data=dataFr) # Regression line (y,x)

  # Creating plot with first correlation
  plot(matrix1[,1], matrix2[,1], xlab=xLab, ylab=yLab, cex.lab=2.0, cex.axis=1.5, cex.main=1.7, cex.sub=2.0,
       col = colorList[1], pch = 19, xlim=xLim, ylim=yLim, xaxt="n", yaxt="n")
  axis(1, at=c(0,0.2,0.4,0.6,0.8,1.0), labels=c("","","","","",""))
  axis(2, at=c(0,0.5,1.0,1.5), labels=c("","","",""))
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
  legendLabel = rep(" ",length(legendLab)+1)
  #for (ll in 1:length(legendLab)){
  #  legendLabel = c(legendLabel,paste(legendLab[ll],", R=",round(cList[ll],digits=4),sep=""))
  #}
  #legendLabel = c(legendLabel[newIndexList],paste("Number of TFs = ",ntfs,sep=""))
  colorList = c(colorList[newIndexList],"NA")
  legend(1.03,legY, legend=legendLabel, lty=NA, col=colorList, xpd = TRUE, ncol=1, bty = "n", lwd=5.0, cex=2.0,
         text.width = 0.32, pch = 19)

  # End of graph
  grid()
  dev.off()
  system(paste("epstopdf",outFileName,sep=" "))
  #system(paste("rm",outFileName,sep=" "))

  # Writing graph in tabular form
  toWrite = c("FACTOR","METHOD","OBSERVED VS PREDICTED BIAS SCORE (OBS)","METHOD AUC / TC AUC")
  for(j in 1:length(legendLab)){
    for(i in 1:nrow(matrix1)){
      toWrite = c(toWrite,labelMatrix[i,j],legendLab[j],matrix1[i,j],matrix2[i,j])
    }
  }
  write(toWrite,file=outFileNameTable,ncolumns=4,append=FALSE,sep='\t')

}

correlationGraph <- function(aucType,cellList,methodList,legendList,methodIndexList,colorList,inLoc,outFileName,myMar,legY,ntfs,outFileNameTable,outFileNamePvalue){

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
  bigMatrixAll1 = bigMatrixAll1[2:nrow(bigMatrixAll1),]
  bigMatrixAll2 = bigMatrixAll2[2:nrow(bigMatrixAll2),]

  # Creating big grouped graph
  createGroupedCorrelation(bigMatrixAll1, bigMatrixAll2, bigTFLabelList, outFileName, 
                           "", paste("",sep=""), legendList, c(0.0,1.0), c(0.0,1.6),
                           colorList,myMar,legY,ntfs,outFileNameTable,outFileNamePvalue)

}

###########################################################
# Execution
###########################################################

# Parameters
inLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson_AUC_Table/"
ol = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFBiasGraphs/Figure/"

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
outFileNameTable = paste(ol,"BiasCorr_AllMethods.txt",sep="")
outFileNamePvalue = paste(ol,"BiasCorr_AllMethods_corr.txt",sep="")
myMar = c(1,1,1,3) # bottom, left, top and right
legY = 1.7
ntfs = "88"
correlationGraph(aucType,cellList,methodList,legendList,methodIndexList,colorList,inLoc,
                 outFileName,myMar,legY,ntfs,outFileNameTable,outFileNamePvalue)


