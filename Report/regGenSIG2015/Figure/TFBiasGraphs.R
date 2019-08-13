
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
# Input
###########################################################

# Input
loc = '/home/egg/Projects/TfbsPrediction/Results/StatisticalTests/TFSpecificBias'

# Parameters
mar.default = c(5,5,4,2)
standardize_values = FALSE
flag_Methods = FALSE
flag_Method_Counts = FALSE
flag_Method_Counts_All = TRUE

###########################################################
# Correlation Function
###########################################################

# Creating correlation graphs for multiple entries (version 2)
createGroupedCorrelation <- function(matrix1, matrix2, labelMatrix, outFileName, xLab, yLab, legendLab, xLim, yLim){

  # Plotting parameters
  #colorList = rainbow(ncol(matrix1))
  # Hack for better 8-color
  #colorList[3] = colorList[2]
  #colorList[2] = "#000000"
  ##temp = colorList[6]
  ##colorList[6] = colorList[3]
  ##colorList[3] = temp
  #colorList[7] = "#FF6600"
  #print(colorList)
  colorList = rev(c("#FF6600FF", "#000000FF", "#00FFFFFF", "#FF00BFFF",
                "#0040FFFF", "#FFBF00FF", "#FF0000FF", "#00FF40FF"))

  # Creating color matrix
  COLOR = matrix(rep("#000000",nrow(matrix1)*length(colorList)),nrow=nrow(matrix1),ncol=length(colorList))
  for (i in 1:length(colorList)){
    COLOR[,i] = rep(colorList[i],nrow(COLOR))
  }

  # Hack to output only the names of the top delta HINT(BC) - HINT TFs
  selectTF = c("NRF1","EGR1")
  newLabelMatrix = labelMatrix
  for (i in 1:nrow(newLabelMatrix)){
    for (j in 1:ncol(newLabelMatrix)){
      myFlag = FALSE
      for (stf in selectTF){
        if(stf == newLabelMatrix[i,j]){
          myFlag = TRUE
          break
        }
      }
      if(myFlag == FALSE){
        newLabelMatrix[i,j] = ""
        #for (k in 1:length(colorList)){
        #  COLOR[i,k] = alpha(COLOR[i,k],0.3)
        #}
      }
      else {
        #for (k in 1:length(colorList)){
        #  COLOR[i,k] = alpha(COLOR[i,k],1.0)
        #}
      }
    }
  }

  # Initializing graph
  postscript(outFileName, width=6.0, height=7.0, horizontal=FALSE, paper='special')
  #pdf(outFileName, width=7.0, height=6.0)
  mar.default2 = c(4,4.2,4,1)
  par(mar=mar.default2)

  # Calculating first correlation line
  dataFr = data.frame(X=matrix1[,1], Y=matrix2[,1])
  regLine = lm(Y~X, data=dataFr) # Regression line (y,x)

  # Creating plot with first correlation
  #plot(matrix1[,1], matrix2[,1], xlab=xLab, ylab=yLab, cex.lab=1.5, cex.axis=1.2, cex.main=0.1, cex.sub=1.0,
  #     col = COLOR[,1], pch = 19, xlim=xLim, ylim=yLim)
  plot(matrix1[,1], matrix2[,1], xlab=xLab, ylab=yLab, cex.lab=1.5, cex.axis=1.2, cex.main=0.1, cex.sub=1.0,
       col = COLOR[,1], pch = 19, xlim=xLim, ylim=yLim, xaxt="n", yaxt="n")
  axis(1, at=c(0,0.2,0.4,0.6,0.8,1.0), labels=c("","","","","",""))
  axis(2, at=c(0,0.5,1.0,1.5), labels=c("","","",""))
  #text(matrix1[,1], matrix2[,1], labels=newLabelMatrix[,1], cex= 0.4, pos = 4, col = colorList[1], offset=0.3)
  abline(regLine, lty=1, lwd=3.0, col=colorList[1])

  # Ploting rest of data
  for (col in 2:ncol(matrix1)){
    dataFr = data.frame(X=matrix1[,col], Y=matrix2[,col])
    regLine = lm(Y~X, data=dataFr) # Regression line (y,x)
    points(matrix1[,col], matrix2[,col], col = COLOR[,col], pch = 19)
    #text(matrix1[,col], matrix2[,col], labels=newLabelMatrix[,col], cex= 0.4, pos = 4, col = colorList[col], offset=0.3)
    abline(regLine, lty=1, lwd=3.0, col=colorList[col])
  }

  # Legend
  legend(-0.06,1.88, legend=legendLab, lty=NA, col=colorList, xpd = TRUE, ncol=4, bty = "n", lwd=2.0, cex=1.2,
         pch = 19,  x.intersp = -0.3)

  # End of graph
  grid()
  dev.off()
  system(paste("epstopdf",outFileName,sep=" "))
  #system(paste("rm",outFileName,sep=" "))

}

# Function to standardize values
standardize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}


# AUC Type Parameters
aucTypeList = c("10AUC")

# AUC Type Loop
for (j in 1:length(aucTypeList)){

  # Input Parameters
  aucType = aucTypeList[j]
  inList = c("DU_K562")
  bigMatrixAll1 = matrix(ncol=8)
  bigMatrixAll2 = matrix(ncol=8)
  bigTFLabelList = matrix(ncol=8)

  # Input Loop
  for (i in 1:length(inList)){

    # Parameters
    cell = inList[i]
    inFile = paste(loc,"/Pearson_AUC_Table/",cell,"_",aucType,".txt",sep="")
  
    # Reading data
    data = read.table(inFile, sep='\t', header=T)
    tfLabelList = data[1:nrow(data),1]

    # Count AUC Parameters
    countNameList = c("TC")
    countIndexList = c(3)

    # Count AUC Loop
    for (j in 1:length(countNameList)){

      # Method parameters
      countName = countNameList[j]
      countIndex = countIndexList[j]

      # Methods in increasing order according to Friedman Nemenyi
      methodList = rev(c("FOS","PWM","CUELLAR","PIQUE","NEPH","BOYLE","HMMD","HMMD_BC"))
      legendList = rev(c("FOS","PWM","Cuellar","Centipede","Neph","Boyle","HINT","HINT(BC)"))
      methodIndexList = rev(c(4,5,12,13,11,10,8,9))

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
        if(standardize_values){
          vec1 = standardize(data[,2]) # Pearson Correlation
          vec2 = standardize(data[,methodIndex]/data[,countIndex]) # Method AUC / Count AUC
        }
        else{
          vec1 = data[,2] # Pearson Correlation
          vec2 = data[,methodIndex]/data[,countIndex] # Method AUC / Count AUC
        }

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
  bigMatrixAll1 = bigMatrixAll1[2:nrow(bigMatrixAll1),]
  bigMatrixAll2 = bigMatrixAll2[2:nrow(bigMatrixAll2),]
  bigTFLabelList = bigTFLabelList[2:nrow(bigTFLabelList),]

  #print(colMeans(bigMatrixAll2))
  #print(apply(bigMatrixAll2, 2, sd))

  # Creating big grouped graph
  #outLoc = "/home/egg/Projects/TfbsPrediction/Report/regGenSIG2015/Figure/"
  #createGroupedCorrelation(bigMatrixAll1, bigMatrixAll2, bigTFLabelList, paste(outLoc,"/",aucType,"_withLabel.eps",sep=""), 
  #                         "Observed vs Bias Signal (OBS)", paste("Method AUC / TC AUC",sep=""), legendList, c(0.0,1.0), c(0.0,1.6))
  legendList = c("               "," "," "," "," "," "," "," ")
  createGroupedCorrelation(bigMatrixAll1, bigMatrixAll2, bigTFLabelList, paste(outLoc,"/",aucType,"_woLabel.eps",sep=""), 
                           "", paste("",sep=""), legendList, c(0.0,1.0), c(0.0,1.6))

}


