
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
loc = '/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias'

# Parameters
mar.default = c(5,5,4,2)
standardize_values = FALSE
flag_Methods = FALSE
flag_Method_Counts = FALSE
flag_Method_Counts_All = TRUE

###########################################################
# Correlation Function
###########################################################

# Creating Correlation Function
createCorrelation <- function(vec1, vec2, labelVec, outFileName, xlab, ylab, xLim, yLim){

  # Calculating correlation
  dataFr = data.frame(X=vec1, Y=vec2)
  regLine = lm(Y~X, data=dataFr) # Regression line (y,x)
  corrTest = cor.test(vec1, vec2, alternative = "two.sided", method = "pearson", conf.level = 0.95) # Correlation
  pValue = corrTest$p.value
  corr = corrTest$estimate

  # Plotting graph
  postscript(outFileName, width=7.0, height=6.0, horizontal=FALSE, paper='special')
  par(mar=mar.default)
  plot(vec1, vec2, xlab=xlab, ylab=ylab,
       main=paste('Correlation = ',round(corr,digits = 4),' / p-value = ',round(pValue,digits = 4),sep=''),
       cex.lab=2.0, cex.axis=1.5, cex.main=1.7, cex.sub=2.0, xlim=xLim, ylim=yLim)
  #text(vec1, vec2, labels=labelVec, cex= 0.3, pos = 3)
  abline(regLine,lty=1,lwd=3.0,col="black")
  grid()
  dev.off()
  system(paste("epstopdf",outFileName,sep=" "))
  system(paste("rm",outFileName,sep=" "))

}

# Creating correlation graphs for multiple entries
createGroupedCorrelation <- function(matrix1, matrix2, labelMatrix, outFileName, xLab, yLab, legendLab, xLim, yLim){

  # Plotting parameters
  #colorList = rainbow(ncol(matrix1))
  # Hack for better 8-color
  #colorList[3] = colorList[2]
  #colorList[2] = "#000000"
  colorList = c("#FF6600FF", "#000000FF", "#FF0000FF", "#00FF40FF", "#00FFFFFF", "#FF00BFFF",
                "#0040FFFF", "#FFBF00FF")

  # Calculating correlation coefficients
  pList = c()
  cList = c()
  for (col in 1:ncol(matrix1)){
    corrTest = cor.test(matrix1[,col], matrix2[,col], alternative = "two.sided", method = "pearson", conf.level = 0.95) # Correlation
    pList = c(pList,corrTest$p.value)
    cList = c(cList,corrTest$estimate)
  }

  # Initializing graph
  postscript(outFileName, width=7.0, height=6.0, horizontal=FALSE, paper='special')
  par(mar=mar.default)

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
    #legendLabel = c(legendLabel,paste(legendLab[ll],", R=",round(cList[ll],digits=4),", p-value=",round(pList[ll],digits=4),sep=""))
    legendLabel = c(legendLabel,paste(legendLab[ll],", R=",round(cList[ll],digits=4),sep=""))
  }
  #legend(0.0,2.55, legend=legendLabel, lty=1, col=colorList, xpd = TRUE, ncol=2, bty = "n", lwd=10.0, cex=1.0,
  #       text.width = 0.4)
  legend(0.0,1.9, legend=legendLabel, lty=1, col=colorList, xpd = TRUE, ncol=2, bty = "n", lwd=10.0, cex=1.0,
         text.width = 0.4)
  #legend(-0.25,1.94, legend=legendLabel, lty=1, col=colorList, xpd = TRUE, ncol=2, bty = "n", lwd=10.0, cex=1.0,
  #       text.width = 0.5)

  # End of graph
  grid()
  dev.off()
  system(paste("epstopdf",outFileName,sep=" "))
  system(paste("rm",outFileName,sep=" "))

}

# Creating correlation graphs for multiple entries (version 2)
createGroupedCorrelation2 <- function(matrix1, matrix2, labelMatrix, outFileName, xLab, yLab, legendLab, xLim, yLim){

  # Plotting parameters
  #colorList = rainbow(ncol(matrix1))
  # Hack for better 8-color
  #colorList[3] = colorList[2]
  #colorList[2] = "#000000"
  colorList = c("#FF6600FF", "#000000FF", "#00FFFFFF", "#FF00BFFF",
                "#0040FFFF", "#FFBF00FF", "#FF0000FF", "#00FF40FF")

  # Creating color matrix
  COLOR = matrix(rep("#000000",nrow(matrix1)*length(colorList)),nrow=nrow(matrix1),ncol=length(colorList))
  for (i in 1:length(colorList)){
    COLOR[,i] = rep(colorList[i],nrow(COLOR))
  }

  # Hack to output only the names of the top delta HINT(BC) - HINT TFs
  selectTF = c("NRF1","E2F4","EGR1","SP2","SP1")
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
        for (k in 1:length(colorList)){
          COLOR[i,k] = alpha(COLOR[i,k],0.3)
        }
      }
      else {
        for (k in 1:length(colorList)){
          COLOR[i,k] = alpha(COLOR[i,k],1.0)
        }
      }
    }
  }

  # Initializing graph
  #postscript(outFileName, width=8.0, height=6.0, horizontal=FALSE, paper='special')
  pdf(outFileName, width=8.0, height=6.0)
  mar.default2 = c(7,5,1,1)
  par(mar=mar.default2)

  # Calculating first correlation line
  dataFr = data.frame(X=matrix1[,1], Y=matrix2[,1])
  regLine = lm(Y~X, data=dataFr) # Regression line (y,x)

  # Creating plot with first correlation
  #plot(matrix1[,1], matrix2[,1], xlab=xLab, ylab=yLab, cex.lab=1.5, cex.axis=1.3, cex.main=0.1, cex.sub=1.0,
  #     col = colorList[1], pch = 19, xlim=xLim, ylim=yLim)
  plot(matrix1[,1], matrix2[,1], xlab=xLab, ylab=yLab, cex.lab=1.5, cex.axis=1.3, cex.main=0.1, cex.sub=1.0,
       col = COLOR[,1], pch = 19, xlim=xLim, ylim=yLim)
  #text(matrix1[,1], matrix2[,1], labels=newLabelMatrix[,1], cex= 0.4, pos = 4, col = colorList[1], offset=0.3)
  abline(regLine, lty=1, lwd=3.0, col=alpha(colorList[1],0.3))

  # Ploting rest of data
  for (col in 2:ncol(matrix1)){
    dataFr = data.frame(X=matrix1[,col], Y=matrix2[,col])
    regLine = lm(Y~X, data=dataFr) # Regression line (y,x)
    points(matrix1[,col], matrix2[,col], col = COLOR[,col], pch = 19)
    #text(matrix1[,col], matrix2[,col], labels=newLabelMatrix[,col], cex= 0.4, pos = 4, col = colorList[col], offset=0.3)
    abline(regLine, lty=1, lwd=3.0, col=alpha(colorList[col],0.3))
  }

  # Legend
  legend(0.1,-0.4, legend=legendLab, lty=1, col=colorList, xpd = TRUE, ncol=4, bty = "y", lwd=10.0, cex=0.8, text.width = 0.12)

  # End of graph
  grid()
  dev.off()
  #system(paste("epstopdf",outFileName,sep=" "))
  #system(paste("rm",outFileName,sep=" "))

}

createGroupedCorrelation3 <- function(matrix1, matrix2, labelMatrix, outFileName, xLab, yLab, legendLab, xLim, yLim){

  # Plotting parameters
  #colorList = rainbow(ncol(matrix1))
  # Hack for better 8-color
  #colorList[3] = colorList[2]
  #colorList[2] = "#000000"
  colorList = c("#FF6600FF", "#000000FF", "#00FFFFFF", "#FF00BFFF",
                "#0040FFFF", "#FFBF00FF", "#FF0000FF", "#00FF40FF")

  # Calculating correlation coefficients
  pList = c()
  cList = c()
  for (col in 1:ncol(matrix1)){
    corrTest = cor.test(matrix1[,col], matrix2[,col], alternative = "two.sided", method = "pearson", conf.level = 0.95) # Correlation
    pList = c(pList,corrTest$p.value)
    cList = c(cList,corrTest$estimate)
  }

  # Initializing graph
  postscript(outFileName, width=7.0, height=6.0, horizontal=FALSE, paper='special')
  par(mar=mar.default)

  # Calculating first correlation line
  dataFr = data.frame(X=matrix1[,1], Y=matrix2[,1])
  regLine = lm(Y~X, data=dataFr) # Regression line (y,x)

  # Creating plot with first correlation
  plot(matrix1[,1], matrix2[,1], xlab=xLab, ylab=yLab, cex.lab=2.0, cex.axis=1.5, cex.main=1.7, cex.sub=2.0,
       col = colorList[1], pch = 19, xlim=xLim, ylim=yLim)
  text(matrix1[,1], matrix2[,1], labels=labelMatrix[,1], cex= 0.3, pos = 3, col = colorList[1])
  abline(regLine, lty=1, lwd=3.0, col=colorList[1])

  # Ploting rest of data
  for (col in 2:ncol(matrix1)){
    dataFr = data.frame(X=matrix1[,col], Y=matrix2[,col])
    regLine = lm(Y~X, data=dataFr) # Regression line (y,x)
    points(matrix1[,col], matrix2[,col], col = colorList[col], pch = 19)
    text(matrix1[,col], matrix2[,col], labels=labelMatrix[,col], cex= 0.3, pos = 3, col = colorList[col])
    abline(regLine, lty=1, lwd=3.0, col=colorList[col])
  }

  # Legend
  legendLabel = c()
  for (ll in 1:length(legendLab)){
    #legendLabel = c(legendLabel,paste(legendLab[ll],", R=",round(cList[ll],digits=4),", p-value=",round(pList[ll],digits=4),sep=""))
    legendLabel = c(legendLabel,paste(legendLab[ll],", R=",round(cList[ll],digits=4),sep=""))
  }
  #legend(0.0,2.55, legend=legendLabel, lty=1, col=colorList, xpd = TRUE, ncol=2, bty = "n", lwd=10.0, cex=1.0,
  #       text.width = 0.4)
  legend(0.0,2.04, legend=legendLabel, lty=1, col=colorList, xpd = TRUE, ncol=2, bty = "n", lwd=10.0, cex=1.0,
         text.width = 0.4)
  #legend(-0.25,1.94, legend=legendLabel, lty=1, col=colorList, xpd = TRUE, ncol=2, bty = "n", lwd=10.0, cex=1.0,
  #       text.width = 0.5)

  # End of graph
  grid()
  dev.off()
  system(paste("epstopdf",outFileName,sep=" "))
  system(paste("rm",outFileName,sep=" "))

}

# Function to standardize values
standardize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

###########################################################
# Methods
###########################################################

if(flag_Methods){

# Input Parameters
inList = c("DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","UW_HepG2","UW_Huvec","UW_K562")

# Input Loop
for (i in 1:length(inList)){

  # AUC Type Parameters
  aucTypeList = c("ALLAUC","10AUC","HMMAUC","BIASAUC")

  # AUC Type Loop
  for (j in 1:length(aucTypeList)){

    # Parameters
    cell = inList[i]
    aucType = aucTypeList[j]
    inFile = paste(loc,"/Pearson_AUC_Table/",cell,"_",aucType,".txt",sep="")
    outLoc = paste(loc,"/TFBiasGraphs/Methods/",cell,sep="")
    system(paste("mkdir -p",outLoc,sep=" "))
  
    # Reading data
    data = read.table(inFile, sep='\t', header=T)
    tfLabelList = data[,1]

    # Method parameters
    if(cell == "DU_H1hesc" || cell == "DU_K562"){
      #methodList = c("TC","FS","PWM","HMM","HMM_BC","HINT","HINT-BC","BOYLE","NEPH","CUELLAR","PIQUE")
      methodList = c("TC","FS","PWM","HINT","HINT-BC","BOYLE","NEPH","CUELLAR","PIQUE")
      methodIndexList = c(3,4,5,8,9,10,11,12,13)
    }
    else{
      #methodList = c("TC","FS","PWM","HMM","HMM_BC","HINT","HINT-BC")
      methodList = c("TC","FS","PWM","HINT","HINT-BC")
      methodIndexList = c(3,4,5,8,9)
    }

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
        vec2 = standardize(data[,methodIndex]) # Method AUC
      }
      else{
        vec1 = data[,2] # Pearson Correlation
        vec2 = data[,methodIndex] # Method AUC
      }
 
      # Creating individual correlation graph
      createCorrelation(vec1, vec2, tfLabelList, paste(outLoc,"/",methodName,"_",aucType,".eps",sep=""),
                        "Observed vs. Bias Signal (OBS)", paste(methodName,"AUC",sep=" "), c(0.0,1.0), c(0.0,1.0))

      # Parameters for grouped graph
      matrix1 = cbind(matrix1,vec1)
      matrix2 = cbind(matrix2,vec2)
      labelMatrix = cbind(labelMatrix,as.character(tfLabelList))

    }

    # Creating grouped graph
    createGroupedCorrelation(matrix1, matrix2, labelMatrix, paste(outLoc,"/MethodAUC_",aucType,".eps",sep=""), 
                           "Observed vs. Bias Signal (OBS)", "Method AUC", methodList, c(0.0,1.0), c(0.0,1.0))

  }

}

}

###########################################################
# Methods_Counts
###########################################################

if(flag_Method_Counts){

# Input Parameters
inList = c("DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","UW_HepG2","UW_Huvec","UW_K562")

# Input Loop
for (i in 1:length(inList)){

  # AUC Type Parameters
  aucTypeList = c("ALLAUC","10AUC","HMMAUC","BIASAUC")

  # AUC Type Loop
  for (j in 1:length(aucTypeList)){

    # Parameters
    aucType = aucTypeList[j]
    cell = inList[i]
    inFile = paste(loc,"/Pearson_AUC_Table/",cell,"_",aucType,".txt",sep="")
    outLoc = paste(loc,"/TFBiasGraphs/Methods_Counts/",cell,sep="")
    system(paste("mkdir -p",outLoc,sep=" "))
  
    # Reading data
    data = read.table(inFile, sep='\t', header=T)
    tfLabelList = data[,1]

    # Count AUC Parameters
    #countNameList = c("DNase","D13","DNaseS","D13S")
    #countIndexList = c(3,4,5,6)
    countNameList = c("TC")
    countIndexList = c(3)

    # Count AUC Loop
    for (j in 1:length(countNameList)){

      # Method parameters
      countName = countNameList[j]
      countIndex = countIndexList[j]
      if(cell == "DU_H1hesc" || cell == "DU_K562"){
        #methodList = c("FS","PWM","HMM","HMM_BC","HINT","HINT-BC","BOYLE","NEPH","CUELLAR","PIQUE")
        methodList = c("FS","PWM","HINT","HINT-BC","BOYLE","NEPH","CUELLAR","PIQUE")
        methodIndexList = c(4,5,8,9,10,11,12,13)
      }
      else{
        #methodList = c("FS","PWM","HMM","HMM_BC","HINT","HINT-BC")
        methodList = c("FS","PWM","HINT","HINT-BC")
        methodIndexList = c(4,5,8,9)
      }

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
          yLim = c(0.0,1.0)
        }
        else{
          vec1 = data[,2] # Pearson Correlation
          vec2 = data[,methodIndex]/data[,countIndex] # Method AUC / Count AUC
          yLim = c(0.0,1.55)
        }
 
        # Create individual correlation graph
        createCorrelation(vec1, vec2, tfLabelList, paste(outLoc,"/",methodName,"_",countName,"_",aucType,".eps",sep=""),
                          "Observed vs. Bias Signal (OBS)", paste(methodName," AUC / ",countName," AUC",sep=""), c(0.0,1.0), yLim)

        # Parameters for grouped graph
        matrix1 = cbind(matrix1,vec1)
        matrix2 = cbind(matrix2,vec2)
        labelMatrix = cbind(labelMatrix,as.character(tfLabelList))

      }

      # Creating grouped graph
      createGroupedCorrelation(matrix1, matrix2, labelMatrix, paste(outLoc,"/",countName,"_",aucType,".eps",sep=""), 
                               "Observed vs. Bias Signal (OBS)", paste("Method AUC / ",countName," AUC",sep=""), methodList, c(0.0,1.0), yLim)

    }

  }

}

}

###########################################################
# Methods_Counts_All
###########################################################

if(flag_Method_Counts_All){

# AUC Type Parameters
aucTypeList = c("10AUC")

# AUC Type Loop
for (j in 1:length(aucTypeList)){

  # Input Parameters
  aucType = aucTypeList[j]  
  #inList = c("DU_H1hesc","DU_HeLaS3","DU_Huvec","DU_K562","UW_HepG2","UW_Huvec","UW_K562")
  inList = c("DU_H1hesc","DU_K562","DU_Mcf7","UW_LnCaP","UW_m3134_with_DEX")
  #inList = c("DU_H1hesc","DU_K562")
  bigMatrixAll1 = matrix(ncol=5)
  bigMatrixAll2 = matrix(ncol=5)
  bigTFLabelList = matrix(ncol=5)
  #bigMatrixAll1 = matrix(ncol=4)
  #bigMatrixAll2 = matrix(ncol=4)
  #bigTFLabelList = matrix(ncol=4)

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
      #methodList = c("FS","PWM","CUELLAR","PIQUE","NEPH","BOYLE","HINT","HINT-BC","HINT-BCN")
      #legendList = c("FS","PWM","Cuellar","Centipede","Neph","Boyle","HINT","HINT-BC","HINT-BCN")
      #methodIndexList = c(4,5,11,12,10,9,6,7,8)

      methodList = c("FS","PWM","HINT","HINT-BC","HINT-BCN")
      legendList = c("FS","PWM","HINT","HINT-BC","HINT-BCN")
      methodIndexList = c(4,5,6,7,8)

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
  bigTFLabelList = bigTFLabelList[2:nrow(bigTFLabelList),]

  # Creating big grouped graph
  outLoc = paste(loc,"/TFBiasGraphs_NM/Methods_Counts/",sep="")
  createGroupedCorrelation3(bigMatrixAll1, bigMatrixAll2, bigTFLabelList, paste(outLoc,"/",aucType,".eps",sep=""), 
                           "Observed vs. Bias Signal (OBS)", paste("Method AUC / TC AUC",sep=""), legendList, c(0.0,1.0), c(0.0,1.6))

}

}


