
###########################################################################################################################################
# Import
###########################################################################################################################################

# Import
library(lattice)
library(reshape)
library(plotrix)
library(ggplot2)

###########################################################################################################################################
# Input
###########################################################################################################################################

# Input
loc = '/home/egg/Projects/TfbsPrediction/Results/StatisticalTests/TFSpecificBias'
ourLoc = paste(loc,'Pearson',sep='/')
heFileName = paste(loc,'Pearson_AUC_Table','UW_K562_HE.txt',sep='/')

# Parameters
flag_HePearson_OurPearson = FALSE
flag_OurPearson = FALSE
flag_dAUC_Pearson = TRUE

###########################################################################################################################################
# Correlation Function
###########################################################################################################################################

# Creating Correlation Function
# Formula are the name of the columns. It must be y~x
createCorrelation <- function(vec1, vec2, formula, data, outFileName, xlab, ylab){

  # Calculating correlation
  regLine = lm(formula, data=data) # Regression line (y,x)
  corrTest = cor.test(vec1, vec2, alternative = "two.sided", method = "pearson", conf.level = 0.95) # Correlation
  pValue = corrTest$p.value
  corr = corrTest$estimate

  # Plotting graph
  postscript(outFileName, width=7.0, height=6.0, horizontal=FALSE, paper='special')
  par(mar=c(5,5,4,2))
  plot(vec1, vec2, xlab=xlab, ylab=ylab,
       main=paste('Correlation = ',round(corr,digits = 4),' / p-value = ',round(pValue,digits = 4),sep=''),
       cex.lab=2.0, cex.axis=1.5, cex.main=1.7, cex.sub=2.0)
  #text(vec1, vec2, cex= 0.3, pos = 3)
  abline(regLine,lty=1,lwd=3.0,col="black")
  grid()
  dev.off()
  system(paste('epstopdf',outFileName,sep=' '))

  # Return correlation
  return(round(corr,digits = 6))

}

# Function to standardize values
standardize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}


###########################################################################################################################################
# HePearson vs OurPearson
###########################################################################################################################################
# Testing pearson correlations evaluated by He et al in UW_K562 against the correlation evaluated
# by our script in all datasets

if(flag_HePearson_OurPearson){

# Reading He et al correlations
dataHe = read.table(heFileName, sep='\t', header=T)
dataHe = dataHe[,c(1,2)] # Cutting table only for necessary contents
dataHe[,2] = standardize(dataHe[,2]) # Standardizing correlations

# Input Parameters
inList = c("DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","UW_HepG2","UW_Huvec","UW_K562")
outLoc = paste(loc,"CorrelationGraphs","HePearson_OurPearson",sep="/")
toWrite = c()

# Input Loop
for (i in 1:length(inList)) {

  # Reading our table
  inFileName = paste(ourLoc,inList[i],"pearson.txt",sep="/")
  dataOur = read.table(inFileName, sep='\t', header=T)
  dataOur = dataOur[,c(1,2)] # Cutting table only for necessary contents # 50 bp pearson
  #dataOur = dataOur[,c(1,4)] # Cutting table only for necessary contents # Motif len pearson
  dataOur[,2] = standardize(dataOur[,2]) # Standardizing correlations

  # Creating data frame with only the overlapping TFs
  vecHe = c()
  vecOur = c()
  for (k in 1:length(dataOur[,1])) {
    if(sum(dataHe[,1]==as.character(dataOur[k,1])) == 1){
      vecHe = c(vecHe,dataHe[dataHe[,1]==as.character(dataOur[k,1]),2])
      vecOur = c(vecOur,dataOur[k,2])
    }
  }
  #vecHe = standardize(vecHe)
  #vecOur = standardize(vecOur)
  dataFr = data.frame(X=vecHe, Y=vecOur)

  # Calculating correlation
  curr_corr = createCorrelation(dataFr[,1], dataFr[,2], Y~X, dataFr, paste(outLoc,"/",inList[i],".eps",sep=""), 
                                "He et al Pearson", paste(inList[i],"Pearson",sep=" "))
  toWrite = c(toWrite,inList[i],curr_corr)

}

# Writing correlation to file
write(toWrite,file=paste(outLoc,"/","correlation.txt",sep=""),ncolumns=2,append=FALSE,sep='\t')

}


###########################################################################################################################################
# OurPearson
###########################################################################################################################################
# Testing pearson correlations evaluated by our script between different datasets

if(flag_OurPearson){

# Input Parameters
inList = c("DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","UW_HepG2","UW_Huvec","UW_K562")
outLoc = paste(loc,"CorrelationGraphs","OurPearson",sep="/")
toWrite = c("",inList)

# Input Loop
for (i in 1:length(inList)) {

  # Writing correlation heading
  toWrite = c(toWrite,inList[i])

  for (j in 1:length(inList)) {

    # Parameters
    name1 = inList[i]
    name2 = inList[j]

    # Reading our tables
    inFileName1 = paste(ourLoc,name1,"pearson.txt",sep="/")
    data1 = read.table(inFileName1, sep='\t', header=T)
    data1 = data1[,c(1,2)] # Cutting table only for necessary contents # 50 bp pearson
    #data1 = data1[,c(1,4)] # Cutting table only for necessary contents # Motif len pearson
    data1[,2] = standardize(data1[,2]) # Standardizing correlations

    inFileName2 = paste(ourLoc,name2,"pearson.txt",sep="/")
    data2 = read.table(inFileName2, sep='\t', header=T)
    data2 = data2[,c(1,2)] # Cutting table only for necessary contents # 50 bp pearson
    #data2 = data2[,c(1,4)] # Cutting table only for necessary contents # Motif len pearson
    data2[,2] = standardize(data2[,2]) # Standardizing correlations

    # Creating data frame with only the overlapping TFs
    vec1 = c()
    vec2 = c()
    for (k in 1:length(data2[,1])) {
      if(sum(data1[,1]==as.character(data2[k,1])) == 1){
        vec1 = c(vec1,data1[data1[,1]==as.character(data2[k,1]),2])
        vec2 = c(vec2,data2[k,2])
      }
    }
    dataFr = data.frame(X=vec1, Y=vec2)

    # Calculating correlation
    curr_corr = createCorrelation(dataFr[,1], dataFr[,2], Y~X, dataFr, paste(outLoc,"/",name1,"_vs_",name2,".eps",sep=""), 
                                  paste(name1,"Pearson",sep=" "), paste(name2,"Pearson",sep=" "))
    toWrite = c(toWrite,curr_corr)

  }
}

# Writing correlation to file
write(toWrite,file=paste(outLoc,"/","correlation.txt",sep=""),ncolumns=length(inList)+1,append=FALSE,sep='\t')

}


###########################################################################################################################################
# Delta AUC vs. Pearson
###########################################################################################################################################

if(flag_dAUC_Pearson){

# Input Parameters
inList = c("DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","UW_HepG2","UW_Huvec","UW_K562")
outLoc = paste(loc,"CorrelationGraphs","dAUC_Pearson",sep="/")
toWrite = c()

# Input Loop
for (i in 1:length(inList)) {

  # Reading our table
  inFileName = paste(loc,"/Pearson_AUC_Table/",inList[i],"_10AUC.txt",sep="")
  dataOur = read.table(inFileName, sep='\t', header=T)
  dataOur = dataOur[,c(1,2,8,9)] # Factor, Pearson, HMM, HMM(BC)

  # Evaluating delta AUC
  dataOur[,3] = dataOur[,4] - dataOur[,3]
  #dataOur[,2] = standardize(dataOur[,2]) # Standardizing correlations
  #dataOur[,3] = standardize(dataOur[,3]) # Standardizing dAUC
  
  # Printing list of delta
  dataOur = dataOur[,1:3]
  dataOur = dataOur[order(-dataOur[,3]),]
  print(inList[i])
  print(dataOur)

  # Creating data frame
  dataFr = data.frame(X=dataOur[,2], Y=dataOur[,3])

  # Calculating correlation
  curr_corr = createCorrelation(dataFr[,1], dataFr[,2], Y~X, dataFr, paste(outLoc,"/",inList[i],".eps",sep=""), 
                                "Pearson Correlation", paste(inList[i]," delta AUC",sep=" "))
  toWrite = c(toWrite,inList[i],curr_corr)

}

# Writing correlation to file
write(toWrite,file=paste(outLoc,"/","correlation.txt",sep=""),ncolumns=2,append=FALSE,sep='\t')

}

