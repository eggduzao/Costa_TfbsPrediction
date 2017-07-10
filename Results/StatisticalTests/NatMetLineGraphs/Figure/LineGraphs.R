
###################################################################################################
# IMPORT
###################################################################################################

# Import
library(lattice)
library(reshape)
library(plotrix)
library(ggplot2)

###################################################################################################
# INPUT
###################################################################################################

# Input Paths
inloc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias"
outloc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/NatMetLineGraphs/Figure"

# Graph Parameters
plot_colors_strand <- c("red", "blue")
plot_colors_logo <- c("#FF7777", "#006600")

# Data Parameters
cellList = c("DU_H1hesc","UW_LnCaP","DU_Mcf7","UW_m3134_with_DEX")

###################################################################################################
# LINE PLOT FUNCTION
###################################################################################################

createLinePlot <- function(originalF,originalR,biasF,biasR,corrF,corrR,title,nsites,outFileName,outputFileNameTxt){

  ###############################################
  # Initialization
  ###############################################

  # Initialization
  postscript(outFileName, width=6.0, height=6.0, horizontal=FALSE, paper="special")
  par(mfrow=c(2,1), mar=c(1,1,1,1), cex.axis=1.2, cex.lab=2.0, cex.main=2.0, xpd=TRUE) #bottom, left, top and right
  xVec = (-length(originalF)/2):((length(originalF)/2)-1)
  toWrite = c("X axis",xVec)

  ###############################################
  # Bias plot
  ###############################################

  # Min/Max Values
  minValue = signif(min(c(biasF,biasR)),digits = 2)
  maxValue = signif(max(c(biasF,biasR)),digits = 2)

  # Plot forward signal / Initialize plot
  par(mar=c(1,1,1,1)) # bottom, left, top and right
  plot(xVec, biasF, type="l", col=plot_colors_strand[1], xlab="", ylab="", lwd=2, 
       xlim=c(xVec[1],xVec[length(xVec)]), ylim=c(minValue,maxValue), main="", lty=1, axes = FALSE)

  # Plot reverse signal
  lines(xVec, biasR, type="l", lwd=2, col=plot_colors_strand[2], lty=1)

  # Axis
  axis(side = 1, at=c(-20,-10,0,10,20), labels=c("","","","",""))
  axis(side = 2, at=c(minValue,maxValue),c("",""))

  # Support lines X
  for(i in seq(-20, 20, by=10)){
    lines(c(i,i), c(minValue,maxValue), type="l", lwd=1, col="lightgray", lty=2)
  }

  # Support lines Y
  for(i in c(minValue,(minValue+maxValue)/2,maxValue)){
    lines(c(-25,25), c(i,i), type="l", lwd=1, col="lightgray", lty=2)
  }

  toWrite = c(toWrite,"Y axis bias signal forward strand",biasF)
  toWrite = c(toWrite,"Y axis bias signal reverse strand",biasR)

  ###############################################
  # Logo plot
  ###############################################

  # Preparting data
  logoUnc = standardize(originalF + originalR)
  logoCorr = standardize(corrF + corrR)

  # Plot uncorrected signal / Initialize plot
  par(mar=c(1,1,1,1)) # bottom, left, top and right
  plot(xVec, logoUnc, type="l", col=plot_colors_logo[1], xlab="", 
       ylab="", lwd=2, xlim=c(xVec[1],xVec[length(xVec)]), 
       main="", lty=1, axes = FALSE, ylim=c(-0.4,1.05))
  
  # Axis
  axis(side = 1, at=c(-20,-10,0,10,20), labels=c("","","","",""))
  axis(side = 2, at=c(0.0,1.0),c("",""))

  # Support lines X
  for(i in seq(-20, 20, by=10)){
    lines(c(i,i), c(-0.4,1.0), type="l", lwd=1, col="lightgray", lty=2)
  }

  # Support lines Y
  for(i in seq(0.0, 1.0, by=0.5)){
    lines(c(-25,25), c(i,i), type="l", lwd=1, col="lightgray", lty=2)
  }

  # Plot uncorrected signal
  lines(xVec, logoCorr, type="l", lwd=2, col=plot_colors_logo[2], lty=1)

  toWrite = c(toWrite,"Y axis uncorrected DNase signal",logoUnc)
  toWrite = c(toWrite,"Y axis corrected DNase signal",logoCorr)

  ###############################################
  # Termination
  ###############################################

  dev.off()
  system(paste("epstopdf",outFileName,sep=" "))
  write(toWrite,file=outputFileNameTxt,ncolumns=length(logoCorr)+1,append=FALSE,sep='\t')

}

# Function to standardize vectors
standardize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

###################################################################################################
# EXECUTION
###################################################################################################

# Reading nsites
nsitesFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/NatMetLineGraphs/nsites.txt"
nsitesTable = read.table(nsitesFileName, sep="\t", header=F)
nsitesDict = c()
motifNames = c()
for(i in 1:nrow(nsitesTable)){
  nsitesDict = c(nsitesDict,nsitesTable[i,3])
  motifNames = c(motifNames,paste(nsitesTable[i,1],nsitesTable[i,2],sep=""))
}
names(nsitesDict) = motifNames

# Cell Type Loop
toWrite = c("FACTOR","FLANK","CENTER")
for(c in 1:length(cellList)){

  cell = cellList[c]

  # Read Signals
  dataOriginal = read.table(paste(inloc,"PearsonNakedSignal",cell,"signal_original.txt",sep="/"), sep="\t", header=T)
  dataBias = read.table(paste(inloc,"Pearson",cell,"signal_bias.txt",sep="/"), sep="\t", header=T)
  dataCorrected = read.table(paste(inloc,"Pearson",cell,"signal_corrected.txt",sep="/"), sep="\t", header=T)

  # Column Names
  cNames = colnames(dataOriginal)

  # Iterating on columns
  for(i in 0:((ncol(dataOriginal)/2)-1)){

    # Original Data
    originalF = dataOriginal[,(i*2)+1]
    originalR = dataOriginal[,(i*2)+2]

    # Ratio Data
    biasF = dataBias[,(i*2)+1]
    biasR = dataBias[,(i*2)+2]

    # Logo Data
    corrF = dataCorrected[,(i*2)+1]
    corrR = dataCorrected[,(i*2)+2]

    # Fetching nsites
    nameVec = strsplit(cNames[(i*2)+1],"_")[[1]]
    name = paste(nameVec[1:length(nameVec)-1],collapse="_")
    fullname = paste(strsplit(cell,"_")[[1]][2],name,sep="")
    nsites = nsitesDict[fullname]

    # Parameters
    #title = paste(nameVec[1:length(nameVec)-1],collapse=" ")
    if(nameVec[1] == "UW"){title = paste("De novo motif #",nameVec[3],sep="")}
    else if(nameVec[1] == "ER" || nameVec[1] == "AR" || nameVec[1] == "GR"){
      title = paste(nameVec[1]," (",nameVec[2],")",sep="")
    }
    else{title = nameVec[1]}
    outputLocation = paste(outloc,sep="/")
    system(paste("mkdir -p",outputLocation,sep=" "))
    outputFileName = paste(outputLocation,"/",name,".eps",sep="")
    outputFileNameTxt = paste(outputLocation,"/",name,".txt",sep="")

    # Creating Line Plot
    if(title == "EGR1" || title == "NRF1" || title == "CTCF" || title == "JUN" || title == "ER (40min)"){
      createLinePlot(originalF,originalR,biasF,biasR,corrF,corrR,title,nsites,outputFileName,outputFileNameTxt)

      # Evaluating protection average
      corrAll = standardize(corrF + corrR)
      if(title == "ER (40min)"){
        flank = mean(c(corrAll[10:20],corrAll[30:40]))
        center = mean(corrAll[21:29])
      } else{
        flank = mean(c(corrAll[1:15],corrAll[36:50]))
        center = mean(corrAll[16:35])
      }
      toWrite = c(toWrite,title,flank,center)

    }
  }

}

# Writing protection information
write(toWrite,file="protection_list.txt",ncolumns=3,append=FALSE,sep='\t')


