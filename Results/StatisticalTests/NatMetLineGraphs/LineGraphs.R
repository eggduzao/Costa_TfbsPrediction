
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
outloc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/NatMetLineGraphs/Graphs"

# Graph Parameters
plot_colors_strand <- c("red", "blue")
plot_colors_logo <- c("#FF7777", "#006600")

# Data Parameters
cellList = c("DU_H1hesc", "DU_HeLaS3", "DU_HepG2", "DU_Huvec", "DU_K562", 
              "UW_HepG2", "UW_Huvec", "UW_K562", "DU_Mcf7", "DU_Saos2", 
              "UW_denovo", "UW_LnCaP", "UW_m3134_with_DEX", "UW_m3134_wo_DEX", "UW_H7hesc")

cellList = c( "DU_Mcf7", "DU_Saos2",  "UW_denovo", "UW_LnCaP", "UW_m3134_with_DEX", "UW_m3134_wo_DEX", "UW_H7hesc")

###################################################################################################
# LINE PLOT FUNCTION
###################################################################################################

createLinePlot <- function(originalF,originalR,biasF,biasR,corrF,corrR,title,nsites,outFileName,outputFileNameTxt){

  ###############################################
  # Initialization
  ###############################################

  # Initialization
  postscript(outFileName, width=7.0, height=8.0, horizontal=FALSE, paper="special")
  par(mfrow=c(3,1), mar=c(3,4,2,2), cex.axis=1.2, cex.lab=2.0, cex.main=2.0, xpd=TRUE) #bottom, left, top and right
  xVec = (-length(originalF)/2):((length(originalF)/2)-1)
  toWrite = c("X axis",xVec)

  ###############################################
  # Naked DNase plot
  ###############################################

  # Min/Max Values
  minValue = signif(min(c(originalF,originalR)),digits = 2)
  maxValue = signif(max(c(originalF,originalR)),digits = 2)

  # Plot forward signal / Initialize plot
  plot(xVec, originalF, type="l", col=plot_colors_strand[1], xlab="", ylab="", lwd=2, 
       xlim=c(xVec[1],xVec[length(xVec)]), ylim=c(minValue,maxValue), main="", lty=1, axes = FALSE)

  # Plot reverse signal
  lines(xVec, originalR, type="l", lwd=2, col=plot_colors_strand[2], lty=1)

  # Axis
  axis(side = 1, at=c(1,-20,-15,-10,-5,0,5,10,15,20,24), labels = )
  axis(side = 2, at=c(minValue,maxValue))

  # Legend
  if(title == "GR (withDEX)"){mvf = 1.05}
  else{mvf = 1.11}
  legend(0, maxValue*mvf, c("Forward          ","Reverse"), cex=1.4, col=plot_colors_strand, 
         lwd=2, lty=1, bty="n", ncol=2)

  # Title
  titleX = -20
  if(title == "GR (withDEX)"){mvf2 = 1.02}
  else if(title == "De novo motif #0458" || title == "De novo motif #0500"){
    titleX = -15
    mvf2 = 1.05
  }
  else{mvf2 = 1.05}
  text(x=titleX,y=maxValue*mvf2,title,font=2,cex=2.0,bty="n")
  
  # N sites
  if(title == "GR (withDEX)"){mvf3 = 0.95}
  else{mvf3 = 0.95}
  text(x=-20,y=maxValue*mvf3,paste("# Sites = ",nsites,sep=""),font=2,cex=1.4,bty="n")

  # Y-axis title
  text(x=-30,y=(maxValue+minValue)/2,"Average Naked\nDNase-seq Signal",font=2,cex=1.4,bty="n",srt=90)

  toWrite = c(toWrite,"Y axis naked DNase forward strand",originalF)
  toWrite = c(toWrite,"Y axis naked DNase reverse strand",originalR)

  ###############################################
  # Bias plot
  ###############################################

  # Min/Max Values
  minValue = signif(min(c(biasF,biasR)),digits = 2)
  maxValue = signif(max(c(biasF,biasR)),digits = 2)

  # Plot forward signal / Initialize plot
  par(mar=c(3,4,2,2)) # bottom, left, top and right
  plot(xVec, biasF, type="l", col=plot_colors_strand[1], xlab="", ylab="", lwd=2, 
       xlim=c(xVec[1],xVec[length(xVec)]), ylim=c(minValue,maxValue), main="", lty=1, axes = FALSE)

  # Plot reverse signal
  lines(xVec, biasR, type="l", lwd=2, col=plot_colors_strand[2], lty=1)

  # Axis
  axis(side = 1, at=c(-25,-20,-15,-10,-5,0,5,10,15,20,24))
  axis(side = 2, at=c(minValue,maxValue))

  # Y-axis title
  text(x=-30,y=(maxValue+minValue)/2,"Average Bias\nSignal",font=2,cex=1.4,bty="n",srt=90)

  toWrite = c(toWrite,"Y axis bias signal forward strand",biasF)
  toWrite = c(toWrite,"Y axis bias signal reverse strand",biasR)

  ###############################################
  # Logo plot
  ###############################################

  # Preparting data
  logoUnc = standardize(originalF + originalR)
  logoCorr = standardize(corrF + corrR)

  # Plot uncorrected signal / Initialize plot
  par(mar=c(4,4,1,2)) # bottom, left, top and right
  plot(xVec, logoUnc, type="l", col=plot_colors_logo[1], xlab="", 
       ylab="", lwd=2, xlim=c(xVec[1],xVec[length(xVec)]), 
       main="", lty=1, axes = FALSE, ylim=c(-0.4,1.05))
  
  # Axis
  axis(side = 1, at=c(-25,-20,-15,-10,-5,0,5,10,15,20,24))
  axis(side = 2, at=c(0.0,1.0))

  # Support lines X
  for(i in seq(-25, 25, by=5)){
    if(i == 25){i = 24}
    lines(c(i,i), c(-0.4,1.0), type="l", lwd=1, col="lightgray", lty=2)
  }

  # Support lines Y
  for(i in seq(0.0, 1.0, by=0.5)){
    lines(c(-25,25), c(i,i), type="l", lwd=1, col="lightgray", lty=2)
  }

  # Plot uncorrected signal
  lines(xVec, logoCorr, type="l", lwd=2, col=plot_colors_logo[2], lty=1)

  # Legend
  legend(0, 1.25, c("Uncorrected   ","Corrected"), cex=1.4, col=plot_colors_logo, 
         lwd=2, lty=1, bty="n", ncol=2)

  # Y-axis title
  text(x=-30,y=0.5,"Average DNase-seq\nSignal",font=2,cex=1.4,bty="n",srt=90)

  # X-axis title
  text(x=0,y=-0.75,"Coordinates from Motif Center",font=2,cex=1.4,bty="n")

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
    outputLocation = paste(outloc,cell,sep="/")
    system(paste("mkdir -p",outputLocation,sep=" "))
    outputFileName = paste(outputLocation,"/",name,".eps",sep="")
    outputFileNameTxt = paste(outputLocation,"/",name,".txt",sep="")

    # Creating Line Plot
    createLinePlot(originalF,originalR,biasF,biasR,corrF,corrR,title,nsites,outputFileName,outputFileNameTxt)

  }
}


