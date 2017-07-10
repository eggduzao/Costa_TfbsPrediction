
###################################################################################################
# INPUT
###################################################################################################

# Import
library(lattice)
library(reshape)
library(plotrix)

###################################################################################################
# FUNCTIONS
###################################################################################################

lineplot <- function(dataFrame, outFileName){

  minValue = min(dataFrame)
  maxValue = max(dataFrame)

  postscript(outFileName,width=6.0,height=5.0,horizontal=FALSE,paper='special')
  par(mar=c(2,2,1,1))
  
  plot(1:nrow(dataFrame), dataFrame[,1], type = "n", xlim=c(1,200), ylim=c(minValue,maxValue), xlab="", ylab="", main="", lty=1, axes = FALSE)
  lines(1:nrow(dataFrame), dataFrame[,1], col="darkgreen")
  lines(1:nrow(dataFrame), dataFrame[,2], col="orange")
  lines(1:nrow(dataFrame), dataFrame[,3], col="red")
  lines(1:nrow(dataFrame), dataFrame[,4], col="blue")

  axis(side = 1, at=c(1,50,101,150,200), labels = c("-100","-50","0","50","100"))
  axis(side = 2)

  dev.off()
  #system(paste("epstopdf",outFileName,sep=" "))
  #system(paste("rm",outFileName,sep=" "))

}

###################################################################################################
# EXECUTION
###################################################################################################

inLoc = "./contingency_profiles/"
outLoc = "./graphs/"

tfList = c("ARID3A", "ATF1", "ATF3", "BACH1", "BHLHE40", "CCNT2", "CEBPB", "CTCF", "CTCFL", "E2F4", "E2F6", "EFOS", "EGATA", "EGR1", "EJUNB", "EJUND", "ELF1", "ELK1", "ETS1", "FOS", "FOSL1", "GABP", "GATA1", "GATA2", "IRF1", "JUN", "JUND", "MAFF", "MAFK", "MAX", "MEF2A", "MYC", "NFE2", "NFYA", "NFYB", "NR2F2", "NRF1", "PU1", "RAD21", "REST", "RFX5", "SIX5", "SMC3", "SP1", "SP2", "SRF", "STAT1", "STAT2", "STAT5A", "TAL1", "TBP", "THAP1", "TR4", "USF1", "USF2", "YY1", "ZBTB7A", "ZBTB33", "ZNF143", "ZNF263")

for(tf in tfList){

  # Parameters
  inFileName = paste(inLoc,tf,".txt",sep="")
  outFileName = paste(outLoc,tf,".eps",sep="")

  # Reading table
  bx = read.table(inFileName, header=TRUE)

  # Making plot
  lineplot(bx,outFileName)

}



