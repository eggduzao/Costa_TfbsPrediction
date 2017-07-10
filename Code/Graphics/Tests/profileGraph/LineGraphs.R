
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

  postscript(outFileName,width=12.0,height=5.0,horizontal=FALSE,paper='special')
  par(mar=c(7,4,1,1))
  
  plot(1:nrow(dataFrame), dataFrame[,1], type = "n", ylim=c(minValue,maxValue))
  lines(1:nrow(dataFrame), dataFrame[,1], col="red")
  lines(1:nrow(dataFrame), dataFrame[,2], col="green")

  dev.off()
  system(paste("epstopdf",outFileName,sep=" "))
  system(paste("rm",outFileName,sep=" "))

}

###################################################################################################
# EXECUTION
###################################################################################################

# Parameters
inFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/Graphics/Tests/profileGraph/profile.txt"
outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/Graphics/Tests/profileGraph/profile.eps"

# Reading table
bx = read.table(inFileName, header=TRUE)

# Making plot
lineplot(bx,outFileName)


