
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

createScatterplot <- function(dataFrame, outFileName){

  postscript(outFileName,width=12.0,height=5.0,horizontal=FALSE,paper='special')
  par(mar=c(7,4,1,1))
  
  plot(1:nrow(dataFrame), dataFrame[,2], type = "n")
  lines(1:nrow(dataFrame), dataFrame[,2])

  dev.off()
  system(paste("epstopdf",outFileName,sep=" "))
  system(paste("rm",outFileName,sep=" "))

}

###################################################################################################
# EXECUTION
###################################################################################################

# Parameters
inFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/CgContent/pwmcg.txt"
outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/CgContent/pwmcg.eps"

# Reading table
bx = read.table(inFileName, header=FALSE)

# Making plot
createScatterplot(bx,outFileName)


