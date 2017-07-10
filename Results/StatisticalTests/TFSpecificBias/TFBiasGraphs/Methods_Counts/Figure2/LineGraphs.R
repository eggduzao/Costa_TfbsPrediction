
# Import
library(lattice)
library(reshape)
library(plotrix)
library(ggplot2)

# Parameters
loc = "/home/egg/Projects/TfbsPrediction/Results/StatisticalTests/TFSpecificBias"
plot_colors <- c("#FF7777", "#006600")
plot_lty <- c(1,1)
mar.default <- c(5,5,4,2)
cellList = c("DU_H1hesc")

# Function to create line plot without strand separation
createLine1 <- function(vec1,vec2,vec3,vec4,title,outFileName){

  # Merging strands
  newvec1 = standardize(vec1 + vec2)
  newvec2 = standardize(vec3 + vec4)

  # Plotting graph
  postscript(outFileName, width=6.0, height=7.0, horizontal=FALSE, paper='special')
  par(mar=mar.default, cex.axis=1.2, cex.lab=2.0, cex.main=2.0, xpd=TRUE)
  
  xVec = (6:46)
  newvec1 = newvec1[xVec]
  newvec2 = newvec2[xVec]
  xVec = (-20:20)
  plot(xVec, newvec1, type="l", col=plot_colors[1], xlab="Coordinates from Motif Center", ylab="Signal Intensity", lwd=2,
       xlim=c(-20,20), ylim=c(-0.25,1.05), main="", lty=plot_lty[1], axes = FALSE)

  axis(side = 1, at=c(-20,-10,0,10,20))
  axis(side = 2, at=c(0.0,0.2,0.4,0.6,0.8,1.0))

  # Support lines X
  lines(c(-20,-20), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-10,-10), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(0,0), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(10,10), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(20,20), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)

  # Support lines Y
  lines(c(-20,20), c(0.0,0.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-20,20), c(0.2,0.2), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-20,20), c(0.4,0.4), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-20,20), c(0.6,0.6), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-20,20), c(0.8,0.8), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-20,20), c(1.0,1.0), type="l", lwd=1, col="lightgray", lty=2)

  # Plot lines
  lines(xVec, newvec2, type="l", lwd=2, col=plot_colors[2], lty=plot_lty[2])

  legend(-10.0,1.26, c("Observed DNase I Signal","Corrected DNase I Signal"), cex=1.4, col=plot_colors, lwd=5, lty=plot_lty, bty="n");

  text(x=-20.0,y=1.12,title,font=2,cex=2.0,bty="n")

  dev.off()

  system(paste('epstopdf',outFileName,sep=' '))

}

# Function to create line plot without strand separation
createLine2 <- function(vec1,vec2,vec3,vec4,title,outFileName){

  # Merging strands
  newvec1 = standardize(vec1 + vec2)
  newvec2 = standardize(vec3 + vec4)

  # Plotting graph
  postscript(outFileName, width=6.0, height=7.0, horizontal=FALSE, paper='special')
  par(mar=mar.default, cex.axis=1.2, cex.lab=2.0, cex.main=2.0, xpd=TRUE)
  
  xVec = (6:46)
  newvec1 = newvec1[xVec]
  newvec2 = newvec2[xVec]
  xVec = (-20:20)
  plot(xVec, newvec1, type="l", col=plot_colors[1], xlab="", ylab="", lwd=2,
       xlim=c(-20,20), ylim=c(-0.25,1.05), main="", lty=plot_lty[1], axes = FALSE)

  axis(side = 1, at=c(-20,-10,0,10,20), labels=c("","","","",""))
  axis(side = 2, at=c(0.0,0.2,0.4,0.6,0.8,1.0), labels=c("","","","","",""))

  # Support lines X
  lines(c(-20,-20), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-10,-10), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(0,0), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(10,10), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(20,20), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)

  # Support lines Y
  lines(c(-20,20), c(0.0,0.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-20,20), c(0.2,0.2), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-20,20), c(0.4,0.4), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-20,20), c(0.6,0.6), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-20,20), c(0.8,0.8), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-20,20), c(1.0,1.0), type="l", lwd=1, col="lightgray", lty=2)

  # Plot lines
  lines(xVec, newvec2, type="l", lwd=2, col=plot_colors[2], lty=plot_lty[2])

  #legend(-10.0,1.26, c("                         ",""), cex=1.4, col=plot_colors, lwd=5, lty=plot_lty, bty="n");

  dev.off()

  system(paste('epstopdf',outFileName,sep=' '))

}

# Function to standardize vectors
standardize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

# Cell Type Loop
for(c in 1:length(cellList)){

  cell = cellList[c]

  # Open data matrices
  dataO = read.table(paste(loc,"Pearson",cell,'signal_original.txt',sep='/'), sep='\t', header=T)
  dataC = read.table(paste(loc,"Pearson",cell,'signal_corrected.txt',sep='/'), sep='\t', header=T)
  pwmLoc = paste(loc,"Pearson",cell,'PWM',sep='/')
  cNames = c("EGR1","NRF1")

  # Iterating on columns
  for(cName in cNames){

    # Original
    vecO1 = standardize(dataO[,paste(cName,"_F",sep="")])
    vecO2 = standardize(dataO[,paste(cName,"_R",sep="")])

    # Corrected
    vecC1 = standardize(dataC[,paste(cName,"_F",sep="")])
    vecC2 = standardize(dataC[,paste(cName,"_R",sep="")])

    name = cName
    out = paste(loc,"/TFBiasGraphs/Methods_Counts/Figure2/",name,'_withLabel.eps',sep="")
    createLine1(vecO1,vecO2,vecC1,vecC2,name,out)

    out = paste(loc,"/TFBiasGraphs/Methods_Counts/Figure2/",name,'_woLabel.eps',sep="")
    createLine2(vecO1,vecO2,vecC1,vecC2,name,out)

  }
}


