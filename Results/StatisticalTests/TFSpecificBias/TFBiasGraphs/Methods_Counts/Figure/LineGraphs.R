
# Import
library(lattice)
library(reshape)
library(plotrix)
library(ggplot2)

# Parameters
loc = "/home/egg/Projects/TfbsPrediction/Results/StatisticalTests/TFSpecificBias"
plot_colors2 <- c("#FF9999", "red", "#339900")
plot_lty2 <- c(1,1,1)
mar.default <- c(1,5,7,1)
cellList = c("DU_H1hesc")

# Function to create line plot without strand separation
createLine <- function(vec1,vec2,vec3,vec4,vec5,vec6,title,outFileName){

  # Merging strands
  newvec1 = standardize(vec1 + vec2)
  newvec2 = standardize(vec3 + vec4)
  newvec3 = standardize(vec5 + vec6)

  # Plotting graph
  postscript(outFileName, width=6.0, height=7.0, horizontal=FALSE, paper='special')
  par(mar=mar.default, cex.axis=1.2, cex.lab=1.5, cex.main=2.0, xpd=TRUE)
  
  #xVec = (-length(newvec1)/2):((length(newvec1)/2)-1)
  #plot(xVec, newvec1, type="l", col=plot_colors2[1], xlab="Coordinates from Motif Center", ylab="        Signal Intensity", lwd=2,
  #     xlim=c((-length(newvec1)/2),((length(newvec1)/2)-1)), ylim=c(-0.25,1.05), main="", lty=plot_lty2[1], axes = FALSE)

  xVec = (11:41)
  newvec1 = newvec1[xVec]
  newvec3 = newvec3[xVec]
  xVec = (-15:15)
  plot(xVec, newvec1, type="l", col=plot_colors2[1], xlab="", ylab="Signal Intensity", lwd=2,
       xlim=c(-15,15), ylim=c(0.0,0.98), main="", lty=plot_lty2[1], axes = FALSE)

  #axis(side = 1, at=c(-25,-20,-15,-10,-5,0,5,10,15,20,24))
  axis(side = 3, at=c(-15,-10,-5,0,5,10,15))
  axis(side = 2, at=c(0.0,0.2,0.4,0.6,0.8,1.0))

  # Support lines X
  #lines(c(-25,-25), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(-20,-20), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(-15,-15), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(-10,-10), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(-5,-5), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(0,0), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(5,5), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(10,10), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(15,15), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(20,20), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(24,24), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)

  lines(c(-15,-15), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-10,-10), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-5,-5), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(0,0), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(5,5), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(10,10), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(15,15), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)

  # Support lines Y
  #lines(c(-25,25), c(0.0,0.0), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(-25,25), c(0.2,0.2), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(-25,25), c(0.4,0.4), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(-25,25), c(0.6,0.6), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(-25,25), c(0.8,0.8), type="l", lwd=1, col="lightgray", lty=2)
  #lines(c(-25,25), c(1.0,1.0), type="l", lwd=1, col="lightgray", lty=2)

  lines(c(-15,15), c(0.0,0.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-15,15), c(0.2,0.2), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-15,15), c(0.4,0.4), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-15,15), c(0.6,0.6), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-15,15), c(0.8,0.8), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-15,15), c(1.0,1.0), type="l", lwd=1, col="lightgray", lty=2)

  # Plot lines
  lines(xVec, newvec3, type="l", lwd=2, col=plot_colors2[3], lty=plot_lty2[3])

  legend(-10.0,1.32, c("Observed DNase I Signal","Corrected DNase I Signal"), cex=1.4, col=c(plot_colors2[1],plot_colors2[3]), lwd=5, lty=plot_lty2, bty="n");

  text(x=-18.0,y=1.25,title,font=2,cex=2.0,bty="n")
  text(x=0.0,y=1.13,"Coordinates from Motif Center",font=1,cex=1.5,bty="n")

  dev.off()

  system(paste('epstopdf',outFileName,sep=' '))
  #system(paste('rm',outFileName,sep=' '))

}

createLine2 <- function(vec1,vec2,vec3,vec4,vec5,vec6,title,outFileName){

  # Merging strands
  newvec1 = standardize(vec1 + vec2)
  newvec2 = standardize(vec3 + vec4)
  newvec3 = standardize(vec5 + vec6)

  # Plotting graph
  postscript(outFileName, width=6.0, height=7.0, horizontal=FALSE, paper='special')
  par(mar=mar.default, cex.axis=1.2, cex.lab=1.5, cex.main=2.0, xpd=TRUE)
  
  xVec = (11:41)
  newvec1 = newvec1[xVec]
  newvec3 = newvec3[xVec]
  xVec = (-15:15)
  plot(xVec, newvec1, type="l", col=plot_colors2[1], xlab="", ylab="", lwd=2,
       xlim=c(-15,15), ylim=c(0.0,0.98), main="", lty=plot_lty2[1], axes = FALSE)

  axis(side = 3, at=c(-15,-10,-5,0,5,10,15), labels=c("","","","","","",""))
  axis(side = 2, at=c(0.0,0.2,0.4,0.6,0.8,1.0), labels=c("","","","","",""))

  lines(c(-15,-15), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-10,-10), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-5,-5), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(0,0), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(5,5), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(10,10), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(15,15), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)

  lines(c(-15,15), c(0.0,0.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-15,15), c(0.2,0.2), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-15,15), c(0.4,0.4), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-15,15), c(0.6,0.6), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-15,15), c(0.8,0.8), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-15,15), c(1.0,1.0), type="l", lwd=1, col="lightgray", lty=2)

  # Plot lines
  lines(xVec, newvec3, type="l", lwd=2, col=plot_colors2[3], lty=plot_lty2[3])

  legend(-10.0,1.32, c("                         ",""), cex=1.4, col=c(plot_colors2[1],plot_colors2[3]), lwd=5, lty=plot_lty2, bty="n");

  dev.off()

  system(paste('epstopdf',outFileName,sep=' '))

}

# Function to create line plot without strand separation
createLine3 <- function(vec1,vec2,vec3,vec4,vec5,vec6,title,outFileName){

  # Merging strands
  newvec1 = standardize(vec1 + vec2)
  newvec2 = standardize(vec3 + vec4)
  newvec3 = standardize(vec5 + vec6)

  # Plotting graph
  postscript(outFileName, width=6.0, height=6.0, horizontal=FALSE, paper='special')
  mar.default2 <- c(1,1,1,1)
  par(mar=mar.default2, cex.axis=1.2, cex.lab=1.5, cex.main=2.0, xpd=TRUE)

  xVec = (22:31)
  newvec1 = newvec1[xVec]
  newvec3 = newvec3[xVec]
  xVec = (-4:5)
  plot(xVec, newvec1, type="l", col=plot_colors2[1], xlab="", ylab="", lwd=2,
       xlim=c(-4,5), ylim=c(0.0,0.98), main="", lty=plot_lty2[1], axes = FALSE)

  axis(side = 2, at=c(0.0,0.2,0.4,0.6,0.8,1.0), labels=c("","","","","",""))

  lines(c(-4,-4), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(0,0), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(5,5), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)

  lines(c(-4,5), c(0.0,0.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-4,5), c(0.2,0.2), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-4,5), c(0.4,0.4), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-4,5), c(0.6,0.6), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-4,5), c(0.8,0.8), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-4,5), c(1.0,1.0), type="l", lwd=1, col="lightgray", lty=2)

  # Plot lines
  lines(xVec, newvec3, type="l", lwd=2, col=plot_colors2[3], lty=plot_lty2[3])

  dev.off()

  system(paste('epstopdf',outFileName,sep=' '))

}

createLine4 <- function(vec1,vec2,vec3,vec4,vec5,vec6,title,outFileName){

  # Merging strands
  newvec1 = standardize(vec1 + vec2)
  newvec2 = standardize(vec3 + vec4)
  newvec3 = standardize(vec5 + vec6)

  # Plotting graph
  postscript(outFileName, width=6.0, height=6.0, horizontal=FALSE, paper='special')
  mar.default2 <- c(1,1,1,1)
  par(mar=mar.default2, cex.axis=1.2, cex.lab=1.5, cex.main=2.0, xpd=TRUE)

  xVec = (21:31)
  newvec1 = newvec1[xVec]
  newvec3 = newvec3[xVec]
  xVec = (-5:5)
  plot(xVec, newvec1, type="l", col=plot_colors2[1], xlab="", ylab="", lwd=2,
       xlim=c(-5,5), ylim=c(0.0,0.98), main="", lty=plot_lty2[1], axes = FALSE)

  axis(side = 2, at=c(0.0,0.2,0.4,0.6,0.8,1.0), labels=c("","","","","",""))

  lines(c(-5,-5), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(0,0), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(5,5), c(0,1.0), type="l", lwd=1, col="lightgray", lty=2)

  lines(c(-4,5), c(0.0,0.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-4,5), c(0.2,0.2), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-4,5), c(0.4,0.4), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-4,5), c(0.6,0.6), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-4,5), c(0.8,0.8), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-4,5), c(1.0,1.0), type="l", lwd=1, col="lightgray", lty=2)

  # Plot lines
  lines(xVec, newvec3, type="l", lwd=2, col=plot_colors2[3], lty=plot_lty2[3])

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
  dataB = read.table(paste(loc,"Pearson",cell,'signal_bias.txt',sep='/'), sep='\t', header=T)
  dataC = read.table(paste(loc,"Pearson",cell,'signal_corrected.txt',sep='/'), sep='\t', header=T)
  pwmLoc = paste(loc,"Pearson",cell,'PWM',sep='/')
  cNames = c("EGR1","NRF1")

  # Iterating on columns
  for(cName in cNames){

    # Original
    vecO1 = standardize(dataO[,paste(cName,"_F",sep="")])
    vecO2 = standardize(dataO[,paste(cName,"_R",sep="")])

    # Bias
    vecB1 = standardize(dataB[,paste(cName,"_F",sep="")])
    vecB2 = standardize(dataB[,paste(cName,"_R",sep="")])

    # Corrected
    vecC1 = standardize(dataC[,paste(cName,"_F",sep="")])
    vecC2 = standardize(dataC[,paste(cName,"_R",sep="")])

    name = cName
    out = paste(loc,"/TFBiasGraphs/Methods_Counts/Figure/",name,'_withLabel.eps',sep="")
    createLine(vecO1,vecO2,vecB1,vecB2,vecC1,vecC2,name,out)

    out = paste(loc,"/TFBiasGraphs/Methods_Counts/Figure/",name,'_woLabel.eps',sep="")
    createLine2(vecO1,vecO2,vecB1,vecB2,vecC1,vecC2,name,out)

    if(cName == "EGR1"){

      out = paste(loc,"/TFBiasGraphs/Methods_Counts/Figure/",name,'_zoom.eps',sep="")
      createLine3(vecO1,vecO2,vecB1,vecB2,vecC1,vecC2,name,out)

    }
    else{

      out = paste(loc,"/TFBiasGraphs/Methods_Counts/Figure/",name,'_zoom.eps',sep="")
      createLine4(vecO1,vecO2,vecB1,vecB2,vecC1,vecC2,name,out)

    }

  }
}


