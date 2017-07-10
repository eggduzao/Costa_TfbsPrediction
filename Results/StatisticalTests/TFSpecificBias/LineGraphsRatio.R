
# Import
library(lattice)
library(reshape)
library(plotrix)
library(ggplot2)

# Parameters
loc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias"
plot_colors <- c("red", "blue", "red", "blue", "red", "blue")
plot_lty <- c(2,2,3,3,1,1)
#plot_colors <- rainbow(6)
#plot_lty <- c(1,1,1,1,1,1)
#plot_colors2 <- c("black", "red", "green")
invList <- c()
plot_colors2 <- c("#FF7777", "red", "#006600")
plot_lty2 <- c(1,1,1)
mar.default <- c(5,5,4,2)
cellList = c("DU_H1hesc", "DU_HeLaS3", "DU_HepG2", "DU_Huvec", "DU_K562", "UW_HepG2", "UW_Huvec", "UW_K562", "DU_Mcf7", "DU_Saos2", "UW_denovo", "UW_LnCaP", "UW_m3134_with_DEX", "UW_m3134_wo_DEX")
cellList = c( "UW_denovo" )

# Function to create line plot with strand separation
createLine1 <- function(vec1,vec2,vec3,vec4,vec5,vec6,title,outFileName){

  # Plotting graph
  postscript(outFileName, width=7.0, height=6.0, horizontal=FALSE, paper='special')
  par(mar=mar.default)
  
  xVec = (-length(vec1)/2):((length(vec1)/2)-1)
  plot(xVec, vec1, type="l", col=plot_colors[1], xlab="Coordinates from Motif Center", ylab="DNase I Cut Intensity", lwd=2,
       xlim=c((-length(vec1)/2),((length(vec1)/2)-1)), main=title, lty=plot_lty[1])

  lines(xVec, vec2, type="l", lwd=2, col=plot_colors[2], lty=plot_lty[2])
  lines(xVec, vec3, type="l", lwd=2, col=plot_colors[3], lty=plot_lty[3])
  lines(xVec, vec4, type="l", lwd=2, col=plot_colors[4], lty=plot_lty[4])
  lines(xVec, vec5, type="l", lwd=2, col=plot_colors[5], lty=plot_lty[5])
  lines(xVec, vec6, type="l", lwd=2, col=plot_colors[6], lty=plot_lty[6])

  legend(0, 0, c("Observed +","Observed -","Bias +","Bias -","Corrected +","Corrected -"), cex=1.5, col=plot_colors, lwd=4, lty=plot_lty, bty="n")

  dev.off()
  #system(paste('epstopdf',outFileName,sep=' '))
  #system(paste('rm',outFileName,sep=' '))

}

# Function to create line plot without strand separation
createLine2 <- function(vec1,vec2,vec3,vec4,vec5,vec6,title,outFileName){

  # Merging strands
  newvec1 = standardize(vec1 + vec2)
  newvec2 = standardize(vec3 + vec4)
  newvec3 = standardize(vec5 + vec6)

  # Inv
  if(title %in% invList){
    t <- newvec1
    tt <- newvec3
    newvec1 <- c()
    newvec3 <- c()
    newvec1 <- tt
    newvec3 <- t
  }

  # Plotting graph
  postscript(outFileName, width=7.0, height=6.0, horizontal=FALSE, paper='special')
  par(mar=mar.default, cex.axis=1.2, cex.lab=2.0, cex.main=2.0, xpd=TRUE)
  
  xVec = (-length(newvec1)/2):((length(newvec1)/2)-1)
  plot(xVec, newvec1, type="l", col=plot_colors2[1], xlab="Coordinates from Motif Center", ylab="       Average DNase-seq Signal", lwd=2,
       xlim=c((-length(newvec1)/2),((length(newvec1)/2)-1)), ylim=c(-0.25,1.05), main="", lty=plot_lty2[1], axes = FALSE)

  axis(side = 1, at=c(-25,-20,-15,-10,-5,0,5,10,15,20,24))
  axis(side = 2, at=c(0.0,0.2,0.4,0.6,0.8,1.0))

  # Support lines X
  lines(c(-25,-25), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-20,-20), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-15,-15), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-10,-10), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-5,-5), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(0,0), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(5,5), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(10,10), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(15,15), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(20,20), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(24,24), c(-0.3,1.0), type="l", lwd=1, col="lightgray", lty=2)

  # Support lines Y
  lines(c(-25,25), c(0.0,0.0), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-25,25), c(0.2,0.2), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-25,25), c(0.4,0.4), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-25,25), c(0.6,0.6), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-25,25), c(0.8,0.8), type="l", lwd=1, col="lightgray", lty=2)
  lines(c(-25,25), c(1.0,1.0), type="l", lwd=1, col="lightgray", lty=2)

  # Plot lines
  #lines(xVec, newvec2, type="l", lwd=2, col=plot_colors2[2], lty=plot_lty2[2])
  lines(xVec, newvec3, type="l", lwd=2, col=plot_colors2[3], lty=plot_lty2[3])

  legend(-10.0,1.26, c("Uncorrected","Bias-Corrected"), cex=1.4, col=c(plot_colors2[1],plot_colors2[3]), lwd=5, lty=plot_lty2, bty="n");
  #legend(-10.0,1.4, c("Observed DNase I Cleavage","Bias Score","Corrected DNase I Signal"), cex=1.3, col=plot_colors2, lwd=4, lty=plot_lty2, bty="n");

  text(x=-20.0,y=1.12,title,font=2,cex=2.0,bty="n")
  #text(x=-20.0,y=1.23,title,font=2,cex=2.0,bty="n")

  dev.off()

  system(paste('epstopdf',outFileName,sep=' '))
  #system(paste('rm',outFileName,sep=' '))

}

# Function to standardize vectors
standardize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

# Cell Type Loop
for(c in 1:length(cellList)){

  cell = cellList[c]

  # Open data matrices
  dataO = read.table(paste(loc,"PearsonRatio",cell,'signal_original.txt',sep='/'), sep='\t', header=T)
  dataB = read.table(paste(loc,"PearsonRatio",cell,'signal_bias.txt',sep='/'), sep='\t', header=T)
  dataC = read.table(paste(loc,"PearsonRatio",cell,'signal_corrected.txt',sep='/'), sep='\t', header=T)
  pwmLoc = paste(loc,"PearsonRatio",cell,'PWM',sep='/')
  cNames = colnames(dataO)

  # Iterating on columns
  for(i in 0:((ncol(dataO)/2)-1)){

    # Original
    vecO1 = standardize(dataO[,(i*2)+1])
    vecO2 = standardize(dataO[,(i*2)+2])

    # Bias
    vecB1 = standardize(dataB[,(i*2)+1])
    vecB2 = standardize(dataB[,(i*2)+2])

    # Corrected
    vecC1 = standardize(dataC[,(i*2)+1])
    vecC2 = standardize(dataC[,(i*2)+2])

    nameVec = strsplit(cNames[(i*2)+1],'_')[[1]]
    name = paste(nameVec[1:length(nameVec)-1],collapse="_")
    out = paste(loc,'/LineGraphsRatio/',cell,'/',name,'.eps',sep='')
    createLine2(vecO1,vecO2,vecB1,vecB2,vecC1,vecC2,name,out)

  }
}


