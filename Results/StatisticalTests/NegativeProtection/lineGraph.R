
inFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/NegativeProtection/results/histograms.txt"
outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/NegativeProtection/results/histograms.eps"

data = read.table(inFileName, sep="\t", header=T)
colVec = rainbow(ncol(data))

postscript(outFileName, width=7.0, height=7.0, horizontal=FALSE, paper="special")
par(mar=c(5,4,2,2)) #bottom, left, top and right
xVec = (-nrow(data)/2):((nrow(data)/2)-1)

plot(xVec, data[,1], type="l", col=colVec[1], lwd=2,
     xlab = "Distance from Motif Center", ylab = "Proportion of Overlapping Footprints")
for(i in 2:ncol(data)){
  lines(xVec, data[,i], type="l", col=colVec[i], lwd=2)
}
lines(c(0,0), c(0,.6), type="l", lty="dashed", col="gray", lwd=1)

legend(22, 0.55, colnames(data), cex=1.0, col=colVec, lwd=2, ncol=1)

dev.off()
system(paste("epstopdf",outFileName,sep=" "))


