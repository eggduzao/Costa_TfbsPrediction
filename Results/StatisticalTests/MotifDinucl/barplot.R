
# Parameters
inLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MotifDinucl/table/"
outLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MotifDinucl/barplot/"
data = read.table(paste(inLoc,'kmer_barplot_table.txt',sep=''), sep='\t', header=T)
xName = 1:10

for(i in 1:ncol(data)){

  outFileName = paste(outLoc,colnames(data)[i],'.eps',sep='')
  postscript(outFileName, width=4.0, height=8.0, horizontal=FALSE, paper='special')
  par(mar=c(5,5,4,2))
  barplot(rev(data[,i]), main=colnames(data)[i], horiz=TRUE, names.arg=rev(xName),
          xlab="CG Content", ylab="Percentile")
  dev.off()
  system(paste('epstopdf',outFileName,sep=' '))

}



