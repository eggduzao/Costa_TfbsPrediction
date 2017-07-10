
#######################################################################################################
# Correlation Function
#######################################################################################################

createCorrelation <- function(vec1, vec2, formula, data, outFileName, xlab, ylab, corrtype){

  # Calculating correlation
  regLine = lm(formula, data=data) # Regression line (y,x)
  corrTest = cor.test(vec1, vec2, alternative = "two.sided", method = corrtype, conf.level = 0.95) # Correlation
  pValue = corrTest$p.value
  corr = corrTest$estimate

  # Plotting graph
  postscript(outFileName, width=7.0, height=6.0, horizontal=FALSE, paper='special')
  par(mar=c(5,5,4,2))
  plot(vec1, vec2, xlab=xlab, ylab=ylab,
       main=paste('Correlation = ',round(corr,digits = 4),' / p-value = ',round(pValue,digits = 4),sep=''),
       cex.lab=2.0, cex.axis=1.5, cex.main=1.7, cex.sub=2.0)
  #text(vec1, vec2, cex= 0.3, pos = 3)
  abline(regLine,lty=1,lwd=3.0,col="black")
  grid()
  dev.off()
  system(paste('epstopdf',outFileName,sep=' '))

  # Return correlation
  return(c(round(corr,digits = 6),pValue))

}


#######################################################################################################
# Execution
#######################################################################################################

# Parameters
inLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MotifDinucl/table/"
outLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MotifDinucl/graphs/"
corrtype = "spearman"
#corrtype = "pearson"
data = read.table(paste(inLoc,'table.txt',sep=''), sep='\t', header=T)
aucVec = data[,3]

# Execution
to_write = c("BP","CORRELATION","PVALUE")
for(i in 4:ncol(data)){
  nuc = colnames(data)[i]
  outFileName = paste(outLoc,nuc,'.eps',sep='')
  dataF = data.frame(X=aucVec, Y=data[,i])
  r = createCorrelation(aucVec,data[,i],Y~X,dataF,outFileName,"HINT(BC) AUC - HINT AUC",paste(nuc,'Frequency',sep=' '),corrtype)
  to_write = c(to_write,nuc,r)
}

write(to_write,file=paste(outLoc,'correlation.txt',sep=''),ncolumns=3,append=FALSE,sep='\t')


