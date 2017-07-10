
###########################################################################################################################################
# Correlation Function
###########################################################################################################################################

# Creating Correlation Function
# Formula are the name of the columns. It must be y~x
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
  #abline(regLine,lty=1,lwd=3.0,col="black")
  grid()
  dev.off()
  system(paste('epstopdf',outFileName,sep=' '))
  system(paste('rm',outFileName,sep=' '))

  # Return correlation
  return(round(corr,digits = 6))

}


###########################################################################################################################################
# Execution
###########################################################################################################################################

# Parameters
inFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/CgContent/final_table.txt"
outLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/CgContent/"
corrtype = "spearman"

# Data
data = read.table(inFileName, sep='\t', header=T)
name1 = "MOTIF_CG_CONTENT"
vec1 = data[,name1]

name2 = "HINT"
vec2 = data[,"HINT_AUC_10"]
data.sampled = data.frame(X=vec1, Y=vec2)
outFileName = paste(outLoc,name2,'.eps',sep='')
corr = createCorrelation(data.sampled[,1],data.sampled[,2],Y~X,data.sampled,outFileName,name1,name2,corrtype)

name2 = "HINTBC"
vec2 = data[,"HINT_BC_AUC_10"]
data.sampled = data.frame(X=vec1, Y=vec2)
outFileName = paste(outLoc,name2,'.eps',sep='')
corr = createCorrelation(data.sampled[,1],data.sampled[,2],Y~X,data.sampled,outFileName,name1,name2,corrtype)

name2 = "HINTBCN"
vec2 = data[,"HINT_BCN_AUC_10"]
data.sampled = data.frame(X=vec1, Y=vec2)
outFileName = paste(outLoc,name2,'.eps',sep='')
corr = createCorrelation(data.sampled[,1],data.sampled[,2],Y~X,data.sampled,outFileName,name1,name2,corrtype)

name2 = "BIAS"
vec2 = data[,"OBS"]
data.sampled = data.frame(X=vec1, Y=vec2)
outFileName = paste(outLoc,name2,'.eps',sep='')
corr = createCorrelation(data.sampled[,1],data.sampled[,2],Y~X,data.sampled,outFileName,name1,name2,corrtype)

name2 = "HINTBC-HINT"
vec2 = data[,"HINT_BC_AUC_10"] - data[,"HINT_AUC_10"]
data.sampled = data.frame(X=vec1, Y=vec2)
outFileName = paste(outLoc,name2,'.eps',sep='')
corr = createCorrelation(data.sampled[,1],data.sampled[,2],Y~X,data.sampled,outFileName,name1,name2,corrtype)

name2 = "HINTBC-HINTBCN"
vec2 = data[,"HINT_BC_AUC_10"] - data[,"HINT_BCN_AUC_10"]
data.sampled = data.frame(X=vec1, Y=vec2)
outFileName = paste(outLoc,name2,'.eps',sep='')
corr = createCorrelation(data.sampled[,1],data.sampled[,2],Y~X,data.sampled,outFileName,name1,name2,corrtype)

name2 = "HINTBCN-HINT"
vec2 = data[,"HINT_BCN_AUC_10"] - data[,"HINT_AUC_10"]
data.sampled = data.frame(X=vec1, Y=vec2)
outFileName = paste(outLoc,name2,'.eps',sep='')
corr = createCorrelation(data.sampled[,1],data.sampled[,2],Y~X,data.sampled,outFileName,name1,name2,corrtype)


