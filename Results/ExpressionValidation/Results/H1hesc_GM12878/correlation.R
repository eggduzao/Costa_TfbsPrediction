
###########################################################################################################################################
# Correlation Function
###########################################################################################################################################

# Creating Correlation Function
# Formula are the name of the columns. It must be y~x
createCorrelation <- function(vec1, vec2, formula, data, outFileName, xlab, ylab, corrtype){

  # Calculating correlation
  regLine = lm(formula, data=data) # Regression line (y,x)
  resid = residuals(regLine)
  colorVec = c()
  rThresh = 0.2
  totalRes = 0
  for(r in resid){
    if(r > rThresh | r < -rThresh){
      #colorVec = c(colorVec,"#C00000")
      colorVec = c(colorVec,"black")
      totalRes = totalRes + 1
    }
    else{colorVec = c(colorVec,"black")}
  }
  corrTest = cor.test(vec1, vec2, alternative = "two.sided", method = corrtype, conf.level = 0.95) # Correlation
  pValue = corrTest$p.value
  corr = corrTest$estimate

  # Plotting graph
  postscript(outFileName, width=7.0, height=6.0, horizontal=FALSE, paper='special')
  par(mar=c(5,5,4,2))
  plot(vec1, vec2, xlab=xlab, ylab=ylab, col = colorVec,
       main=paste('r = ',round(corr,digits = 4),' / p = ',round(pValue,digits = 4),' / res = ',totalRes,sep=''),
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
inLoc = "./multivar_table/"
outLoc = "./correlation/"
corrtype = "spearman"
methodVec = c(
"Boyle",
"Centipede_80",
"Centipede_85",
"Centipede_90",
"Centipede_95",
"Centipede_99",
"Cuellar_80",
"Cuellar_85",
"Cuellar_90",
"Cuellar_95",
"Cuellar_99",
"Dnase2Tf",
"FLR_80",
"FLR_85",
"FLR_90",
"FLR_95",
"FLR_99",
"FS",
"HINTBC",
"HINTBCN",
"HINT",
"Neph",
"PIQ_80",
"PIQ_85",
"PIQ_90",
"PIQ_95",
"PIQ_99",
"Protection",
"TC",
"Wellington",
"PWM")
metricVec = c("fos","protection","tc","flr")

# Data
inFileName = paste(inLoc,'multivar_table.txt',sep='')
data = read.table(inFileName, sep='\t', header=T, row.names = 1)
name1 = "EXPR_FOLDCHANGE"
vec1 = data[,name1]

# Loop
for(method in methodVec){
  for(metric in metricVec){
 
    # Parameters
    name2 = paste(method,metric,"ks_stat",sep="_")
    vec2 = data[,name2]
    data.sampled = data.frame(X=vec1, Y=vec2)
    outFileName = paste(outLoc,paste(method,metric,sep="_"),'.eps',sep='')

    corr = createCorrelation(data.sampled[,1],data.sampled[,2],Y~X,data.sampled,outFileName,name1,name2,corrtype)

  }

}


