
###########################################################################################################################################
# Import
###########################################################################################################################################

# Import
#library(lattice)
#library(reshape)
#library(plotrix)
#library(ggplot2)

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
  #postscript(outFileName, width=7.0, height=6.0, horizontal=FALSE, paper='special')
  #par(mar=c(5,5,4,2))
  #plot(vec1, vec2, xlab=xlab, ylab=ylab,
  #     main=paste('Correlation = ',round(corr,digits = 4),' / p-value = ',round(pValue,digits = 4),sep=''),
  #     cex.lab=2.0, cex.axis=1.5, cex.main=1.7, cex.sub=2.0)
  #abline(regLine,lty=1,lwd=3.0,col="black")
  #grid()
  #dev.off()
  #system(paste('epstopdf',outFileName,sep=' '))

  # Return correlation
  return(round(corr,digits = 6))

}

# Function to standardize values
standardize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

###########################################################################################################################################
# Correlation between datasets
###########################################################################################################################################
# All datasets are compared with each other. Only same-strand datasets are compared.
# Only kmer tables generated using HS method were used in this test.

# Parameters
loc = '/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBiasFull'
inLoc = paste(loc,'bias','table',sep='/')
outLoc = paste(loc,'correlation',sep='/')
corrtype = "spearman"

# Strand Parameters
strandList = c("All")

# Strand Loop
for (str in 1:length(strandList)) {

  # Dataset Parameters
  strand = strandList[str]
  cellData = read.table(paste(loc,'bias','CellFull.txt',sep='/'), sep='\t', header=F)
  inList = as.character(cellData[,1])
  labelList = as.character(cellData[,1])

  # Initializing correlation table
  toWriteSam = c('',labelList)
  toWriteArt = c('',labelList)

  # Correlating sets two-by-two
  for (s1 in 1:length(inList)) {

    # Reading input
    name1 = inList[s1]
    data1 = read.table(paste(inLoc,'/hs_',name1,'_6_',strand,'.txt',sep=''), sep='\t', header=T)
    data1 = data1[order(data1[,1]),]
    v1_1 = standardize(data1[,2])
    v1_2 = standardize(data1[,3])

    # Writing correlation heading
    toWriteSam = c(toWriteSam,labelList[s1])
    toWriteArt = c(toWriteArt,labelList[s1])

    for (s2 in 1:length(inList)) {
    
      # Reading input
      name2 = inList[s2]
      data2 = read.table(paste(inLoc,'/hs_',name2,'_6_',strand,'.txt',sep=''), sep='\t', header=T)
      data2 = data2[order(data2[,1]),]
      v2_1 = standardize(data2[,2])
      v2_2 = standardize(data2[,3])

      # Creating data table
      data.sampled = data.frame(X=v1_1, Y=v2_1)

      # Correlation graphs
      corr_sam = createCorrelation(data.sampled[,1], data.sampled[,2], Y~X, data.sampled,
                                   paste(outLoc,'/',corrtype,'_',strand,'_',name1,'_',name2,'.eps',sep=''),
                                   name1, name2, corrtype)
    
      # Writing correlations
      toWriteSam = c(toWriteSam,corr_sam)

    }
  }

  # Writing correlation to file
  write(toWriteSam,file=paste(outLoc,'/',corrtype,'_',strand,'_corr.txt',sep=''),ncolumns=length(inList)+1,append=FALSE,sep='\t')

}



