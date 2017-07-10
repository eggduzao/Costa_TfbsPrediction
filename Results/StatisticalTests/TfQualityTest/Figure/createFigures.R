# Import
library(lattice)
library(reshape)
library(plotrix)

###########################################################
# Correlation
###########################################################

# Input
loc = '/home/egg/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ParameterTest/TfQualityTest/'
mar.default <- c(5,5,4,2)
data = read.table(paste(loc,'Input','table.txt',sep='/'), sep='\t', header=T)
for(i in 10:19){
  data[,i] = data[,i]*100.0
}
myFacLabels = c()
for(i in 1:nrow(data)){
  myFacLabels = c(myFacLabels,paste(data[i,2],'(',substr(data[i,1],1,1),')',sep=''))
}

# Function
createCorrelation <- function(vec1, vec2, formula, data, outFileName, xlab, ylab){
  # Calculating correlation
  regLine = lm(formula, data=data) # Regression line (y,x)
  corrTest = cor.test(vec1, vec2, alternative = "two.sided", method = "pearson", conf.level = 0.95) # Correlation
  pValue = corrTest$p.value
  corr = corrTest$estimate
  # Plotting graph
  postscript(outFileName, width=7.0, height=6.0, horizontal=FALSE, paper='special')
  par(mar=mar.default)
  plot(vec1, vec2, xlab=xlab, ylab=ylab,
       #main=paste('Correlation = ',round(corr,digits = 4),' / p-value = ',round(pValue,digits = 4),sep=''),
       main='Correlation between AUC and PWM IC',
       cex.lab=1.5, cex.axis=1.2, cex.main=1.8, cex.sub=1.0)
  text(vec1, vec2, labels=myFacLabels, cex= 0.4, pos = 3)
  abline(regLine,lty=1,lwd=3.0,col="black")
  grid()
  dev.off()
  system(paste('epstopdf',outFileName,sep=' '))
  system(paste('rm ',outFileName,sep=' '))
}

# Call
createCorrelation(data[,17], data[,8], PWM_IC~AUC_DH3, data, paste(loc,'Figure','TfQuality_Correlation.eps',sep='/'),
                  'AUC (%) for DH-HMM (3)', 'PWM Information Content')

###########################################################
# Boxplot
###########################################################

# Input
loc = '/home/egg/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ParameterTest/TfQualityTest/'
mar.default <- c(5,5,4,2)
w2l <- function(xx) melt(xx, measure.vars = colnames(xx))
classNbVec = c("0.0","1.1","1.2","2.1","2.2","2.3","3.1","3.3","3.5","4.2","5.1","6.2")
classNameVec = c("Yet undefined\nDNA-binding\ndomains","Basic leucine\nzipper factors\n(bZIP)",
                 "Basic helix-\nloop-helix\nfactors (bHLH)","Nuclear receptors\nwith C4 zinc\nfingers",
                 "Other C4 zinc\nfinger-type\nfactors","C2H2 zinc\nfinger factors","Homeo domain\nfactors",
                 "Fork head /\nwinged helix\nfactors","Tryptophan\ncluster factors",
                 "Heteromeric\nCCAAT-binding\nfactors","MADS box\nfactors","STAT domain\nfactors")

# Parameters
inFileName = paste(loc,'Input','boxplot','class_dh-hmm.txt',sep='/')
outFileName = paste(loc,'Figure','TfQuality_Boxplot.eps',sep='/')
outTestName = paste(loc,'Figure','t_test.txt',sep='/')

# Reading dataset
bx = read.table(inFileName,header=TRUE)
bx = bx * 100.0 # Multiplying AUC by 100
tfSizeVec = c('\n(1 TFs)','\n(21 TFs)','\n(12 TFs)','\n(3 TFs)','\n(3 TFs)','\n(21 TFs)','\n(3 TFs)','\n(4 TFs)','\n(7 TFs)','\n(2 TFs)','\n(3 TFs)','\n(3 TFs)')
#selCol = c(9,6,2,3) # Selection sorted by higher to low AUC median
#selCol = c(7,10,9,5,6,8,2,12,3,11,4) # Selection sorted by higher to low AUC median
selCol = c(6,2,3,9,8,7,5,12,11,4,10) # Selection sorted by higher to low number of TFs
bx = bx[,selCol] # Selecting columns
methodNameVec = classNameVec[selCol] # Selecting columns
tfSizeVec = tfSizeVec[selCol] # Selecting columns

# Creating legend with tf sizes
legendForPlot = c()
for (i in 1:length(methodNameVec)){
  legendForPlot = c(legendForPlot,paste(methodNameVec[i],tfSizeVec[i],sep=''))
}
colnames(bx) = legendForPlot

# Creating boxplot
postscript(outFileName,width=10.0,height=5.0,horizontal=FALSE,paper='special')
par(cex.axis=1.0)
bxx <- w2l(bx)
bwplot(value ~ variable, data = bxx, scales=list(tck=0, x=list(rot=0,cex=0.6), y=list(cex=1.0)), # ,label=c()
     col = 'black', main = list('AUC distribution for DH-HMM (3) based on TF\'s class',cex=1.8), xlab = list('Transcription factor class (from TFClass)',cex=1.5),
     ylab = list('AUC (%)',cex=1.5), 
     ylim=c(65,103), par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
     box.dot = list(col = 'black'), box.umbrella= list(lty=1, col = 'black')),
     panel=function(...) {
       panel.abline(h=c(70,80,90,100),lty=2,lwd=1.0,col="gray") 
       panel.bwplot(...)
     })
dev.off()
system(paste('epstopdf',outFileName,sep=' '))
system(paste('rm ',outFileName,sep=' '))

# Performing and writing t-test
methodNameVec = classNbVec[selCol]
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx[,i],bx[,j],conf.level = 0.95,paired=FALSE)
      toWrite = c(toWrite,round(teste$p.value,digits=4))
    }
  }
}
write(toWrite,file=outTestName,ncolumns=ncol(bx)+1,append=FALSE,sep='\t')


