
###########################################################
# Import
###########################################################

# Import
library(lattice)
library(reshape)
library(plotrix)
library(ggplot2)

###########################################################
# Input
###########################################################

# Input
loc = '/home/egg/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ParameterTest/TfQualityTest/'

# Parameters
mar.default <- c(5,5,4,2)

# Reading input
data = read.table(paste(loc,'Input','table.txt',sep='/'), sep='\t', header=T)

###########################################################
# Data Fix
###########################################################

# Multiplying AUC by 100
for(i in 10:19){
  data[,i] = data[,i]*100.0
}

# All factorLabels
myFacLabels = c()
for(i in 1:nrow(data)){
  myFacLabels = c(myFacLabels,paste(data[i,2],'(',substr(data[i,1],1,1),')',sep=''))
}

# Select factorLabels
#myFacLabels = c('NRF1(H)','REST(H)','TCF12(H)','SRF(H)','RFX5(H)','STAT1(K)','NRF1(K)','IRF1(K)','STAT5A(K)','REST(K)',
#'SRF(K)','RFX5(K)','NFYA(K)','STAT2(K)','ZNF263(K)','NFYB(K)','RXRA(H)','CEBPB(H)','SP1(H)','JUND(H)',
#'JUN(H)','FOSL1(H)','GABP(H)','SP2(H)','SP4(H)','E2F6(K)','BHLHE40(K)','CEBPB(K)','ATF1(K)','GATA2(K)',
#'EGATA(K)','SP1(K)','EJUND(K)','JUND(K)','GATA1(K)','FOS(K)','ETS1(K)','EFOS(K)','EJUNB(K)','JUN(K)',
#'FOSL1(K)','GABP(K)','SP2(K)','CCNT2(K)','E2F4(K)','TAL1(K)','MEF2A(K)','ZBTB33(K)','USF1(H)','ATF3(H)',
#'MAFK(H)','USF2(H)','BACH1(H)','ELK1(K)','USF1(K)','ATF3(K)','MAFF(K)','MAFK(K)','USF2(K)','NFE2(K)',
#'BACH1(K)','ZNF143(H)','ZNF143(K)','SIX5(H)','EGR1(H)','SIX5(K)','EGR1(K)','ELF1(K)','TR4(K)','POU5F1(H)',
#'MYC(H)','MAX(H)','PU1(K)','NR2F2(K)','MYC(K)','MAX(K)','RAD21(H)','CTCF(H)','RAD21(K)','CTCFL(K)',
#'SMC3(K)','CTCF(K)','YY1(H)','YY1(K)')

###########################################################
# Correlation Function
###########################################################

# Creating Correlation Function
# Formula are the name of the columns. It must be y~x
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
       main=paste('Correlation = ',round(corr,digits = 4),' / p-value = ',round(pValue,digits = 4),sep=''),
       cex.lab=2.0, cex.axis=1.5, cex.main=1.7, cex.sub=2.0)
  text(vec1, vec2, labels=myFacLabels, cex= 0.3, pos = 3)
  abline(regLine,lty=1,lwd=3.0,col="black")
  grid()
  dev.off()
  system(paste('epstopdf',outFileName,sep=' '))

}

###########################################################
# Execution
###########################################################

# PWM_IC vs AUC_DH3
createCorrelation(data[,17], data[,8], PWM_IC~AUC_DH3, data, paste(loc,'Correlation','IC_vs_DH3.eps',sep='/'),
                  'AUC (H3K4me1+H3K4me3+H3K9ac)', 'PWM Information Content')

# CHIP_PWM vs AUC_DH3
createCorrelation(data[,17], data[,9], CHIP_PWM~AUC_DH3, data, paste(loc,'Correlation','CHIP_vs_DH3.eps',sep='/'),
                  'AUC (H3K4me1+H3K4me3+H3K9ac)', 'ChIP + MPBS (%)')

# PWM_IC vs AUC_CUE3
createCorrelation(data[,14], data[,8], PWM_IC~AUC_CUE3, data, paste(loc,'Correlation','IC_vs_Cuellar.eps',sep='/'),
                  'AUC (Cuellar)', 'PWM Information Content')

# CHIP_PWM vs AUC_CUE3
createCorrelation(data[,14], data[,9], CHIP_PWM~AUC_CUE3, data, paste(loc,'Correlation','CHIP_vs_Cuellar.eps',sep='/'),
                  'AUC (Cuellar)', 'ChIP + MPBS (%)')

# PWM_IC vs AUC_CENT
createCorrelation(data[,15], data[,8], PWM_IC~AUC_CENT, data, paste(loc,'Correlation','IC_vs_Centipede.eps',sep='/'),
                  'AUC (Centipede)', 'PWM Information Content')

# CHIP_PWM vs AUC_CENT
createCorrelation(data[,15], data[,9], CHIP_PWM~AUC_CENT, data, paste(loc,'Correlation','CHIP_vs_Centipede.eps',sep='/'),
                  'AUC (Centipede)', 'ChIP + MPBS (%)')

# PWM_IC vs AUC_BOYLE
createCorrelation(data[,11], data[,8], PWM_IC~AUC_BOYLE, data, paste(loc,'Correlation','IC_vs_Boyle.eps',sep='/'),
                  'AUC (Boyle)', 'PWM Information Content')

# CHIP_PWM vs AUC_BOYLE
createCorrelation(data[,11], data[,9], CHIP_PWM~AUC_BOYLE, data, paste(loc,'Correlation','CHIP_vs_Boyle.eps',sep='/'),
                  'AUC (Boyle)', 'ChIP + MPBS (%)')

# PWM_IC vs AUC_NEPH
createCorrelation(data[,12], data[,8], PWM_IC~AUC_NEPH, data, paste(loc,'Correlation','IC_vs_Neph.eps',sep='/'),
                  'AUC (Neph)', 'PWM Information Content')

# CHIP_PWM vs AUC_NEPH
createCorrelation(data[,12], data[,9], CHIP_PWM~AUC_NEPH, data, paste(loc,'Correlation','CHIP_vs_Neph.eps',sep='/'),
                  'AUC (Neph)', 'ChIP + MPBS (%)')

# PWM_IC vs AUC_PWM
createCorrelation(data[,10], data[,8], PWM_IC~AUC_PWM, data, paste(loc,'Correlation','IC_vs_PWM.eps',sep='/'),
                  'AUC (PWM)', 'PWM Information Content')

# CHIP_PWM vs AUC_PWM
createCorrelation(data[,10], data[,9], CHIP_PWM~AUC_PWM, data, paste(loc,'Correlation','CHIP_vs_PWM.eps',sep='/'),
                  'AUC (PWM)', 'ChIP + MPBS (%)')


#postscript('test.eps', width=7.0, height=6.0, horizontal=FALSE, paper='special')
#d <- data.frame(x = c(102856,17906,89697,74384,91081,52457,73749,29910,75604,28267,122136,
#                      54210,48925,58937,76281,67789,69138,18026,90806,44893), 
#                y = c(2818, 234, 2728, 2393, 2893, 1015, 1403, 791, 2243, 596, 2468, 1495, 
#                      1232, 1746, 2410, 1791, 1706, 259, 1982, 836), 
#                names = c("A","C","E","D","G","F","I","H","K","M","L","N","Q","P","S","R","T","W","V","Y"))
#ggplot(d, aes(x,y)) + geom_point() + geom_text(aes(label=names))
#dev.off()
