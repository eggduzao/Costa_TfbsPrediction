
###########################################################
# Import
###########################################################

# Import
library(lattice)
library(reshape)
library(plotrix)

###########################################################
# Input
###########################################################

# Input
loc = '/home/egg/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ParameterTest/TfQualityTest/'

# Parameters
mar.default <- c(5,5,4,2)
superClassNbVec = c("0","1","2","3","4","5","6")
superClassNameVec = c("Yet undefined\nDNA-binding\ndomains",
                      "Basic domains",
                      "Zinc-coordinating\nDNA-binding\ndomains",
                      "Helix-turn-helix\ndomains",
                      "Other all-alpha-\nhelical DNA-binding\ndomains",
                      "alpha-Helices\nexposed by\nbeta-structures",
                      "Immunoglobulin\nfold")
classNbVec = c("0.0","1.1","1.2","2.1","2.2","2.3","3.1","3.3","3.5","4.2","5.1","6.2")
classNameVec = c("Yet undefined\nDNA-binding\ndomains",
                 "Basic leucine\nzipper factors\n(bZIP)",
                 "Basic helix-\nloop-helix\nfactors (bHLH)",
                 "Nuclear receptors\nwith C4 zinc\nfingers",
                 "Other C4 zinc\nfinger-type\nfactors",
                 "C2H2 zinc\nfinger factors",
                 "Homeo domain\nfactors",
                 "Fork head /\nwinged helix\nfactors",
                 "Tryptophan\ncluster factors",
                 "Heteromeric\nCCAAT-binding\nfactors",
                 "MADS box\nfactors",
                 "STAT domain\nfactors")

###########################################################
# Boxplot Function
###########################################################

# Auxiliary function
w2l <- function(xx) melt(xx, measure.vars = colnames(xx))

# Create boxplot function
createBoxplot <- function(inFileName,outFileName,outTestName,mainText,ylim,abline){

  # Reading dataset
  bx = read.table(inFileName,header=TRUE)
  bx = bx * 100.0 # Multiplying AUC by 100
  #selCol = 2:7 # Selecting columns
  selCol = 2:12
  bx = bx[,selCol] # Selecting columns
  methodNameVec = classNameVec[selCol] # Selecting columns
  colnames(bx) = methodNameVec
  
  # Creating boxplot
  postscript(outFileName,width=10.0,height=5.0,horizontal=FALSE,paper='special')
  par(cex.axis=1.0)
  bxx <- w2l(bx)
  p = bwplot(value ~ variable, data = bxx, scales=list(tck=0, x=list(rot=0,cex=0.55), y=list(cex=1.0)), # ,label=c()
       col = 'black', main = list(mainText,cex=1.5), xlab = list('TF class',cex=1.5),
       ylab = list('AUC (%)',cex=1.5), 
       ylim=ylim, par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(lty=1, col = 'black')),
       panel=function(...) {
         panel.abline(h=abline,lty=2,lwd=1.0,col="gray") 
         panel.bwplot(...)
       })
  print(p)
  dev.off()
  system(paste('epstopdf',outFileName,sep=' '))

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
        teste = wilcox.test(bx[,i],bx[,j],conf.level = 0.99,paired=FALSE)
        toWrite = c(toWrite,round(teste$p.value,digits=4))
      }
    }
  }
  write(toWrite,file=outTestName,ncolumns=ncol(bx)+1,append=FALSE,sep='\t')

}

###########################################################
# Execution
###########################################################

ylim = c(30,105)
abline = c(40,60,80,100)

# PWM
createBoxplot(paste(loc,'Input','boxplot','class_pwm.txt',sep='/'), paste(loc,'Boxplot','class_pwm.eps',sep='/'),
              paste(loc,'Boxplot','class_pwm.txt',sep='/'), 'PWM',ylim,abline)

# Boyle
createBoxplot(paste(loc,'Input','boxplot','class_boyle.txt',sep='/'), paste(loc,'Boxplot','class_boyle.eps',sep='/'),
              paste(loc,'Boxplot','class_boyle.txt',sep='/'), 'Boyle',ylim,abline)

# Neph
createBoxplot(paste(loc,'Input','boxplot','class_neph.txt',sep='/'), paste(loc,'Boxplot','class_neph.eps',sep='/'),
              paste(loc,'Boxplot','class_neph.txt',sep='/'), 'Neph',ylim,abline)

# Cuellar
createBoxplot(paste(loc,'Input','boxplot','class_cuellar.txt',sep='/'), paste(loc,'Boxplot','class_cuellar.eps',sep='/'),
              paste(loc,'Boxplot','class_cuellar.txt',sep='/'), 'Cuellar',ylim,abline)

# Centipede
createBoxplot(paste(loc,'Input','boxplot','class_centipede.txt',sep='/'), paste(loc,'Boxplot','class_centipede.eps',sep='/'),
              paste(loc,'Boxplot','class_centipede.txt',sep='/'), 'Centipede',ylim,abline)

# DH-HMM
createBoxplot(paste(loc,'Input','boxplot','class_dh-hmm.txt',sep='/'), paste(loc,'Boxplot','class_dh-hmm.eps',sep='/'),
              paste(loc,'Boxplot','class_dh-hmm.txt',sep='/'), 'DH-HMM',ylim,abline)

# PWM
#createBoxplot(paste(loc,'Input','boxplot','class_pwm.txt',sep='/'), paste(loc,'Boxplot','class_pwm.eps',sep='/'),
#              paste(loc,'Boxplot','class_pwm.txt',sep='/'), 'PWM',c(40,105),c(50,60,70,80,90,100))

# Boyle
#createBoxplot(paste(loc,'Input','boxplot','class_boyle.txt',sep='/'), paste(loc,'Boxplot','class_boyle.eps',sep='/'),
#              paste(loc,'Boxplot','class_boyle.txt',sep='/'), 'Boyle',c(50,105),c(60,70,80,90,100))

# Neph
#createBoxplot(paste(loc,'Input','boxplot','class_neph.txt',sep='/'), paste(loc,'Boxplot','class_neph.eps',sep='/'),
#              paste(loc,'Boxplot','class_neph.txt',sep='/'), 'Neph',c(60,105),c(70,80,90,100))

# Cuellar
#createBoxplot(paste(loc,'Input','boxplot','class_cuellar.txt',sep='/'), paste(loc,'Boxplot','class_cuellar.eps',sep='/'),
#              paste(loc,'Boxplot','class_cuellar.txt',sep='/'), 'Cuellar',c(65,105),c(70,80,90,100))

# Centipede
#createBoxplot(paste(loc,'Input','boxplot','class_centipede.txt',sep='/'), paste(loc,'Boxplot','class_centipede.eps',sep='/'),
#              paste(loc,'Boxplot','class_centipede.txt',sep='/'), 'Centipede',c(30,105),c(40,60,80,100))

# DH-HMM
#createBoxplot(paste(loc,'Input','boxplot','class_dh-hmm.txt',sep='/'), paste(loc,'Boxplot','class_dh-hmm.eps',sep='/'),
#              paste(loc,'Boxplot','class_dh-hmm.txt',sep='/'), 'DH-HMM',c(65,105),c(70,80,90,100))


