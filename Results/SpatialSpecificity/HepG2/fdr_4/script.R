library(lattice)
library(reshape)
w2l <- function(xx) melt(xx, measure.vars = colnames(xx))

# HepG2 - fdr_4 - MAX 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/MAX_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / MAX',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/MAX.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/MAX.eps')

# HepG2 - fdr_4 - REST 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/REST.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/REST_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / REST',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/REST.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/REST.eps')

# HepG2 - fdr_4 - TBP 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/TBP_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / TBP',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/TBP.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/TBP.eps')

# HepG2 - fdr_4 - ARID3A 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/ARID3A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/ARID3A_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / ARID3A',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/ARID3A.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/ARID3A.eps')

# HepG2 - fdr_4 - MAFF 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/MAFF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/MAFF_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / MAFF',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/MAFF.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/MAFF.eps')

# HepG2 - fdr_4 - SRF 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/SRF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/SRF_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / SRF',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/SRF.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/SRF.eps')

# HepG2 - fdr_4 - JUND 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/JUND_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / JunD',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/JUND.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/JUND.eps')

# HepG2 - fdr_4 - GABP 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/GABP_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / GABPA',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/GABP.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/GABP.eps')

# HepG2 - fdr_4 - RXRA 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/RXRA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/RXRA_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / RXRA',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/RXRA.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/RXRA.eps')

# HepG2 - fdr_4 - USF2 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/USF2_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / USF2',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/USF2.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/USF2.eps')

# HepG2 - fdr_4 - JUN 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/JUN_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / C-jun',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/JUN.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/JUN.eps')

# HepG2 - fdr_4 - CTCF 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/CTCF_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / CTCF',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/CTCF.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/CTCF.eps')

# HepG2 - fdr_4 - MAFK 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/MAFK_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / MAFK',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/MAFK.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/MAFK.eps')

# HepG2 - fdr_4 - YY1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/YY1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / YY1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/YY1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/YY1.eps')

# HepG2 - fdr_4 - SP2 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/SP2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/SP2_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / SP2',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/SP2.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/SP2.eps')

# HepG2 - fdr_4 - SP1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/SP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/SP1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / SP1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/SP1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/SP1.eps')

# HepG2 - fdr_4 - MYC 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/MYC_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / Myc',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/MYC.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/MYC.eps')

# HepG2 - fdr_4 - NRF1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/NRF1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / NRF1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/NRF1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/NRF1.eps')

# HepG2 - fdr_4 - fdr_4 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/fdr_4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/fdr_4_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / All TFs',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/fdr_4.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/fdr_4.eps')

# HepG2 - fdr_4 - BHLHE40 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/BHLHE40.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/BHLHE40_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / BHLHE40',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/BHLHE40.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/BHLHE40.eps')

# HepG2 - fdr_4 - RAD21 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/RAD21_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / RAD21',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/RAD21.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/RAD21.eps')

# HepG2 - fdr_4 - P300 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/P300.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/P300_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / p300',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/P300.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/P300.eps')

# HepG2 - fdr_4 - CEBPB 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/CEBPB_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / CEBPB',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/CEBPB.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/CEBPB.eps')

# HepG2 - fdr_4 - USF1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/USF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/USF1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / USF1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/USF1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/USF1.eps')

# HepG2 - fdr_4 - BRCA1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/BRCA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/BRCA1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / BRCA1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/BRCA1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/BRCA1.eps')

# HepG2 - fdr_4 - ELF1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/ELF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/ELF1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / ELF1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/ELF1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/ELF1.eps')

# HepG2 - fdr_4 - ATF3 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/ATF3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HepG2/fdr_4/ATF3_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HepG2 / ATF3',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
toWrite = c('XXXXX',methodNameVec)
for (i in (1:ncol(bx2))){
  toWrite = c(toWrite,methodNameVec[i])
  for (j in (1:ncol(bx2))){
    if(i == j){
      toWrite = c(toWrite,'XXXXX')
    }
    else{
      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)
      toWrite = c(toWrite,format(teste$p.value,digits=5))
    }
  }
}
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/ATF3.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HepG2/fdr_4/ATF3.eps')

