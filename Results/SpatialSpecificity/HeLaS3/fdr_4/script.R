library(lattice)
library(reshape)
w2l <- function(xx) melt(xx, measure.vars = colnames(xx))

# HeLaS3 - fdr_4 - MAX 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/MAX_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / MAX',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/MAX.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/MAX.eps')

# HeLaS3 - fdr_4 - REST 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/REST.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/REST_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / REST',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/REST.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/REST.eps')

# HeLaS3 - fdr_4 - TBP 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/TBP_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / TBP',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/TBP.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/TBP.eps')

# HeLaS3 - fdr_4 - JUND 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/JUND_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / JunD',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/JUND.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/JUND.eps')

# HeLaS3 - fdr_4 - STAT1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/STAT1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/STAT1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / STAT1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/STAT1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/STAT1.eps')

# HeLaS3 - fdr_4 - GABP 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/GABP_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / GABPA',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/GABP.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/GABP.eps')

# HeLaS3 - fdr_4 - FOS 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/FOS.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/FOS_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / C-fos',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/FOS.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/FOS.eps')

# HeLaS3 - fdr_4 - USF2 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/USF2_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / USF2',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/USF2.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/USF2.eps')

# HeLaS3 - fdr_4 - JUN 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/JUN_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / C-jun',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/JUN.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/JUN.eps')

# HeLaS3 - fdr_4 - E2F4 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/E2F4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/E2F4_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / E2F4',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/E2F4.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/E2F4.eps')

# HeLaS3 - fdr_4 - CTCF 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/CTCF_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / CTCF',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/CTCF.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/CTCF.eps')

# HeLaS3 - fdr_4 - MAFK 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/MAFK_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / MAFK',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/MAFK.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/MAFK.eps')

# HeLaS3 - fdr_4 - ELK1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/ELK1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/ELK1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / ELK1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/ELK1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/ELK1.eps')

# HeLaS3 - fdr_4 - MYC 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/MYC_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / Myc',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/MYC.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/MYC.eps')

# HeLaS3 - fdr_4 - NFYB 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/NFYB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/NFYB_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / NF-YB',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/NFYB.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/NFYB.eps')

# HeLaS3 - fdr_4 - NRF1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/NRF1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / NRF1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/NRF1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/NRF1.eps')

# HeLaS3 - fdr_4 - fdr_4 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/fdr_4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/fdr_4_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / All TFs',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/fdr_4.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/fdr_4.eps')

# HeLaS3 - fdr_4 - RAD21 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/RAD21_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / RAD21',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/RAD21.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/RAD21.eps')

# HeLaS3 - fdr_4 - ZNF143 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/ZNF143.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/ZNF143_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / ZNF143',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/ZNF143.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/ZNF143.eps')

# HeLaS3 - fdr_4 - P300 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/P300.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/P300_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / p300',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/P300.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/P300.eps')

# HeLaS3 - fdr_4 - CEBPB 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/CEBPB_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / CEBPB',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/CEBPB.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/CEBPB.eps')

# HeLaS3 - fdr_4 - NFYA 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/NFYA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/NFYA_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / NF-YA',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/NFYA.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/NFYA.eps')

# HeLaS3 - fdr_4 - BRCA1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/BRCA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/BRCA1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / BRCA1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/BRCA1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/BRCA1.eps')

# HeLaS3 - fdr_4 - E2F6 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/E2F6.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/HeLaS3/fdr_4/E2F6_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,2,3,4)])
methodNameVec = c('DH-HMM(ESC)','DH-HMM(HeLa)','DH-HMM(Hep)','DH-HMM(K562)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('HeLa-S3 / E2F6',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/E2F6.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/HeLaS3/fdr_4/E2F6.eps')

