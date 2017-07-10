library(lattice)
library(reshape)
w2l <- function(xx) melt(xx, measure.vars = colnames(xx))

# K562 - fdr_4 - MEF2A 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MEF2A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/MEF2A_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / MEF2A',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MEF2A.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MEF2A.eps')

# K562 - fdr_4 - MAX 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/MAX_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / MAX',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MAX.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MAX.eps')

# K562 - fdr_4 - CTCFL 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/CTCFL.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/CTCFL_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / CTCFL',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/CTCFL.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/CTCFL.eps')

# K562 - fdr_4 - REST 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/REST.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/REST_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / REST',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/REST.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/REST.eps')

# K562 - fdr_4 - ZBTB33 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ZBTB33.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/ZBTB33_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / ZBTB33',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ZBTB33.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ZBTB33.eps')

# K562 - fdr_4 - TBP 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/TBP_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / TBP',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/TBP.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/TBP.eps')

# K562 - fdr_4 - GATA1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/GATA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/GATA1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / GATA1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/GATA1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/GATA1.eps')

# K562 - fdr_4 - EJUNB 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EJUNB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/EJUNB_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / eGFP-JunB',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EJUNB.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EJUNB.eps')

# K562 - fdr_4 - ARID3A 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ARID3A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/ARID3A_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / ARID3A',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ARID3A.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ARID3A.eps')

# K562 - fdr_4 - MAFF 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MAFF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/MAFF_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / MAFF',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MAFF.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MAFF.eps')

# K562 - fdr_4 - SRF 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SRF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/SRF_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / SRF',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SRF.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SRF.eps')

# K562 - fdr_4 - JUND 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/JUND_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / JunD',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/JUND.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/JUND.eps')

# K562 - fdr_4 - STAT1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/STAT1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/STAT1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / STAT1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/STAT1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/STAT1.eps')

# K562 - fdr_4 - GABP 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/GABP_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / GABPA',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/GABP.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/GABP.eps')

# K562 - fdr_4 - SMC3 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SMC3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/SMC3_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / SMC3',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SMC3.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SMC3.eps')

# K562 - fdr_4 - CCNT2 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/CCNT2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/CCNT2_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / CCNT2',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/CCNT2.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/CCNT2.eps')

# K562 - fdr_4 - FOS 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/FOS.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/FOS_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / C-fos',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/FOS.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/FOS.eps')

# K562 - fdr_4 - EGATA 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EGATA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/EGATA_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / eGFP-GATA2',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EGATA.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EGATA.eps')

# K562 - fdr_4 - EJUND 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EJUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/EJUND_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / eGFP-JunD',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EJUND.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EJUND.eps')

# K562 - fdr_4 - USF2 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/USF2_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / USF2',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/USF2.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/USF2.eps')

# K562 - fdr_4 - JUN 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/JUN_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / C-jun',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/JUN.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/JUN.eps')

# K562 - fdr_4 - E2F4 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/E2F4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/E2F4_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / E2F4',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/E2F4.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/E2F4.eps')

# K562 - fdr_4 - ZNF263 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ZNF263.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/ZNF263_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / ZNF263',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ZNF263.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ZNF263.eps')

# K562 - fdr_4 - CTCF 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/CTCF_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / CTCF',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/CTCF.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/CTCF.eps')

# K562 - fdr_4 - THAP1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/THAP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/THAP1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / THAP1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/THAP1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/THAP1.eps')

# K562 - fdr_4 - NR2F2 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NR2F2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/NR2F2_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / NR2F2',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NR2F2.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NR2F2.eps')

# K562 - fdr_4 - MAFK 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/MAFK_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / MAFK',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MAFK.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MAFK.eps')

# K562 - fdr_4 - ATF1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ATF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/ATF1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / ATF1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ATF1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ATF1.eps')

# K562 - fdr_4 - STAT5A 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/STAT5A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/STAT5A_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / STAT5A',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/STAT5A.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/STAT5A.eps')

# K562 - fdr_4 - BACH1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/BACH1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/BACH1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / BACH1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/BACH1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/BACH1.eps')

# K562 - fdr_4 - YY1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/YY1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / YY1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/YY1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/YY1.eps')

# K562 - fdr_4 - ETS1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ETS1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/ETS1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / ETS1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ETS1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ETS1.eps')

# K562 - fdr_4 - FOSL1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/FOSL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/FOSL1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / FOSL1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/FOSL1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/FOSL1.eps')

# K562 - fdr_4 - SIX5 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/SIX5_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / SIX5',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SIX5.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SIX5.eps')

# K562 - fdr_4 - SP2 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SP2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/SP2_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / SP2',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SP2.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SP2.eps')

# K562 - fdr_4 - SP1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/SP1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / SP1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SP1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/SP1.eps')

# K562 - fdr_4 - TR4 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/TR4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/TR4_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / TR4',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/TR4.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/TR4.eps')

# K562 - fdr_4 - ELK1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ELK1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/ELK1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / ELK1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ELK1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ELK1.eps')

# K562 - fdr_4 - MYC 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/MYC_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / Myc',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MYC.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/MYC.eps')

# K562 - fdr_4 - NFYB 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NFYB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/NFYB_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / NF-YB',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NFYB.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NFYB.eps')

# K562 - fdr_4 - ZBTB7A 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ZBTB7A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/ZBTB7A_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / ZBTB7A',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ZBTB7A.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ZBTB7A.eps')

# K562 - fdr_4 - NRF1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/NRF1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / NRF1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NRF1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NRF1.eps')

# K562 - fdr_4 - NFE2 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NFE2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/NFE2_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / NF-E2',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NFE2.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NFE2.eps')

# K562 - fdr_4 - GATA2 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/GATA2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/GATA2_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / GATA2',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/GATA2.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/GATA2.eps')

# K562 - fdr_4 - IRF1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/IRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/IRF1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / IRF1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/IRF1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/IRF1.eps')

# K562 - fdr_4 - RFX5 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/RFX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/RFX5_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / RFX5',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/RFX5.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/RFX5.eps')

# K562 - fdr_4 - EFOS 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EFOS.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/EFOS_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / eGFP-FOS',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EFOS.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EFOS.eps')

# K562 - fdr_4 - STAT2 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/STAT2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/STAT2_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / STAT2',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/STAT2.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/STAT2.eps')

# K562 - fdr_4 - fdr_4 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/fdr_4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/fdr_4_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / All TFs',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/fdr_4.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/fdr_4.eps')

# K562 - fdr_4 - BHLHE40 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/BHLHE40.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/BHLHE40_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / BHLHE40',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/BHLHE40.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/BHLHE40.eps')

# K562 - fdr_4 - TAL1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/TAL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/TAL1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / TAL1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/TAL1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/TAL1.eps')

# K562 - fdr_4 - EGR1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EGR1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/EGR1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / EGR1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EGR1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/EGR1.eps')

# K562 - fdr_4 - RAD21 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/RAD21_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / RAD21',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/RAD21.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/RAD21.eps')

# K562 - fdr_4 - PU1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/PU1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/PU1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / PU.1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/PU1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/PU1.eps')

# K562 - fdr_4 - ZNF143 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ZNF143.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/ZNF143_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / ZNF143',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ZNF143.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ZNF143.eps')

# K562 - fdr_4 - CEBPB 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/CEBPB_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / CEBPB',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/CEBPB.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/CEBPB.eps')

# K562 - fdr_4 - NFYA 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NFYA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/NFYA_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / NF-YA',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NFYA.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/NFYA.eps')

# K562 - fdr_4 - USF1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/USF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/USF1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / USF1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/USF1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/USF1.eps')

# K562 - fdr_4 - ELF1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ELF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/ELF1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / ELF1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ELF1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ELF1.eps')

# K562 - fdr_4 - E2F6 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/E2F6.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/E2F6_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / E2F6',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/E2F6.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/E2F6.eps')

# K562 - fdr_4 - ATF3 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ATF3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/ATF3_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,9,11,5,7)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('K562 / ATF3',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ATF3.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/K562/fdr_4/ATF3.eps')

