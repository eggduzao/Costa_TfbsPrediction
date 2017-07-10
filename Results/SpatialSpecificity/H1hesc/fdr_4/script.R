library(lattice)
library(reshape)
w2l <- function(xx) melt(xx, measure.vars = colnames(xx))

# H1hesc - fdr_4 - MAX 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/MAX_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / MAX',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/MAX.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/MAX.eps')

# H1hesc - fdr_4 - REST 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/REST.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/REST_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / REST',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/REST.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/REST.eps')

# H1hesc - fdr_4 - TBP 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/TBP_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / TBP',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/TBP.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/TBP.eps')

# H1hesc - fdr_4 - SP4 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SP4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/SP4_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / SP4',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SP4.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SP4.eps')

# H1hesc - fdr_4 - SRF 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SRF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/SRF_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / SRF',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SRF.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SRF.eps')

# H1hesc - fdr_4 - JUND 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/JUND_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / JunD',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/JUND.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/JUND.eps')

# H1hesc - fdr_4 - GABP 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/GABP_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / GABPA',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/GABP.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/GABP.eps')

# H1hesc - fdr_4 - RXRA 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/RXRA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/RXRA_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / RXRA',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/RXRA.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/RXRA.eps')

# H1hesc - fdr_4 - USF2 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/USF2_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / USF2',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/USF2.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/USF2.eps')

# H1hesc - fdr_4 - JUN 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/JUN_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / C-jun',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/JUN.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/JUN.eps')

# H1hesc - fdr_4 - CTCF 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/CTCF_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / CTCF',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/CTCF.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/CTCF.eps')

# H1hesc - fdr_4 - MAFK 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/MAFK_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / MAFK',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/MAFK.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/MAFK.eps')

# H1hesc - fdr_4 - BACH1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/BACH1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/BACH1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / BACH1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/BACH1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/BACH1.eps')

# H1hesc - fdr_4 - YY1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/YY1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / YY1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/YY1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/YY1.eps')

# H1hesc - fdr_4 - FOSL1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/FOSL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/FOSL1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / FOSL1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/FOSL1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/FOSL1.eps')

# H1hesc - fdr_4 - SIX5 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/SIX5_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / SIX5',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SIX5.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SIX5.eps')

# H1hesc - fdr_4 - SP2 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SP2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/SP2_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / SP2',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SP2.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SP2.eps')

# H1hesc - fdr_4 - TCF12 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/TCF12.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/TCF12_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / TCF12',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/TCF12.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/TCF12.eps')

# H1hesc - fdr_4 - SP1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/SP1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / SP1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SP1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/SP1.eps')

# H1hesc - fdr_4 - MYC 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/MYC_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / Myc',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/MYC.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/MYC.eps')

# H1hesc - fdr_4 - NRF1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/NRF1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / NRF1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/NRF1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/NRF1.eps')

# H1hesc - fdr_4 - RFX5 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/RFX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/RFX5_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / RFX5',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/RFX5.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/RFX5.eps')

# H1hesc - fdr_4 - fdr_4 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/fdr_4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/fdr_4_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / All TFs',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/fdr_4.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/fdr_4.eps')

# H1hesc - fdr_4 - POU5F1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/POU5F1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/POU5F1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / POU5F1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/POU5F1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/POU5F1.eps')

# H1hesc - fdr_4 - EGR1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/EGR1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/EGR1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / EGR1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/EGR1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/EGR1.eps')

# H1hesc - fdr_4 - RAD21 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/RAD21_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / RAD21',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/RAD21.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/RAD21.eps')

# H1hesc - fdr_4 - ZNF143 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/ZNF143.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/ZNF143_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / ZNF143',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/ZNF143.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/ZNF143.eps')

# H1hesc - fdr_4 - P300 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/P300.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/P300_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / p300',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/P300.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/P300.eps')

# H1hesc - fdr_4 - CEBPB 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/CEBPB_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / CEBPB',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/CEBPB.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/CEBPB.eps')

# H1hesc - fdr_4 - USF1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/USF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/USF1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / USF1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/USF1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/USF1.eps')

# H1hesc - fdr_4 - BRCA1 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/BRCA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/BRCA1_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / BRCA1',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/BRCA1.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/BRCA1.eps')

# H1hesc - fdr_4 - ATF3 
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/ATF3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)
bx=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/ATF3_box.txt',header=TRUE)
bx=replace(bx, abs(bx) > 100, NA)
bx2 = abs(bx[,c(1,3,8,10,4,6)])
methodNameVec = c('Boyle','Neph','H-HMM(2)','H-HMM(3)','DH-HMM(2)','DH-HMM(3)')
colnames(bx2) = methodNameVec
test_df2 <- w2l(bx2)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('H1-hESC / ATF3',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\nLocations to TFBSs (bp)',cex=1.8),
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
write(toWrite,file='/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/ATF3.txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\t')
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/H1hesc/fdr_4/ATF3.eps')

