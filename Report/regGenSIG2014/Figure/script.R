library(lattice)
library(reshape)
library(plotrix)
cols=rep(rainbow(5),2)
styVec=c(2,2,2,2,2,1,1,1,1,1)
pointStyVec=c(22,22,22,22,22,21,21,21,21,21)
segLen=c(rep(2,10),rep(.3,10))
legendStyVec=c(styVec,rep(1,10))
legendCol=c(cols,rep('white',10))
mar.default <- c(5,5,4,2)
source('/home/egg/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/LabelTranslation/myLegend1.R')
currFigPath = '/home/egg/Projects/TfbsPrediction/eg474423_Projects/trunk/TfbsPrediction/reports/regGenSIG2014/Figure/'

# Boxplot
w2l <- function(xx) melt(xx, measure.vars = colnames(xx))
postscript(paste(currFigPath,'boxplot.eps',sep=''),width=12.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=1.0)
bx=read.table('/home/egg/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/reports/regGenSIG2014/Figure/auc.txt',header=TRUE)
#methodNameVec = c("PWM","Boyle et al.","Neph et al.","Cuellar-Partida et al. (2)","Cuellar-Partida et al. (3)","Centipede","DH-HMM (2)","DH-HMM (3)","H-HMM (2)","H-HMM (3)")
methodNameVec = c("PWM","Boyle","Neph","Cuellar (2)","Cuellar (3)","Centipede","DH-HMM (2)","DH-HMM (3)","H-HMM (2)","H-HMM (3)")
bx = bx*100 # Converting values
selectVec = c(8,7,6,3,4,5,10,9,2,1)
bx=bx[,selectVec]
colnames(bx) = methodNameVec[selectVec]
test_df2 <- w2l(bx)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=90,cex=2.0), y=list(cex=2.0)), # label=c()
       col = 'black', main = list('',cex=1.5), xlab = list('',cex=1.5), # Signal
       ylab = list('AUC Distribution (%)',cex=2.0), # 
       ylim=c(30,105), par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(lty=1, col = 'black')),
       panel=function(...) {
         panel.abline(h=c(40,60,80,100),lty=2,lwd=1.0,col="gray") 
         panel.bwplot(...)
       })
dev.off()
system(paste('epstopdf ',currFigPath,'boxplot.eps',sep=''))

# H1hesc - fdr_4 - ZNF143 
roc=read.table('/home/egg/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/ZNF143_roc.txt',header=TRUE)
postscript(paste(currFigPath,'H_ZNF143.eps',sep=''),width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / ZNF143',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','70.16','80.55','80.52','79.04','84.76','84.05','78.37','80.73','91.62','92.23'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8885,0.7212,0.8903,0.7179,0.9808,0.5963,0.9916,0.4691,0.9941,0.4363,0.9183,0.4251,0.8302,0.6047,0.9764,0.7088,0.968,0.7417)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system(paste('epstopdf ',currFigPath,'H_ZNF143.eps',sep=''))

# K562 - fdr_4 - ZNF143 
roc=read.table('/home/egg/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/ZNF143_roc.txt',header=TRUE)
postscript(paste(currFigPath,'K_ZNF143.eps',sep=''),width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ZNF143',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','70.65','79.99','79.99','91.71','84.32','80.02','78.67','81.4','93.13','93.8'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8725,0.7248,0.8722,0.7236,0.9777,0.8154,0.9961,0.479,0.9983,0.3058,0.8373,0.5454,0.8009,0.6566,0.9627,0.7888,0.9602,0.8169)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system(paste('epstopdf ',currFigPath,'K_ZNF143.eps',sep=''))


