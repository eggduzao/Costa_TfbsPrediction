library(plotrix)
cols=rainbow(14)
styVec=c(1,1,1,1,1,1,1,1,1,1,1,1,1,1)
segLen=c(rep(2,14),rep(.3,14))
legendStyVec=c(styVec,rep(1,14))
legendCol=c(cols,rep('white',14))
mar.default <- c(5,5,4,2)
source('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend10.R')

# K562 - SRF 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/SRF_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/SRF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / SRF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','54.42','47.47','47.19','43.41','43.39','40.82','40.92','27.29','27.21','25.72','24.65','18.09','3.17','0.87'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/SRF.eps')

# K562 - CTCFL 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/CTCFL_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/CTCFL.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / CTCFL',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','77.85','71.21','67.61','63.99','58.21','64.1','63.63','48.44','47.71','47.39','33.71','33.51','15.12','8.39'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/CTCFL.eps')

# K562 - YY1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/YY1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','85.11','81.77','81.93','78.7','75.42','76.31','76.22','64.89','64.81','60.51','55.19','53.15','7.88','0.62'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/YY1.eps')

# K562 - E2F6 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/E2F6_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/E2F6.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / E2F6',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','66.43','59.85','58.31','51.62','49.19','47.94','50.1','31.08','30.91','27.93','18.44','18.91','1.28','0.48'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/E2F6.eps')

# K562 - SP2 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/SP2_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/SP2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / SP2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','55.78','45.33','49.49','39.15','42.41','31.85','39.3','21.46','21.42','20.64','12.11','12.22','0.71','0.2'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/SP2.eps')

# K562 - STAT2 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/STAT2_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/STAT2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / STAT2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','32.04','26.33','25.99','21.27','19.79','19.75','19.32','9.5','9.48','9.57','8.39','5.17','1.39','0.18'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/STAT2.eps')

# K562 - MAFK 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/MAFK_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / MAFK',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','34.13','29.92','28.68','26.93','26.97','25.8','24.88','17.61','17.52','16.86','17.75','12.64','9.03','2.28'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/MAFK.eps')

# K562 - TBP 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/TBP_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / TBP',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','77.57','71.46','72.61','69.58','63.54','67.24','65.22','47.52','47.52','41.8','30.63','33.17','0.25','0.31'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/TBP.eps')

# K562 - ZBTB7A 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ZBTB7A_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ZBTB7A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / ZBTB7A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','62.9','56.65','56.68','49.25','46.05','46.1','47.47','28.61','28.54','24.24','14.91','17.21','0.19','0.19'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ZBTB7A.eps')

# K562 - CTCF 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/CTCF_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / CTCF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','84.98','81.99','79.11','78.69','78.34','78.0','78.46','70.72','69.95','71.44','61.36','58.6','49.11','17.48'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/CTCF.eps')

# K562 - PU1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/PU1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/PU1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / PU.1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','50.25','44.46','42.09','39.76','40.89','38.66','36.99','27.63','27.44','26.81','25.5','20.66','12.25','1.72'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/PU1.eps')

# K562 - ETS1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ETS1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ETS1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / ETS1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','73.45','67.09','69.25','60.07','56.28','57.32','59.04','37.18','37.18','32.33','22.04','22.97','0.26','0.18'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ETS1.eps')

# K562 - STAT1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/STAT1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/STAT1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / STAT1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','27.37','21.62','21.79','17.69','18.67','15.13','15.3','6.98','6.98','6.15','3.43','3.67','0.05','0.12'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/STAT1.eps')

# K562 - THAP1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/THAP1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/THAP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / THAP1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','48.01','44.41','46.35','38.07','37.0','31.15','36.42','19.64','19.64','17.42','10.68','11.08','0.07','0.05'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/THAP1.eps')

# K562 - NFE2 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/NFE2_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/NFE2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / NF-E2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','68.49','61.14','60.54','56.12','55.22','50.92','52.12','33.35','33.23','29.71','26.45','20.88','11.51','2.52'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/NFE2.eps')

# K562 - RAD21 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/RAD21_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / RAD21',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','81.25','75.31','69.66','69.91','70.55','66.21','66.62','52.33','51.31','53.57','38.41','36.9','35.42','15.31'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/RAD21.eps')

# K562 - MAX 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/MAX_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / MAX',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','84.91','81.3','80.51','77.24','75.54','75.62','73.47','61.23','61.18','55.57','45.96','49.7','0.6','0.87'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/MAX.eps')

# K562 - SIX5 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/SIX5_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / SIX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','69.89','62.86','32.25','57.54','56.79','54.01','55.03','36.31','36.3','34.14','21.68','23.48','3.76','0.25'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/SIX5.eps')

# K562 - TAL1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/TAL1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/TAL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / TAL1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','84.66','80.5','78.68','76.93','78.14','75.55','74.83','64.22','64.01','60.96','47.21','52.59','6.51','4.01'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/TAL1.eps')

# K562 - EGATA 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/EGATA_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/EGATA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / eGFP-GATA2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','73.84','68.09','67.52','63.32','63.15','59.73','61.02','44.24','44.13','39.41','25.78','30.8','0.55','1.24'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/EGATA.eps')

# K562 - NRF1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/NRF1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / NRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','84.99','79.99','75.48','75.5','79.79','71.93','74.59','65.05','64.32','63.46','49.95','49.89','29.7','42.49'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/NRF1.eps')

# K562 - IRF1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/IRF1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/IRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / IRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','39.23','32.96','31.78','27.79','26.36','25.8','25.87','15.05','14.95','15.23','10.7','9.01','6.71','0.4'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/IRF1.eps')

# K562 - JUN 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/JUN_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','82.17','77.37','76.53','73.61','73.89','71.17','69.46','53.74','53.7','49.85','37.04','40.13','0.67','7.69'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/JUN.eps')

# K562 - ATF3 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ATF3_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ATF3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / ATF3',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','63.75','59.8','58.17','52.28','46.75','47.5','46.91','26.44','26.44','25.73','15.23','15.32','0.81','8.97'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ATF3.eps')

# K562 - ZNF143 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ZNF143_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ZNF143.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / ZNF143',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','75.47','69.91','37.28','65.33','63.92','63.57','62.72','46.2','46.16','43.63','27.99','32.71','3.98','0.43'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ZNF143.eps')

# K562 - ARID3A 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ARID3A_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ARID3A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / ARID3A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','51.88','43.9','43.62','35.81','39.84','34.81','34.02','18.34','18.33','18.08','10.44','10.44','0.03','1.37'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ARID3A.eps')

# K562 - NFYA 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/NFYA_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/NFYA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / NF-YA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','78.02','74.81','73.7','71.34','68.89','68.93','69.7','57.39','57.26','56.45','44.41','44.77','2.96','3.39'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/NFYA.eps')

# K562 - STAT5A 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/STAT5A_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/STAT5A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / STAT5A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','65.27','57.89','57.3','54.35','57.06','48.33','48.93','30.89','30.84','28.29','20.35','19.28','0.55','1.86'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/STAT5A.eps')

# K562 - USF2 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/USF2_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / USF2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','87.76','84.18','84.19','81.24','78.57','78.73','79.4','66.56','66.54','64.43','54.7','53.91','2.19','4.19'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/USF2.eps')

# K562 - CEBPB 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/CEBPB_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / CEBPB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','47.82','41.04','38.49','37.23','36.74','36.29','34.89','28.26','27.75','27.14','23.85','21.97','8.3','3.83'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/CEBPB.eps')

# K562 - ZNF263 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ZNF263_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ZNF263.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / ZNF263',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','20.52','15.71','14.74','12.51','11.27','10.86','11.14','5.77','5.66','5.69','3.28','3.27','1.12','0.36'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ZNF263.eps')

# K562 - RFX5 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/RFX5_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/RFX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / RFX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','34.41','25.98','29.42','24.53','24.09','18.87','22.95','12.28','12.26','11.95','7.18','6.96','0.58','0.22'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/RFX5.eps')

# K562 - ELK1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ELK1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ELK1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / ELK1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','86.75','81.95','81.8','76.69','75.6','74.35','76.2','58.11','57.93','54.76','41.02','41.32','2.26','2.26'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ELK1.eps')

# K562 - EJUND 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/EJUND_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/EJUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / eGFP-JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','86.01','81.21','78.9','77.32','78.04','76.57','77.09','68.5','67.97','63.39','56.61','56.98','5.93','4.61'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/EJUND.eps')

# K562 - BACH1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/BACH1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/BACH1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / BACH1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','47.21','40.23','42.16','36.26','34.44','32.02','32.64','19.14','18.98','17.09','10.68','11.3','0.73','2.0'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/BACH1.eps')

# K562 - NFYB 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/NFYB_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/NFYB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / NF-YB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','61.49','58.26','56.52','55.32','53.9','54.41','53.31','45.85','45.59','47.59','41.82','37.91','19.38','4.49'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/NFYB.eps')

# K562 - GATA1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/GATA1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/GATA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / GATA1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','87.5','82.82','84.96','80.54','79.18','77.75','79.55','64.5','64.49','57.75','46.78','50.0','0.34','1.68'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/GATA1.eps')

# K562 - SMC3 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/SMC3_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/SMC3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / SMC3',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','89.38','85.84','82.47','82.05','81.16','80.15','79.83','67.06','66.42','67.74','53.14','51.58','43.23','18.31'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/SMC3.eps')

# K562 - GABP 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/GABP_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','90.72','87.74','86.6','84.95','83.88','84.21','84.48','75.89','75.55','73.25','63.12','64.28','11.49','5.46'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/GABP.eps')

# K562 - MAFF 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/MAFF_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/MAFF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / MAFF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','31.47','27.29','25.83','24.49','24.5','23.46','22.56','16.43','16.3','16.26','20.29','12.24','14.27','2.38'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/MAFF.eps')

# K562 - ATF1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ATF1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ATF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / ATF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','85.18','81.12','72.45','78.0','78.76','76.55','76.69','67.58','67.27','65.83','50.11','56.53','9.49','13.32'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ATF1.eps')

# K562 - EGR1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/EGR1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/EGR1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / EGR1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','70.14','60.74','56.49','53.64','55.67','53.62','54.05','44.32','43.3','42.7','32.23','32.98','9.28','4.08'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/EGR1.eps')

# K562 - USF1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/USF1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/USF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / USF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','78.83','74.13','71.87','70.64','69.86','69.74','66.74','59.54','59.31','60.72','55.5','52.49','10.42','6.42'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/USF1.eps')

# K562 - TR4 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/TR4_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/TR4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / TR4',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','19.9','14.83','16.13','12.65','12.58','9.89','11.25','4.98','4.97','4.58','2.63','2.58','0.81','0.06'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/TR4.eps')

# K562 - FOSL1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/FOSL1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/FOSL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / FOSL1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','89.64','85.96','84.37','82.43','81.81','81.73','82.35','73.28','72.92','67.37','59.87','61.56','4.59','3.8'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/FOSL1.eps')

# K562 - JUND 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/JUND_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','87.08','82.78','80.71','79.71','80.08','79.14','79.44','73.02','72.47','69.51','61.99','63.42','7.89','6.38'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/JUND.eps')

# K562 - ELF1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ELF1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ELF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / ELF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','82.35','79.07','77.55','75.6','74.65','74.85','73.03','63.67','63.56','60.91','52.2','53.91','5.46','1.94'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ELF1.eps')

# K562 - GATA2 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/GATA2_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/GATA2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / GATA2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','85.44','81.41','81.54','78.36','77.32','76.33','76.89','63.17','63.1','61.13','42.59','49.67','0.58','1.51'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/GATA2.eps')

# K562 - ZBTB33 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ZBTB33_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ZBTB33.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / ZBTB33',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','57.48','52.66','50.17','49.38','51.09','47.15','47.21','37.88','37.34','38.76','29.76','28.32','14.99','8.74'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/ZBTB33.eps')

# K562 - FOS 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/FOS_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/FOS.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / C-fos',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','86.44','81.73','81.02','76.51','76.87','74.44','76.08','60.2','60.04','53.14','43.83','44.91','1.52','1.46'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/FOS.eps')

# K562 - MYC 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/MYC_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / Myc',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','87.03','82.89','84.21','77.98','76.81','74.83','77.74','60.49','60.48','54.19','44.42','44.65','0.32','0.36'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/MYC.eps')

# K562 - EFOS 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/EFOS_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/EFOS.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / eGFP-FOS',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','85.65','80.32','77.92','75.6','75.75','74.94','75.55','65.12','64.62','59.53','52.25','52.35','3.81','3.34'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/EFOS.eps')

# K562 - NR2F2 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/NR2F2_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/NR2F2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / NR2F2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','68.9','62.68','61.26','56.42','59.11','53.99','54.65','37.88','37.74','33.16','19.94','25.22','0.69','0.73'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/NR2F2.eps')

# K562 - E2F4 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/E2F4_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/E2F4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / E2F4',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','78.81','70.8','67.22','57.66','64.88','55.0','42.64','32.31','31.8','43.9','31.82','32.23','2.61','2.16'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/E2F4.eps')

# K562 - EJUNB 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/EJUNB_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/EJUNB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / eGFP-JunB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','82.77','77.22','74.64','72.08','72.75','70.78','71.8','60.02','59.52','55.0','45.59','46.49','4.17','2.89'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/EJUNB.eps')

# K562 - MEF2A 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/MEF2A_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/MEF2A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / MEF2A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','61.84','55.9','55.54','49.56','51.47','47.62','47.36','30.84','30.82','30.43','24.07','20.01','0.89','0.3'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/MEF2A.eps')

# K562 - SP1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/SP1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/SP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / SP1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','66.08','58.12','60.36','50.38','54.54','43.86','50.12','29.72','29.68','28.04','17.54','17.71','1.16','0.38'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/SP1.eps')

# K562 - CCNT2 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/CCNT2_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/CCNT2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / CCNT2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','82.54','77.65','78.33','74.12','73.9','69.98','71.6','53.97','53.93','48.72','35.86','38.88','1.78','1.51'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/CCNT2.eps')

# K562 - BHLHE40 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/BHLHE40_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/BHLHE40.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='K562 / BHLHE40',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','70.33','65.33','63.2','61.24','60.33','59.87','54.06','45.37','45.28','46.79','40.03','39.27','3.2','3.96'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/forthesis/K562/BHLHE40.eps')

