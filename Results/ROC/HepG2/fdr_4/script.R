library(plotrix)
cols=rep(rainbow(5),2)
cols=rainbow(5)
styVec=c(2,1,1,1,1)
pointStyVec=c(22,21,21,21,21)
segLen=c(rep(2,5),rep(.3,5))
legendStyVec=c(styVec,rep(1,5))
legendCol=c(cols,rep('white',5))
mar.default <- c(5,5,4,2)
source('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/LabelTranslation/myLegend2.R')

# HepG2 - fdr_4 - YY1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/YY1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','79.29','95.27','95.0','95.44','94.49'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9871,0.7873,0.9814,0.7937,0.9582,0.8428,0.9907,0.7525)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/YY1.eps')

# HepG2 - fdr_4 - SP2 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/SP2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/SP2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / SP2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','65.86','87.94','88.86','85.49','87.42'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.877,0.8403,0.8644,0.8688,0.8186,0.826,0.8797,0.8234)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/SP2.eps')

# HepG2 - fdr_4 - REST 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/REST_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/REST.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / REST',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','87.96','91.39','91.0','89.59','91.18'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9339,0.465,0.9239,0.4663,0.8838,0.5054,0.9377,0.4297)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/REST.eps')

# HepG2 - fdr_4 - ARID3A 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/ARID3A_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/ARID3A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / ARID3A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','45.81','71.94','70.78','79.06','63.96'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9943,0.4847,0.9905,0.4708,0.9747,0.6267,0.998,0.3324)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/ARID3A.eps')

# HepG2 - fdr_4 - BRCA1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/BRCA1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/BRCA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / BRCA1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','45.28','88.94','93.11','86.12','85.7'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9796,0.8667,0.973,0.9333,0.9481,0.8,0.9852,0.7333)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/BRCA1.eps')

# HepG2 - fdr_4 - RAD21 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/RAD21_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / RAD21',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','83.31','90.24','89.45','88.45','89.7'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9176,0.6294,0.9067,0.6111,0.8655,0.646,0.9212,0.5869)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/RAD21.eps')

# HepG2 - fdr_4 - ATF3 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/ATF3_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/ATF3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / ATF3',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','90.15','97.77','98.8','98.04','98.03'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9788,0.9149,0.9728,0.9645,0.9433,0.9574,0.9833,0.8865)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/ATF3.eps')

# HepG2 - fdr_4 - NRF1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/NRF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / NRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','85.91','93.59','94.12','92.15','93.09'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.7178,0.9489,0.7006,0.9665,0.6376,0.9443,0.7287,0.9276)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/NRF1.eps')

# HepG2 - fdr_4 - USF2 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/USF2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / USF2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','77.77','89.15','88.52','87.77','88.37'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9713,0.5111,0.9641,0.5055,0.9339,0.542,0.9757,0.4736)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/USF2.eps')

# HepG2 - fdr_4 - JUND 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/JUND_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','67.18','86.09','84.86','86.45','83.3'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.97,0.5926,0.9633,0.5675,0.9376,0.6526,0.9762,0.5002)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/JUND.eps')

# HepG2 - fdr_4 - ELF1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/ELF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/ELF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / ELF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','65.63','92.47','91.87','91.51','91.01'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9666,0.7791,0.9586,0.766,0.9274,0.8071,0.9726,0.7288)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/ELF1.eps')

# HepG2 - fdr_4 - MAFK 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/MAFK_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / MAFK',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','73.57','74.27','74.09','73.65','74.21'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9852,0.0544,0.98,0.0605,0.9598,0.0937,0.9894,0.0415)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/MAFK.eps')

# HepG2 - fdr_4 - MYC 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/MYC_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / Myc',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','53.5','91.12','91.44','89.12','89.86'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9669,0.8381,0.9595,0.8449,0.9303,0.8321,0.9715,0.8046)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/MYC.eps')

# HepG2 - fdr_4 - SRF 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/SRF_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/SRF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / SRF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','76.4','87.81','87.31','88.7','86.46'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9863,0.4026,0.9805,0.4024,0.9589,0.5119,0.9906,0.3436)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/SRF.eps')

# HepG2 - fdr_4 - JUN 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/JUN_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','77.72','84.03','83.48','83.18','83.18'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9792,0.2211,0.9732,0.2245,0.9491,0.2751,0.9841,0.1852)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/JUN.eps')

# HepG2 - fdr_4 - TBP 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/TBP_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / TBP',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','68.15','94.85','94.75','96.42','93.05'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9913,0.8563,0.9869,0.8465,0.9702,0.9213,0.9949,0.8012)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/TBP.eps')

# HepG2 - fdr_4 - P300 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/P300_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/P300.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / p300',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','51.16','85.54','84.71','84.64','82.26'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9498,0.7511,0.9404,0.7445,0.9046,0.7775,0.9551,0.6883)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/P300.eps')

# HepG2 - fdr_4 - BHLHE40 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/BHLHE40_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/BHLHE40.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / BHLHE40',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','66.13','90.92','89.38','89.46','88.66'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9659,0.7755,0.9578,0.7466,0.925,0.7811,0.9707,0.7134)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/BHLHE40.eps')

# HepG2 - fdr_4 - USF1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/USF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/USF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / USF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','79.2','87.79','87.27','87.3','86.85'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9783,0.4213,0.971,0.4191,0.9424,0.4883,0.9817,0.3735)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/USF1.eps')

# HepG2 - fdr_4 - RXRA 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/RXRA_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/RXRA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / RXRA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','54.04','86.76','84.42','85.98','82.45'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9592,0.7448,0.9521,0.7037,0.9185,0.772,0.9666,0.6478)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/RXRA.eps')

# HepG2 - fdr_4 - MAFF 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/MAFF_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/MAFF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / MAFF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','76.59','78.13','77.84','77.36','77.96'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.985,0.0811,0.9795,0.0867,0.9592,0.123,0.9892,0.0636)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/MAFF.eps')

# HepG2 - fdr_4 - CTCF 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/CTCF_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / CTCF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','83.28','90.36','89.59','88.54','89.89'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9213,0.6215,0.9103,0.6046,0.8691,0.639,0.9249,0.5818)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/CTCF.eps')

# HepG2 - fdr_4 - MAX 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/MAX_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / MAX',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','43.46','83.56','82.07','82.8','81.0'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9776,0.7261,0.9715,0.7034,0.9474,0.7446,0.9818,0.6723)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/MAX.eps')

# HepG2 - fdr_4 - SP1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/SP1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/SP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / SP1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','64.51','90.51','90.93','88.07','88.95'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8987,0.8733,0.8869,0.9002,0.8432,0.8672,0.9022,0.8302)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/SP1.eps')

# HepG2 - fdr_4 - CEBPB 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/CEBPB_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / CEBPB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','62.31','79.18','78.0','81.68','75.29'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9917,0.4069,0.9872,0.3865,0.9672,0.5235,0.9956,0.2988)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/CEBPB.eps')

# HepG2 - fdr_4 - GABP 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HepG2/fdr_4/GABP_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','69.52','92.22','92.81','90.31','91.47'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9153,0.8617,0.9047,0.8914,0.8579,0.8662,0.9238,0.8363)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HepG2/fdr_4/GABP.eps')

