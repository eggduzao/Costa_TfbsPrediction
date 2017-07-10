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

# HeLaS3 - fdr_4 - NFYA 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/NFYA_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/NFYA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / NF-YA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','65.86','93.3','94.34','92.45','91.86'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9684,0.8039,0.9638,0.8403,0.9448,0.8129,0.9754,0.7516)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/NFYA.eps')

# HeLaS3 - fdr_4 - FOS 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/FOS_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/FOS.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / C-fos',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','62.76','88.72','86.15','87.36','85.92'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9373,0.7704,0.9351,0.7033,0.9104,0.7685,0.9525,0.676)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/FOS.eps')

# HeLaS3 - fdr_4 - REST 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/REST_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/REST.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / REST',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','87.36','88.19','87.67','85.93','87.91'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9329,0.3335,0.9257,0.3317,0.8879,0.3693,0.9392,0.2984)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/REST.eps')

# HeLaS3 - fdr_4 - ELK1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/ELK1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/ELK1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / ELK1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','60.79','92.33','93.59','90.26','92.13'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9044,0.9125,0.8959,0.954,0.8522,0.9147,0.9139,0.8957)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/ELK1.eps')

# HeLaS3 - fdr_4 - NFYB 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/NFYB_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/NFYB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / NF-YB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','71.0','89.76','90.53','88.93','88.78'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9745,0.5645,0.9704,0.5897,0.954,0.5824,0.9806,0.5233)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/NFYB.eps')

# HeLaS3 - fdr_4 - BRCA1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/BRCA1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/BRCA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / BRCA1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','49.58','80.98','86.01','81.31','81.3'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9774,0.6923,0.9728,0.7692,0.9527,0.7143,0.986,0.6484)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/BRCA1.eps')

# HeLaS3 - fdr_4 - RAD21 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/RAD21_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / RAD21',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','84.8','90.58','89.96','89.09','90.18'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9115,0.6305,0.9042,0.611,0.864,0.6528,0.918,0.5803)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/RAD21.eps')

# HeLaS3 - fdr_4 - NRF1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/NRF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / NRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','85.26','94.57','94.75','93.32','94.03'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.7582,0.9551,0.7406,0.9647,0.6763,0.9578,0.7655,0.9355)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/NRF1.eps')

# HeLaS3 - fdr_4 - USF2 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/USF2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / USF2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','74.84','89.36','88.68','88.44','88.71'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9702,0.5881,0.9649,0.5751,0.9375,0.6185,0.9773,0.551)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/USF2.eps')

# HeLaS3 - fdr_4 - E2F6 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/E2F6_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/E2F6.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / E2F6',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','66.45','91.91','92.45','89.7','91.5'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9307,0.8534,0.9233,0.8814,0.8842,0.847,0.9385,0.8364)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/E2F6.eps')

# HeLaS3 - fdr_4 - JUND 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/JUND_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','69.87','86.97','85.39','86.75','84.58'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9495,0.6017,0.9461,0.5594,0.9226,0.6399,0.962,0.504)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/JUND.eps')

# HeLaS3 - fdr_4 - ZNF143 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/ZNF143_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/ZNF143.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / ZNF143',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','75.63','95.51','95.38','94.2','94.95'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9578,0.8438,0.9522,0.8489,0.9242,0.8507,0.9642,0.8097)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/ZNF143.eps')

# HeLaS3 - fdr_4 - MAFK 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/MAFK_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / MAFK',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','74.83','81.25','80.69','81.09','80.56'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9809,0.2329,0.9772,0.224,0.9605,0.2765,0.988,0.1912)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/MAFK.eps')

# HeLaS3 - fdr_4 - MYC 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/MYC_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / Myc',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','53.7','88.54','88.49','86.58','88.95'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.965,0.7871,0.9598,0.7866,0.9333,0.7773,0.9726,0.7841)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/MYC.eps')

# HeLaS3 - fdr_4 - JUN 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/JUN_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','63.23','85.52','84.03','86.22','81.65'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9751,0.6125,0.9707,0.5732,0.9496,0.665,0.9833,0.4909)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/JUN.eps')

# HeLaS3 - fdr_4 - TBP 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/TBP_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / TBP',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','63.11','90.78','89.6','93.23','87.54'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.989,0.7754,0.9855,0.7568,0.9715,0.8526,0.9944,0.7024)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/TBP.eps')

# HeLaS3 - fdr_4 - P300 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/P300_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/P300.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / p300',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','52.53','84.42','81.97','82.83','81.87'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9463,0.7294,0.9395,0.6909,0.9067,0.7338,0.9542,0.6678)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/P300.eps')

# HeLaS3 - fdr_4 - CTCF 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/CTCF_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / CTCF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','83.88','89.86','89.37','88.4','89.58'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.919,0.6062,0.9116,0.5918,0.8712,0.6302,0.9252,0.5628)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/CTCF.eps')

# HeLaS3 - fdr_4 - MAX 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/MAX_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / MAX',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','49.37','84.2','82.99','83.75','83.22'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.976,0.7042,0.9716,0.687,0.9508,0.7193,0.983,0.6742)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/MAX.eps')

# HeLaS3 - fdr_4 - E2F4 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/E2F4_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/E2F4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / E2F4',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','62.05','87.77','89.4','84.95','86.98'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.818,0.8752,0.8053,0.9226,0.7481,0.8758,0.8233,0.8511)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/E2F4.eps')

# HeLaS3 - fdr_4 - STAT1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/STAT1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/STAT1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / STAT1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','73.14','83.42','82.87','83.59','82.21'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9778,0.3754,0.9734,0.3683,0.9534,0.437,0.9859,0.3235)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/STAT1.eps')

# HeLaS3 - fdr_4 - CEBPB 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/CEBPB_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / CEBPB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','62.54','73.61','73.26','75.94','70.35'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9916,0.264,0.9881,0.2642,0.9709,0.3648,0.9967,0.1735)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/CEBPB.eps')

# HeLaS3 - fdr_4 - GABP 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/HeLaS3/fdr_4/GABP_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.345,c('PWM','DH-HMM (ESC)','DH-HMM (HeLa)','DH-HMM (Hep)','DH-HMM (K562)  ','71.07','93.02','93.68','91.48','92.87'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.302,'METHOD',font=2,cex=1.3)
text(x=0.57,y=0.302,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9085,0.8903,0.9009,0.9189,0.859,0.8922,0.9194,0.8738)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/HeLaS3/fdr_4/GABP.eps')

