library(plotrix)
cols=rainbow(5)
styVec=c(1,1,1,1,1)
segLen=c(rep(2,5),rep(.3,5))
legendStyVec=c(styVec,rep(1,5))
legendCol=c(cols,rep('white',5))
mar.default <- c(5,5,4,2)
source('/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/myLegend4.R')

# HeLaS3 - NFYA 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/NFYA_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/NFYA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / NF-YA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','7.51','88.86','19.45','93.5','94.1'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/NFYA.eps')

# HeLaS3 - FOS 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/FOS_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/FOS.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / C-fos',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','9.56','79.64','40.27','93.56','94.71'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/FOS.eps')

# HeLaS3 - REST 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/REST_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/REST.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / REST',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','66.88','31.16','20.59','42.5','44.22'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/REST.eps')

# HeLaS3 - ELK1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/ELK1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/ELK1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / ELK1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','7.59','88.83','24.53','97.45','98.42'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/ELK1.eps')

# HeLaS3 - NFYB 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/NFYB_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/NFYB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / NF-YB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','19.84','66.52','16.66','73.07','74.57'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/NFYB.eps')

# HeLaS3 - BRCA1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/BRCA1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/BRCA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / BRCA1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','2.61','94.79','29.06','97.8','98.13'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/BRCA1.eps')

# HeLaS3 - RAD21 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/RAD21_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / RAD21',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','49.29','58.39','29.16','73.39','75.49'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/RAD21.eps')

# HeLaS3 - NRF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/NRF1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / NRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','48.04','70.38','54.99','68.46','70.23'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/NRF1.eps')

# HeLaS3 - USF2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/USF2_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / USF2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','24.46','74.02','30.38','82.98','83.56'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/USF2.eps')

# HeLaS3 - E2F6 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/E2F6_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/E2F6.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / E2F6',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','17.63','90.03','17.42','98.35','98.6'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/E2F6.eps')

# HeLaS3 - JUND 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/JUND_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','15.92','64.24','31.87','78.13','79.88'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/JUND.eps')

# HeLaS3 - ZNF143 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/ZNF143_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/ZNF143.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / ZNF143',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','34.13','91.01','28.94','95.71','96.39'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/ZNF143.eps')

# HeLaS3 - MAFK 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/MAFK_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / MAFK',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','31.08','33.85','16.63','41.21','43.38'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/MAFK.eps')

# HeLaS3 - MYC 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/MYC_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / Myc',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','7.6','95.54','24.39','98.73','98.89'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/MYC.eps')

# HeLaS3 - JUN 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/JUN_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','12.17','76.75','35.14','86.7','87.85'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/JUN.eps')

# HeLaS3 - TBP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/TBP_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / TBP',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','16.31','94.33','45.11','97.03','97.36'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/TBP.eps')

# HeLaS3 - P300 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/P300_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/P300.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / p300',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','6.75','71.88','11.22','84.82','87.11'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/P300.eps')

# HeLaS3 - CTCF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/CTCF_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / CTCF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','48.8','56.98','26.78','73.25','74.54'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/CTCF.eps')

# HeLaS3 - MAX 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/MAX_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / MAX',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','7.01','90.72','25.11','95.52','96.0'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/MAX.eps')

# HeLaS3 - E2F4 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/E2F4_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/E2F4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / E2F4',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','10.27','64.37','14.04','83.54','90.22'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/E2F4.eps')

# HeLaS3 - STAT1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/STAT1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/STAT1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / STAT1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','20.58','52.59','20.92','61.97','64.45'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/STAT1.eps')

# HeLaS3 - CEBPB 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/CEBPB_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / CEBPB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','13.03','55.41','26.11','66.4','69.52'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/CEBPB.eps')

# HeLaS3 - GABP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HeLaS3/GABP_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HeLa-S3 / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','27.26','87.8','24.52','95.71','96.43'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HeLaS3/GABP.eps')

