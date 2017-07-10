library(plotrix)
cols=rainbow(5)
styVec=c(1,1,1,1,1)
segLen=c(rep(2,5),rep(.3,5))
legendStyVec=c(styVec,rep(1,5))
legendCol=c(cols,rep('white',5))
mar.default <- c(5,5,4,2)
source('/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/myLegend4.R')

# HepG2 - YY1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/YY1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','41.72','91.0','38.83','94.91','95.49'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/YY1.eps')

# HepG2 - SP2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/SP2_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/SP2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / SP2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','19.91','74.32','12.8','84.54','88.68'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/SP2.eps')

# HepG2 - REST 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/REST_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/REST.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / REST',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','70.63','47.37','30.18','61.44','63.41'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/REST.eps')

# HepG2 - ARID3A 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/ARID3A_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/ARID3A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / ARID3A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','2.03','87.05','37.45','93.01','94.02'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/ARID3A.eps')

# HepG2 - BRCA1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/BRCA1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/BRCA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / BRCA1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','0.0','92.39','33.62','93.14','93.11'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/BRCA1.eps')

# HepG2 - RAD21 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/RAD21_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / RAD21',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','47.08','59.64','26.58','74.89','76.07'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/RAD21.eps')

# HepG2 - ATF3 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/ATF3_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/ATF3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / ATF3',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','57.7','97.85','67.1','99.14','99.33'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/ATF3.eps')

# HepG2 - NRF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/NRF1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / NRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','48.87','68.97','54.88','67.63','87.78'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/NRF1.eps')

# HepG2 - USF2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/USF2_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / USF2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','28.61','62.85','29.86','70.83','71.77'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/USF2.eps')

# HepG2 - JUND 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/JUND_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','13.41','70.68','25.18','83.14','86.02'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/JUND.eps')

# HepG2 - ELF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/ELF1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/ELF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / ELF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','17.28','86.04','24.08','92.92','93.05'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/ELF1.eps')

# HepG2 - MAFK 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/MAFK_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / MAFK',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','28.34','11.44','7.37','17.0','19.72'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/MAFK.eps')

# HepG2 - MYC 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/MYC_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / Myc',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','7.86','95.98','27.76','99.17','99.33'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/MYC.eps')

# HepG2 - SRF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/SRF_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/SRF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / SRF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','32.23','59.47','21.46','69.81','72.73'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/SRF.eps')

# HepG2 - JUN 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/JUN_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','32.8','30.5','18.21','37.57','39.18'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/JUN.eps')

# HepG2 - TBP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/TBP_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / TBP',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','19.12','97.85','57.09','99.36','99.5'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/TBP.eps')

# HepG2 - P300 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/P300_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/P300.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / p300',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','6.87','77.03','11.02','88.04','89.76'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/P300.eps')

# HepG2 - BHLHE40 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/BHLHE40_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/BHLHE40.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / BHLHE40',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','20.29','87.86','34.2','94.25','94.74'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/BHLHE40.eps')

# HepG2 - USF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/USF1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/USF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / USF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','27.3','58.74','24.6','70.77','72.62'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/USF1.eps')

# HepG2 - RXRA 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/RXRA_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/RXRA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / RXRA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','12.36','80.23','25.96','91.25','92.49'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/RXRA.eps')

# HepG2 - MAFF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/MAFF_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/MAFF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / MAFF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','33.88','15.22','8.44','21.44','24.2'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/MAFF.eps')

# HepG2 - CTCF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/CTCF_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / CTCF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','47.45','60.14','25.78','75.73','76.62'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/CTCF.eps')

# HepG2 - MAX 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/MAX_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / MAX',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','4.65','88.29','27.94','93.34','93.94'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/MAX.eps')

# HepG2 - SP1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/SP1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/SP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / SP1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','17.74','81.74','19.6','91.99','94.22'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/SP1.eps')

# HepG2 - CEBPB 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/CEBPB_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / CEBPB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','11.5','72.73','35.61','81.89','83.69'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/CEBPB.eps')

# HepG2 - GABP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/HepG2/GABP_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='HepG2 / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','25.5','85.73','24.15','93.46','94.5'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/HepG2/GABP.eps')

