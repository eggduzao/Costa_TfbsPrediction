library(plotrix)
cols=rainbow(9)
styVec=c(1,1,1,1,1,1,1,1,1)
segLen=c(rep(2,9),rep(.3,9))
legendStyVec=c(styVec,rep(1,9))
legendCol=c(cols,rep('white',9))
mar.default <- c(5,5,4,2)
source('/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/myLegend3.R')

# H1hesc - YY1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/YY1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','42.98','72.19','28.61','81.76','83.48','78.25','78.23','60.84','58.37'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/YY1.eps')

# H1hesc - SP2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/SP2_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/SP2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SP2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','24.85','75.26','14.02','91.82','94.42','84.92','85.83','25.98','37.51'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/SP2.eps')

# H1hesc - REST 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/REST_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/REST.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / REST',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','61.73','25.98','19.31','38.2','40.54','32.59','32.6','62.27','13.84'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/REST.eps')

# H1hesc - SIX5 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/SIX5_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SIX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','42.49','76.32','28.67','87.6','89.65','83.55','83.42','48.09','61.14'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/SIX5.eps')

# H1hesc - BRCA1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/BRCA1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/BRCA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / BRCA1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','0.99','98.3','34.54','99.79','99.76','99.6','99.57','79.95','96.7'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/BRCA1.eps')

# H1hesc - TCF12 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/TCF12_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/TCF12.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / TCF12',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','7.83','59.03','12.89','76.71','80.57','68.97','68.74','14.15','44.21'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/TCF12.eps')

# H1hesc - EGR1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/EGR1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/EGR1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / EGR1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','22.53','56.19','10.93','77.05','83.67','67.8','66.68','22.96','49.66'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/EGR1.eps')

# H1hesc - RAD21 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/RAD21_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / RAD21',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','46.16','36.08','24.45','55.57','58.26','45.79','45.31','48.4','54.02'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/RAD21.eps')

# H1hesc - ATF3 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/ATF3_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/ATF3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / ATF3',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','32.58','87.46','42.92','95.19','96.0','92.84','92.85','54.69','80.21'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/ATF3.eps')

# H1hesc - NRF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/NRF1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / NRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','45.88','41.44','38.14','47.05','60.52','55.54','54.73','46.99','34.14'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/NRF1.eps')

# H1hesc - USF2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/USF2_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / USF2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','28.47','75.33','35.06','86.99','88.87','82.67','82.65','48.12','61.63'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/USF2.eps')

# H1hesc - JUND 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/JUND_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','15.65','61.2','21.11','77.32','81.26','71.1','71.07','29.57','45.31'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/JUND.eps')

# H1hesc - ZNF143 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/ZNF143_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/ZNF143.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / ZNF143',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','28.99','76.2','21.19','87.63','89.53','83.71','83.61','34.61','62.67'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/ZNF143.eps')

# H1hesc - BACH1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/BACH1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/BACH1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / BACH1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','12.44','39.69','16.48','53.05','57.17','47.5','47.46','20.19','31.08'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/BACH1.eps')

# H1hesc - SP4 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/SP4_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/SP4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SP4',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','16.25','63.84','22.37','79.09','70.12','73.11','74.93','17.08','31.43'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/SP4.eps')

# H1hesc - MAFK 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/MAFK_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / MAFK',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','33.95','25.0','12.7','34.33','37.53','30.3','30.3','42.59','13.57'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/MAFK.eps')

# H1hesc - MYC 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/MYC_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / Myc',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','9.35','86.2','27.75','94.82','95.44','92.17','92.11','14.09','76.69'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/MYC.eps')

# H1hesc - SRF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/SRF_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/SRF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SRF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','39.19','44.55','20.61','57.38','61.09','52.08','52.08','56.8','26.88'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/SRF.eps')

# H1hesc - JUN 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/JUN_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','13.51','87.33','44.43','95.23','96.4','92.81','92.79','45.34','76.36'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/JUN.eps')

# H1hesc - TBP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/TBP_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / TBP',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','21.97','93.4','48.91','96.34','96.68','95.61','95.6','71.43','89.01'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/TBP.eps')

# H1hesc - P300 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/P300_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/P300.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / p300',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','5.19','79.52','12.68','92.27','93.74','87.9','87.85','28.45','73.05'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/P300.eps')

# H1hesc - FOSL1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/FOSL1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/FOSL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / FOSL1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','15.12','90.95','22.09','99.17','99.14','96.71','96.66','26.14','84.77'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/FOSL1.eps')

# H1hesc - USF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/USF1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/USF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / USF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','25.42','41.89','22.77','55.25','58.56','49.35','49.35','40.53','29.12'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/USF1.eps')

# H1hesc - RXRA 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/RXRA_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/RXRA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / RXRA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','11.89','75.97','24.45','90.35','92.12','85.83','85.63','25.83','61.67'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/RXRA.eps')

# H1hesc - CTCF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/CTCF_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / CTCF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','46.74','42.16','25.07','62.52','65.09','52.69','52.18','49.09','59.25'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/CTCF.eps')

# H1hesc - RFX5 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/RFX5_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/RFX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / RFX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','36.24','52.91','21.61','65.84','67.84','60.4','60.39','42.35','40.74'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/RFX5.eps')

# H1hesc - POU5F1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/POU5F1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/POU5F1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / POU5F1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','43.61','78.44','40.5','89.83','91.74','86.33','86.32','59.59','60.43'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/POU5F1.eps')

# H1hesc - MAX 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/MAX_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / MAX',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','9.35','70.06','24.43','83.71','85.11','78.75','78.72','15.35','53.31'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/MAX.eps')

# H1hesc - SP1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/SP1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/SP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SP1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','21.57','66.43','13.38','84.9','88.44','77.5','76.9','23.15','61.68'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/SP1.eps')

# H1hesc - CEBPB 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/CEBPB_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / CEBPB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','16.16','39.84','19.07','49.96','53.03','45.8','45.8','28.58','24.62'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/CEBPB.eps')

# H1hesc - GABP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/GABP_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','32.12','77.54','22.85','95.65','96.91','89.41','88.55','40.27','76.56'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/H1hesc/GABP.eps')

