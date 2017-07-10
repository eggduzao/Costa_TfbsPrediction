library(plotrix)
cols=rainbow(5)
styVec=c(1,1,1,1,1)
segLen=c(rep(2,5),rep(.3,5))
legendStyVec=c(styVec,rep(1,5))
legendCol=c(cols,rep('white',5))
mar.default <- c(5,5,4,2)
source('/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/myLegend4.R')

# Huvec - FOS 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/Huvec/FOS_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Huvec/FOS.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='Huvec / C-fos',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','9.29','69.92','33.44','82.07','83.33'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Huvec/FOS.eps')

# Huvec - GATA2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/Huvec/GATA2_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Huvec/GATA2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='Huvec / GATA2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','9.48','71.43','24.74','80.77','82.69'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Huvec/GATA2.eps')

# Huvec - MYC 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/Huvec/MYC_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Huvec/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='Huvec / Myc',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','7.44','93.71','30.05','98.14','98.31'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Huvec/MYC.eps')

# Huvec - JUN 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/Huvec/JUN_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Huvec/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='Huvec / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','18.43','80.05','37.36','87.96','88.47'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Huvec/JUN.eps')

# Huvec - CTCF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/Huvec/CTCF_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Huvec/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='Huvec / CTCF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','50.9','55.16','36.39','71.31','62.44'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Huvec/CTCF.eps')

# Huvec - MAX 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/Huvec/MAX_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Huvec/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='Huvec / MAX',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.325,c('PWM','TC','FOS','HINT','HINT(BC)','6.63','75.46','25.26','82.17','83.18'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.282,'METHOD',font=2,cex=1.3)
text(x=0.405,y=0.278,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Huvec/MAX.eps')

