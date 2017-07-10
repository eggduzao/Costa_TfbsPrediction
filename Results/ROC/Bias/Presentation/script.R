library(plotrix)
cols=rainbow(5)
styVec=c(1,1,1,1,1)
segLen=c(rep(2,5),rep(.3,5))
legendStyVec=c(styVec,rep(1,5))
legendCol=c(cols,rep('white',5))
mar.default <- c(5,5,4,2)
source('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Presentation/myLegend2.R')

# K562 - GABP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/GABP_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Presentation/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.33,c('PWM','TC','FOS','HINT','HINT(BC) ','24.88','83.21','7.48','93.05','94.05'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.29,'METHOD',font=2,cex=1.3)
text(x=0.418,y=0.2855,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Presentation/GABP.eps')

# K562 - EGR1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/EGR1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Presentation/EGR1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / EGR1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.33,c('PWM','TC','FOS','HINT','HINT(BC) ','18.22','54.71','6.77','71.86','77.66'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.29,'METHOD',font=2,cex=1.3)
text(x=0.418,y=0.2855,'AUC(%)',font=2,cex=1.3)
for (i in (1:5)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Presentation/EGR1.eps')


