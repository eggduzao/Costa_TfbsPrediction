library(plotrix)
#cols=rainbow(9)
styVec=c(1,1,1,1,1,1,1,1,1)
segLen=c(rep(2,9),rep(.3,9))
legendStyVec=c(styVec,rep(1,9))
legendCol=c(cols,rep('white',9))
mar.default <- c(5,5,4,2)
source('/home/egg/Projects/TfbsPrediction/Report/regGenSIG2015/Figure/myLegend.R')

cols = rev(c("#FF6600FF", "#000000FF", "#00FFFFFF", "#FF00BFFF", "#AAAAAAFF", "#0040FFFF", "#FFBF00FF", "#FF0000FF", "#00FF40FF"))

# K562 - CTCFL 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/CTCFL_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Report/regGenSIG2015/Figure/CTCFL_woLabel.eps',width=6.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(21,22, 19,20, 27,28, 29,30, 3,4, 33,34,31,32, 1,2, 7,8)]
#plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='CTCFL',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
plot(roc[,1],roc[,2],type='n',ylab='',xlab='',main='',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0, xaxt="n", yaxt="n")
myLegend(-0.035,0.54,c('HINT(BC)','HINT','Boyle','Neph','TC','Centipede','Cuellar','PWM','FOS','92.52','89.77','84.98','86.72','70.38','75.28','41.46','41.2','22.73'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
axis(1, at=c(0,0.2,0.4,0.6,0.8,1.0), labels=c("","","","","",""))
axis(2, at=c(0,0.2,0.4,0.6,0.8,1.0), labels=c("","","","","",""))
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
#myLegend(-0.035,0.54,c("","","","","","","","","","","","","","","","","",""),
#         lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen, bg="white")
myLegend(-0.035,0.54,c('HINT(BC)','HINT','Boyle','Neph','TC','Centipede','Cuellar','PWM','FOS','92.52','89.77','84.98','86.72','70.38','75.28','41.46','41.2','22.73'),
lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen, bg="white")
text(x=0.26,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.51,y=0.4955,'AUC(%)',font=2,cex=1.3)
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Report/regGenSIG2015/Figure/CTCFL_woLabel.eps')

# K562 - ZBTB7A 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/ZBTB7A_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Report/regGenSIG2015/Figure/ZBTB7A_woLabel.eps',width=6.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(21,22, 19,20, 27,28, 29,30, 3,4, 33,34,31,32, 1,2, 7,8)]
#plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='ZBTB7A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
plot(roc[,1],roc[,2],type='n',ylab='',xlab='',main='',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0, xaxt="n", yaxt="n")
axis(1, at=c(0,0.2,0.4,0.6,0.8,1.0), labels=c("","","","","",""))
axis(2, at=c(0,0.2,0.4,0.6,0.8,1.0), labels=c("","","","","",""))
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
#myLegend(-0.035,0.54,c("","","","","","","","","","","","","","","","","",""),
#         lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen, bg="white")
myLegend(-0.035,0.54,c('HINT(BC)','HINT','Boyle','Neph','TC','Centipede','Cuellar','PWM','FOS','99.26','95.24','93.14','93.21','87.22','80.24','3.76','2.54','1.97'),
lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen, bg="white")
text(x=0.26,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.51,y=0.4955,'AUC(%)',font=2,cex=1.3)
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Report/regGenSIG2015/Figure/ZBTB7A_woLabel.eps')


