library(plotrix)
cols=rainbow(9)
styVec=c(1,1,1,1,1,1,1,1,1)
segLen=c(rep(2,9),rep(.3,9))
legendStyVec=c(styVec,rep(1,9))
legendCol=c(cols,rep('white',9))
mar.default <- c(5,5,4,2)
source('/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/myLegend3.R')

# H1hesc - GABP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/GABP_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/H_GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FS','HINT','HINT-BC','Boyle','Neph','Cuellar','Centipede','32.12','77.54','22.85','95.65','96.91','89.41','88.55','40.27','76.56'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/H_GABP.eps')

# H1hesc - JUN 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/JUN_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/H_JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FS','HINT','HINT-BC','Boyle','Neph','Cuellar','Centipede','13.51','87.33','44.43','95.23','96.4','92.81','92.79','45.34','76.36'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/H_JUN.eps')

# H1hesc - SIX5 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/SIX5_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/H_SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SIX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FS','HINT','HINT-BC','Boyle','Neph','Cuellar','Centipede','42.49','76.32','28.67','87.6','89.65','83.55','83.42','48.09','61.14'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/H_SIX5.eps')

# H1hesc - YY1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/H1hesc/YY1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/H_YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FS','HINT','HINT-BC','Boyle','Neph','Cuellar','Centipede','42.98','72.19','28.61','81.76','83.48','78.25','78.23','60.84','58.37'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/H_YY1.eps')

###################################

# K562 - GABP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/GABP_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/K_GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FS','HINT','HINT-BC','Boyle','Neph','Cuellar','Centipede','24.88','83.21','7.48','93.05','94.05','90.27','90.55','31.44','77.36'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/K_GABP.eps')

# K562 - JUN 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/JUN_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/K_JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FS','HINT','HINT-BC','Boyle','Neph','Cuellar','Centipede','12.76','90.7','28.23','96.53','97.21','95.24','95.06','26.5','85.41'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/K_JUN.eps')

# K562 - SIX5 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/SIX5_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/K_SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SIX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FS','HINT','HINT-BC','Boyle','Neph','Cuellar','Centipede','41.16','89.78','8.55','95.81','96.64','94.12','94.11','44.17','83.34'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/K_SIX5.eps')

# K562 - YY1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/YY1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/K_YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FS','HINT','HINT-BC','Boyle','Neph','Cuellar','Centipede','54.18','92.06','15.23','95.69','96.18','94.61','94.65','68.04','85.83'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/Figure/K_YY1.eps')


