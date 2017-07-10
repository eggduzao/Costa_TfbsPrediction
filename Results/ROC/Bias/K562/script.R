library(plotrix)
cols=rainbow(9)
styVec=c(1,1,1,1,1,1,1,1,1)
segLen=c(rep(2,9),rep(.3,9))
legendStyVec=c(styVec,rep(1,9))
legendCol=c(cols,rep('white',9))
mar.default <- c(5,5,4,2)
source('/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/myLegend3.R')

# K562 - YY1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/YY1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','54.18','92.06','15.23','95.69','96.18','94.61','94.65','68.04','85.83'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/YY1.eps')

# K562 - ZBTB7A 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/ZBTB7A_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ZBTB7A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ZBTB7A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','2.54','87.22','1.97','95.24','99.26','93.14','93.21','3.76','80.24'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ZBTB7A.eps')

# K562 - SP2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/SP2_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/SP2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SP2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','27.19','90.08','9.27','88.28','98.28','96.54','95.49','28.27','74.87'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/SP2.eps')

# K562 - PU1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/PU1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/PU1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / PU.1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','36.16','49.76','6.56','62.06','64.36','57.84','57.24','39.15','41.12'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/PU1.eps')

# K562 - NFYA 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/NFYA_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/NFYA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NF-YA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','20.44','79.83','20.66','86.71','87.55','84.68','84.75','20.67','73.64'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/NFYA.eps')

# K562 - FOS 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/FOS_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/FOS.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / C-fos',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','10.16','89.16','9.26','97.43','98.15','95.86','95.61','17.3','83.21'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/FOS.eps')

# K562 - TAL1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/TAL1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/TAL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / TAL1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','24.31','80.77','13.16','90.68','92.01','88.49','87.62','37.86','75.87'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/TAL1.eps')

# K562 - REST 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/REST_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/REST.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / REST',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','62.3','43.36','17.64','55.52','58.13','50.68','50.85','62.46','32.8'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/REST.eps')

# K562 - ELK1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/ELK1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ELK1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ELK1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','7.26','87.19','7.67','95.05','98.31','96.08','95.77','16.84','75.55'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ELK1.eps')

# K562 - ARID3A 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/ARID3A_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ARID3A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ARID3A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','1.32','94.01','42.3','97.87','98.12','96.96','96.93','1.32','91.96'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ARID3A.eps')

# K562 - STAT5A 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/STAT5A_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/STAT5A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / STAT5A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','10.4','82.26','22.48','90.64','91.82','89.27','88.21','22.83','78.25'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/STAT5A.eps')

# K562 - SIX5 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/SIX5_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SIX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','41.16','89.78','8.55','95.81','96.64','94.12','94.11','44.17','83.34'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/SIX5.eps')

# K562 - NFYB 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/NFYB_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/NFYB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NF-YB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','37.28','56.14','10.13','64.86','65.76','61.78','61.85','38.05','51.8'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/NFYB.eps')

# K562 - GATA1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/GATA1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/GATA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / GATA1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','10.8','96.54','32.12','98.82','99.01','98.45','98.42','25.43','91.39'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/GATA1.eps')

# K562 - EJUNB 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/EJUNB_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/EJUNB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / eGFP-JunB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','13.96','76.72','8.73','89.41','91.24','85.82','85.38','18.31','74.13'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/EJUNB.eps')

# K562 - SMC3 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/SMC3_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/SMC3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SMC3',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','54.79','78.89','25.09','95.16','95.73','91.74','90.97','55.24','81.59'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/SMC3.eps')

# K562 - EGR1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/EGR1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/EGR1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / EGR1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','18.22','54.71','6.77','71.86','77.66','66.76','65.89','18.46','52.6'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/EGR1.eps')

# K562 - RAD21 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/RAD21_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / RAD21',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','54.83','71.8','26.14','91.47','81.95','87.91','85.78','55.2','75.65'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/RAD21.eps')

# K562 - ATF3 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/ATF3_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ATF3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ATF3',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','62.91','97.92','67.97','99.32','99.46','99.26','99.27','71.46','96.0'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ATF3.eps')

# K562 - ETS1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/ETS1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ETS1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ETS1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','12.05','96.63','3.76','99.31','99.56','99.01','99.13','15.93','91.71'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ETS1.eps')

# K562 - GATA2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/GATA2_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/GATA2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / GATA2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','7.46','91.18','17.57','96.41','97.1','95.11','95.07','34.68','87.6'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/GATA2.eps')

# K562 - NRF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/NRF1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','43.58','68.73','51.56','60.84','89.72','87.02','78.86','44.27','62.25'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/NRF1.eps')

# K562 - EJUND 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/EJUND_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/EJUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / eGFP-JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','13.58','76.54','9.65','88.67','90.67','85.2','84.67','16.76','74.04'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/EJUND.eps')

# K562 - USF2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/USF2_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / USF2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','29.36','92.83','28.01','97.46','97.85','96.45','96.47','38.9','89.81'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/USF2.eps')

# K562 - EFOS 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/EFOS_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/EFOS.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / eGFP-FOS',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','10.67','78.81','9.33','90.93','92.77','87.32','87.05','18.83','75.8'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/EFOS.eps')

# K562 - E2F6 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/E2F6_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/E2F6.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / E2F6',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','16.36','83.34','4.66','94.41','95.34','91.8','91.72','20.24','80.25'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/E2F6.eps')

# K562 - JUND 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/JUND_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','13.41','75.18','9.64','87.06','89.19','83.53','83.06','16.55','73.24'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/JUND.eps')

# K562 - ELF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/ELF1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ELF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ELF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','21.72','79.87','6.14','88.79','89.27','86.05','86.01','31.12','72.84'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ELF1.eps')

# K562 - ZNF143 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/ZNF143_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ZNF143.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ZNF143',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','29.83','85.22','6.02','92.05','93.18','90.05','90.05','32.39','79.63'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ZNF143.eps')

# K562 - NFE2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/NFE2_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/NFE2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NF-E2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','63.61','86.53','27.03','95.94','96.58','93.9','93.53','67.65','82.49'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/NFE2.eps')

# K562 - BACH1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/BACH1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/BACH1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / BACH1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','12.59','66.62','21.2','78.06','79.22','75.0','74.37','16.14','63.79'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/BACH1.eps')

# K562 - MAFK 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/MAFK_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / MAFK',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','32.32','37.1','11.79','45.51','47.49','42.2','42.08','41.27','31.2'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/MAFK.eps')

# K562 - ATF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/ATF1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ATF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ATF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','26.74','79.64','25.56','89.42','90.96','86.86','86.34','40.95','76.19'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ATF1.eps')

# K562 - MYC 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/MYC_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / Myc',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','8.05','96.8','6.98','99.44','99.59','99.09','99.08','9.47','94.93'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/MYC.eps')

# K562 - TR4 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/TR4_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/TR4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / TR4',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','54.17','82.68','11.14','90.09','91.5','88.26','88.14','55.07','76.41'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/TR4.eps')

# K562 - SRF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/SRF_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/SRF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SRF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','30.92','71.87','6.42','81.95','84.07','78.7','78.41','40.88','65.67'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/SRF.eps')

# K562 - JUN 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/JUN_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','12.76','90.7','28.23','96.53','97.21','95.24','95.06','26.5','85.41'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/JUN.eps')

# K562 - TBP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/TBP_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / TBP',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','17.92','95.46','20.35','97.49','97.67','97.22','97.04','64.98','93.0'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/TBP.eps')

# K562 - BHLHE40 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/BHLHE40_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/BHLHE40.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / BHLHE40',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','19.02','68.14','15.21','79.44','81.1','75.54','75.41','39.7','53.96'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/BHLHE40.eps')

# K562 - CTCFL 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/CTCFL_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/CTCFL.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / CTCFL',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','41.2','75.28','22.73','89.77','92.52','84.98','86.72','41.46','70.38'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/CTCFL.eps')

# K562 - IRF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/IRF1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/IRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / IRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','40.5','63.58','11.97','76.16','79.52','71.48','71.45','41.58','58.43'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/IRF1.eps')

# K562 - FOSL1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/FOSL1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/FOSL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / FOSL1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','13.02','83.64','9.36','93.34','94.51','90.76','90.64','15.67','79.99'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/FOSL1.eps')

# K562 - NR2F2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/NR2F2_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/NR2F2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NR2F2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','7.5','78.7','6.26','88.84','90.19','86.32','85.64','19.13','74.33'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/NR2F2.eps')

# K562 - USF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/USF1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/USF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / USF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','26.94','70.74','16.71','82.21','84.18','78.1','78.04','40.13','63.15'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/USF1.eps')

# K562 - THAP1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/THAP1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/THAP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / THAP1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','6.8','91.66','1.45','95.56','96.02','94.96','94.92','19.77','86.91'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/THAP1.eps')

# K562 - MAFF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/MAFF_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/MAFF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / MAFF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','36.69','31.54','9.13','39.99','42.18','36.52','36.42','45.15','25.85'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/MAFF.eps')

# K562 - CTCF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/CTCF_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / CTCF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','46.94','65.87','15.66','83.08','83.65','77.84','77.46','47.88','69.64'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/CTCF.eps')

# K562 - MEF2A 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/MEF2A_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/MEF2A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / MEF2A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','35.74','86.74','15.28','92.65','93.53','91.12','90.94','54.83','80.08'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/MEF2A.eps')

# K562 - RFX5 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/RFX5_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/RFX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / RFX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','30.87','69.58','8.43','78.08','79.6','75.7','75.2','33.31','63.9'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/RFX5.eps')

# K562 - STAT2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/STAT2_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/STAT2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / STAT2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','47.68','78.33','18.16','87.59','89.37','84.63','84.54','61.28','72.85'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/STAT2.eps')

# K562 - CCNT2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/CCNT2_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/CCNT2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / CCNT2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','26.92','92.66','19.95','97.94','98.32','97.31','96.78','44.13','88.46'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/CCNT2.eps')

# K562 - EGATA 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/EGATA_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/EGATA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / eGFP-GATA2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','7.85','84.38','16.68','92.75','93.82','90.83','90.41','30.58','81.13'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/EGATA.eps')

# K562 - ZNF263 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/ZNF263_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ZNF263.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ZNF263',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','25.88','49.4','8.87','63.21','66.97','57.94','57.83','25.88','42.19'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ZNF263.eps')

# K562 - MAX 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/MAX_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / MAX',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','5.82','89.05','9.73','95.49','95.9','93.91','93.82','7.24','81.48'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/MAX.eps')

# K562 - E2F4 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/E2F4_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/E2F4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / E2F4',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','8.63','75.39','5.94','77.39','94.26','88.69','84.87','9.26','40.34'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/E2F4.eps')

# K562 - STAT1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/STAT1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/STAT1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / STAT1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','14.68','87.45','20.16','93.62','94.2','91.91','91.82','15.83','82.82'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/STAT1.eps')

# K562 - SP1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/SP1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/SP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SP1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','24.19','90.05','10.73','93.16','97.91','96.68','95.78','25.59','77.64'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/SP1.eps')

# K562 - ZBTB33 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/ZBTB33_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ZBTB33.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ZBTB33',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','23.82','48.41','17.01','59.52','63.34','58.81','56.18','26.61','37.99'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/ZBTB33.eps')

# K562 - CEBPB 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/CEBPB_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / CEBPB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','20.8','42.49','14.5','54.59','58.81','49.49','49.45','30.04','29.36'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/CEBPB.eps')

# K562 - GABP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/K562/GABP_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,19,20,21,22,27,28,29,30,31,32,33,34)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.54,c('PWM','TC','FOS','HINT','HINT(BC)','Boyle','Neph','Cuellar','Centipede','24.88','83.21','7.48','93.05','94.05','90.27','90.55','31.44','77.36'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.50,'METHOD',font=2,cex=1.3)
text(x=0.41,y=0.4955,'AUC(%)',font=2,cex=1.3)
for (i in (1:9)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Bias/K562/GABP.eps')

