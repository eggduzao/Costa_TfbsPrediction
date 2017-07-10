library(plotrix)
cols=rainbow(14)
styVec=c(1,1,1,1,1,1,1,1,1,1,1,1,1,1)
segLen=c(rep(2,14),rep(.3,14))
legendStyVec=c(styVec,rep(1,14))
legendCol=c(cols,rep('white',14))
mar.default <- c(5,5,4,2)
source('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend9.R')

# K562 - YY1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/YY1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','96.18','95.28','94.81','94.72','94.61','94.65','92.99','92.41','91.99','93.28','92.13','92.06','54.18','15.23'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/YY1.eps')

# K562 - ZBTB7A 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ZBTB7A_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ZBTB7A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ZBTB7A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','96.26','94.54','93.6','93.36','93.14','93.21','93.45','91.58','91.23','76.18','69.06','87.22','2.54','1.97'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ZBTB7A.eps')

# K562 - SP2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/SP2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/SP2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SP2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','98.28','97.54','97.05','96.2','96.54','95.49','96.73','94.5','94.31','91.84','88.99','90.08','27.19','9.27'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/SP2.eps')

# K562 - PU1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/PU1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/PU1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / PU.1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','64.36','59.89','57.49','57.35','57.84','57.24','50.84','49.23','48.5','54.88','57.12','49.76','36.16','6.56'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/PU1.eps')

# K562 - NFYA 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/NFYA_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/NFYA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NF-YA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','87.55','86.07','85.09','84.93','84.68','84.75','83.58','82.12','81.82','83.53','79.02','79.83','20.44','20.66'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/NFYA.eps')

# K562 - FOS 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/FOS_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/FOS.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / C-fos',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','98.15','96.92','96.08','95.81','95.86','95.61','95.89','93.76','93.47','91.51','83.55','89.16','10.16','9.26'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/FOS.eps')

# K562 - TAL1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/TAL1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/TAL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / TAL1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','92.01','89.42','87.97','87.78','88.49','87.62','85.09','83.17','82.7','85.06','67.07','80.77','24.31','13.16'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/TAL1.eps')

# K562 - ELK1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ELK1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ELK1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ELK1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','98.31','97.6','96.95','96.05','96.08','95.77','96.46','93.26','93.07','90.35','86.33','87.19','7.26','7.67'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ELK1.eps')

# K562 - ARID3A 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ARID3A_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ARID3A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ARID3A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','98.12','97.51','97.12','97.01','96.96','96.93','95.99','95.08','94.94','96.06','94.01','94.01','1.32','42.3'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ARID3A.eps')

# K562 - STAT5A 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/STAT5A_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/STAT5A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / STAT5A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','91.82','89.65','88.55','88.4','89.27','88.21','86.6','84.86','84.5','86.14','78.51','82.26','10.4','22.48'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/STAT5A.eps')

# K562 - SIX5 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/SIX5_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SIX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','96.64','95.2','50.15','94.24','94.12','94.11','91.4','90.22','89.87','91.88','79.39','89.78','41.16','8.55'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/SIX5.eps')

# K562 - NFYB 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/NFYB_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/NFYB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NF-YB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','65.76','63.6','62.1','61.97','61.78','61.85','58.57','57.17','56.81','62.32','60.2','56.14','37.28','10.13'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/NFYB.eps')

# K562 - GATA1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/GATA1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/GATA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / GATA1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','99.01','98.73','98.61','98.49','98.45','98.42','98.37','97.71','97.66','96.63','86.58','96.54','10.8','32.12'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/GATA1.eps')

# K562 - EJUNB 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/EJUNB_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/EJUNB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / eGFP-JunB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','91.24','87.82','85.81','85.56','85.82','85.38','85.65','83.23','82.53','81.11','72.79','76.72','13.96','8.73'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/EJUNB.eps')

# K562 - SMC3 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/SMC3_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/SMC3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SMC3',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','95.73','93.86','91.57','91.54','91.74','90.97','91.96','88.26','87.47','87.75','79.91','78.89','54.79','25.09'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/SMC3.eps')

# K562 - EGR1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/EGR1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/EGR1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / EGR1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','77.66','70.16','66.43','65.93','66.76','65.89','66.74','64.12','62.66','60.9','54.51','54.71','18.22','6.77'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/EGR1.eps')

# K562 - RAD21 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/RAD21_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / RAD21',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','92.95','89.95','86.53','86.75','87.91','85.78','87.37','83.38','82.07','83.15','73.9','71.8','54.83','26.14'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/RAD21.eps')

# K562 - ATF3 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ATF3_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ATF3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ATF3',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','99.46','99.42','99.42','99.36','99.26','99.27','99.29','98.77','98.77','98.62','96.61','97.92','62.91','67.97'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ATF3.eps')

# K562 - ETS1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ETS1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ETS1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ETS1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','99.56','99.44','99.42','99.21','99.01','99.13','99.17','98.23','98.22','96.53','89.9','96.63','12.05','3.76'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ETS1.eps')

# K562 - GATA2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/GATA2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/GATA2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / GATA2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','97.1','95.92','95.32','95.18','95.11','95.07','94.59','93.38','93.18','93.91','76.71','91.18','7.46','17.57'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/GATA2.eps')

# K562 - NRF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/NRF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','89.72','85.29','81.93','82.8','87.02','78.86','79.12','76.06','75.13','78.01','68.9','68.73','43.58','51.56'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/NRF1.eps')

# K562 - EJUND 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/EJUND_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/EJUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / eGFP-JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','90.67','87.04','85.05','84.84','85.2','84.67','84.97','82.75','82.04','80.96','72.65','76.54','13.58','9.65'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/EJUND.eps')

# K562 - USF2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/USF2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / USF2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','97.85','97.18','96.74','96.6','96.45','96.47','95.36','94.22','94.04','94.73','90.83','92.83','29.36','28.01'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/USF2.eps')

# K562 - EFOS 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/EFOS_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/EFOS.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / eGFP-FOS',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','92.77','89.38','87.44','87.21','87.32','87.05','87.36','85.08','84.37','83.33','75.49','78.81','10.67','9.33'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/EFOS.eps')

# K562 - E2F6 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/E2F6_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/E2F6.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / E2F6',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','95.34','93.67','92.34','91.98','91.8','91.72','91.64','88.93','88.51','87.03','80.05','83.34','16.36','4.66'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/E2F6.eps')

# K562 - JUND 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/JUND_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','89.19','85.4','83.41','83.21','83.53','83.06','83.27','81.16','80.45','80.21','70.01','75.18','13.41','9.64'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/JUND.eps')

# K562 - ELF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ELF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ELF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ELF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','89.27','87.66','86.32','86.13','86.05','86.01','81.3','79.7','79.33','83.31','72.18','79.87','21.72','6.14'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ELF1.eps')

# K562 - ZNF143 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ZNF143_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ZNF143.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ZNF143',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','93.18','91.24','49.37','90.15','90.05','90.05','87.18','85.9','85.53','87.75','68.6','85.22','29.83','6.02'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ZNF143.eps')

# K562 - NFE2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/NFE2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/NFE2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NF-E2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','96.58','95.12','93.84','93.75','93.9','93.53','93.07','90.9','90.5','90.39','87.68','86.53','63.61','27.03'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/NFE2.eps')

# K562 - BACH1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/BACH1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/BACH1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / BACH1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','79.22','76.58','75.21','74.58','75.0','74.37','74.81','72.64','71.93','71.57','52.62','66.62','12.59','21.2'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/BACH1.eps')

# K562 - MAFK 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/MAFK_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / MAFK',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','47.49','43.92','42.24','42.15','42.2','42.08','37.07','36.06','35.68','40.96','49.47','37.1','32.32','11.79'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/MAFK.eps')

# K562 - ATF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ATF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ATF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ATF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','90.96','88.17','78.85','86.49','86.86','86.34','85.23','83.35','82.85','84.1','68.0','79.64','26.74','25.56'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ATF1.eps')

# K562 - MYC 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/MYC_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / Myc',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','99.59','99.43','99.33','99.18','99.09','99.08','98.74','97.94','97.87','97.1','95.06','96.8','8.05','6.98'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/MYC.eps')

# K562 - TR4 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/TR4_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/TR4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / TR4',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','91.5','89.6','87.58','88.38','88.26','88.14','87.96','86.34','86.02','85.6','83.97','82.68','54.17','11.14'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/TR4.eps')

# K562 - SRF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/SRF_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/SRF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SRF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','84.07','80.39','78.68','78.53','78.7','78.41','76.49','74.77','74.23','76.95','76.78','71.87','30.92','6.42'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/SRF.eps')

# K562 - JUN 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/JUN_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','97.21','95.99','95.34','95.18','95.24','95.06','92.78','91.57','91.2','92.93','71.74','90.7','12.76','28.23'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/JUN.eps')

# K562 - TBP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/TBP_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / TBP',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','97.67','97.33','97.15','97.09','97.22','97.04','96.46','95.96','95.93','95.78','92.0','95.46','17.92','20.35'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/TBP.eps')

# K562 - BHLHE40 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/BHLHE40_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/BHLHE40.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / BHLHE40',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','81.24','77.7','75.72','75.55','75.54','75.41','60.71','59.53','59.25','73.09','65.97','68.14','19.02','15.21'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/BHLHE40.eps')

# K562 - CTCFL 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/CTCFL_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/CTCFL.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / CTCFL',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','92.52','89.74','86.98','86.57','84.98','86.72','87.44','84.35','83.23','83.14','75.16','75.28','41.2','22.73'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/CTCFL.eps')

# K562 - IRF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/IRF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/IRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / IRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','79.52','74.08','71.75','71.58','71.48','71.45','70.97','69.04','68.25','70.1','71.9','63.58','40.5','11.97'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/IRF1.eps')

# K562 - FOSL1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/FOSL1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/FOSL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / FOSL1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','94.51','92.36','91.04','90.81','90.76','90.64','91.0','88.88','88.38','87.17','78.81','83.64','13.02','9.36'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/FOSL1.eps')

# K562 - NR2F2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/NR2F2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/NR2F2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NR2F2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','90.19','87.51','85.82','85.8','86.32','85.64','83.57','81.65','81.24','82.77','54.21','78.7','7.5','6.26'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/NR2F2.eps')

# K562 - USF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/USF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/USF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / USF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','84.18','80.38','78.34','78.18','78.1','78.04','71.86','70.19','69.5','76.22','69.91','70.74','26.94','16.71'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/USF1.eps')

# K562 - THAP1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/THAP1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/THAP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / THAP1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','96.02','95.64','95.28','95.07','94.96','94.92','94.55','93.43','93.33','92.58','85.79','91.66','6.8','1.45'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/THAP1.eps')

# K562 - MAFF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/MAFF_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/MAFF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / MAFF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','42.18','38.31','36.56','36.48','36.52','36.42','31.56','30.63','30.34','35.42','49.84','31.54','36.69','9.13'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/MAFF.eps')

# K562 - CTCF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/CTCF_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / CTCF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','83.65','80.83','77.87','77.76','77.84','77.46','80.0','76.79','75.76','75.51','69.25','65.87','46.94','15.66'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/CTCF.eps')

# K562 - MEF2A 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/MEF2A_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/MEF2A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / MEF2A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','93.53','91.97','91.17','91.04','91.12','90.94','89.85','88.66','88.48','89.84','84.39','86.74','35.74','15.28'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/MEF2A.eps')

# K562 - RFX5 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/RFX5_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/RFX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / RFX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','79.6','76.97','75.54','75.39','75.7','75.2','73.74','72.19','71.67','72.92','70.48','69.58','30.87','8.43'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/RFX5.eps')

# K562 - STAT2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/STAT2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/STAT2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / STAT2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','89.37','86.3','84.64','84.67','84.63','84.54','82.2','80.53','80.12','83.1','83.77','78.33','47.68','18.16'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/STAT2.eps')

# K562 - CCNT2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/CCNT2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/CCNT2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / CCNT2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','98.32','97.55','97.12','96.94','97.31','96.78','96.39','95.02','94.87','94.35','80.67','92.66','26.92','19.95'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/CCNT2.eps')

# K562 - EGATA 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/EGATA_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/EGATA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / eGFP-GATA2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','93.82','91.85','90.77','90.57','90.83','90.41','89.65','87.84','87.48','87.34','67.38','84.38','7.85','16.68'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/EGATA.eps')

# K562 - ZNF263 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ZNF263_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ZNF263.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ZNF263',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','66.97','60.91','58.17','58.02','57.94','57.83','57.57','55.58','54.46','55.68','49.49','49.4','25.88','8.87'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ZNF263.eps')

# K562 - MAX 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/MAX_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / MAX',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','95.9','94.85','94.11','93.94','93.91','93.82','88.49','87.32','86.87','91.41','76.9','89.05','5.82','9.73'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/MAX.eps')

# K562 - E2F4 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/E2F4_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/E2F4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / E2F4',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','94.26','91.53','88.72','85.29','88.69','84.87','61.17','58.68','56.27','78.21','74.54','75.39','8.63','5.94'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/E2F4.eps')

# K562 - STAT1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/STAT1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/STAT1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / STAT1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','94.2','92.92','92.08','91.93','91.91','91.82','90.52','89.25','89.08','89.54','71.83','87.45','14.68','20.16'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/STAT1.eps')

# K562 - SP1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/SP1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/SP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SP1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','97.91','97.33','96.96','96.46','96.68','95.78','96.56','94.32','94.17','92.08','89.17','90.05','24.19','10.73'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/SP1.eps')

# K562 - ZBTB33 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/ZBTB33_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ZBTB33.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ZBTB33',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','63.34','59.22','56.88','57.3','58.81','56.18','56.74','54.66','53.81','56.32','50.37','48.41','23.82','17.01'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/ZBTB33.eps')

# K562 - CEBPB 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/CEBPB_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / CEBPB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','58.81','52.2','49.63','49.53','49.49','49.45','48.74','47.17','46.03','48.16','45.28','42.49','20.8','14.5'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/CEBPB.eps')

# K562 - GABP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/K562/GABP_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,25,26,53,54,61,62,47,48,3,4,67,68,9,10,13,14,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.496,c('HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','Filter','FLR','Cuellar','TC','PWM','FS','94.05','92.28','91.06','90.72','90.27','90.55','90.46','88.2','87.76','86.8','80.49','83.21','24.88','7.48'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.465,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.463,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/forthesis/K562/GABP.eps')

