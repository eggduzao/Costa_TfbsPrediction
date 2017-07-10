library(plotrix)
cols=rainbow(15)
styVec=c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
segLen=c(rep(2,15),rep(.3,15))
legendStyVec=c(styVec,rep(1,15))
legendCol=c(cols,rep('white',15))
mar.default <- c(5,5,4,2)
source('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend5.R')

# H1hesc - YY1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/YY1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','83.48','82.83','81.76','80.22','78.47','78.35','78.23','78.25','78.5','76.86','76.36','78.67','72.19','42.98','28.61'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/YY1.eps')

# H1hesc - SP2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/SP2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/SP2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SP2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','94.42','94.06','91.82','89.57','86.79','86.26','85.83','84.92','64.54','62.43','78.04','74.73','75.26','24.85','14.02'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/SP2.eps')

# H1hesc - SIX5 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/SIX5_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SIX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','89.65','88.93','87.6','85.78','46.04','83.63','83.42','83.55','76.99','75.61','81.3','70.04','76.32','42.49','28.67'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/SIX5.eps')

# H1hesc - BRCA1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/BRCA1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/BRCA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / BRCA1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','99.76','99.82','99.79','99.78','99.79','99.69','99.57','99.6','99.66','99.15','98.97','96.52','98.3','0.99','34.54'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/BRCA1.eps')

# H1hesc - TCF12 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/TCF12_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/TCF12.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / TCF12',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','80.57','79.22','76.71','72.53','69.18','69.01','68.74','68.97','56.35','54.55','65.64','58.89','59.03','7.83','12.89'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/TCF12.eps')

# H1hesc - EGR1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/EGR1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/EGR1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / EGR1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','83.67','82.26','77.05','73.34','69.15','68.38','66.68','67.8','67.68','64.97','66.29','55.99','56.19','22.53','10.93'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/EGR1.eps')

# H1hesc - RAD21 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/RAD21_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / RAD21',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','58.26','57.61','55.57','50.02','46.02','46.12','45.31','45.79','59.55','57.09','46.82','43.38','36.08','46.16','24.45'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/RAD21.eps')

# H1hesc - ATF3 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/ATF3_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/ATF3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / ATF3',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','96.0','95.71','95.19','94.26','93.12','92.98','92.85','92.84','93.08','91.49','91.53','86.66','87.46','32.58','42.92'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/ATF3.eps')

# H1hesc - NRF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/NRF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / NRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','60.52','60.26','47.05','56.16','52.31','57.56','54.73','55.54','44.54','42.8','49.11','41.89','41.44','45.88','38.14'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/NRF1.eps')

# H1hesc - USF2 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/USF2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / USF2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','88.87','88.15','86.99','85.08','82.93','82.79','82.65','82.67','83.0','81.06','80.31','78.75','75.33','28.47','35.06'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/USF2.eps')

# H1hesc - JUND 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/JUND_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','81.26','79.92','77.32','74.53','71.4','71.23','71.07','71.1','67.34','65.49','67.58','65.05','61.2','15.65','21.11'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/JUND.eps')

# H1hesc - ZNF143 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/ZNF143_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/ZNF143.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / ZNF143',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','89.53','88.86','87.63','85.89','48.02','83.8','83.61','83.71','78.0','76.44','82.14','64.41','76.2','28.99','21.19'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/ZNF143.eps')

# H1hesc - BACH1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/BACH1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/BACH1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / BACH1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','57.17','55.7','53.05','50.49','56.11','47.58','47.46','47.5','47.24','45.67','46.32','34.96','39.69','12.44','16.48'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/BACH1.eps')

# H1hesc - SP4 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/SP4_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/SP4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SP4',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','85.12','84.72','79.09','81.96','75.13','77.99','74.93','73.11','48.49','46.23','71.88','62.03','63.84','16.25','22.37'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/SP4.eps')

# H1hesc - MAFK 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/MAFK_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / MAFK',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','37.53','36.35','34.33','32.44','30.42','30.36','30.3','30.3','26.79','26.02','28.89','41.7','25.0','33.95','12.7'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/MAFK.eps')

# H1hesc - MYC 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/MYC_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / Myc',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','95.44','95.18','94.82','93.73','92.49','92.31','92.11','92.17','87.92','86.46','89.11','84.68','86.2','9.35','27.75'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/MYC.eps')

# H1hesc - SRF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/SRF_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/SRF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SRF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','61.09','59.74','57.38','54.94','52.28','52.17','52.08','52.08','48.87','47.61','49.98','70.59','44.55','39.19','20.61'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/SRF.eps')

# H1hesc - JUN 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/JUN_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','96.4','96.03','95.23','94.23','93.09','92.94','92.79','92.81','92.49','90.83','89.91','82.67','87.33','13.51','44.43'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/JUN.eps')

# H1hesc - TBP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/TBP_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / TBP',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','96.68','96.59','96.34','96.05','95.74','95.66','95.6','95.61','95.61','94.86','93.41','92.43','93.4','21.97','48.91'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/TBP.eps')

# H1hesc - P300 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/P300_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/P300.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / p300',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','93.74','93.24','92.27','90.33','88.37','88.16','87.85','87.9','84.65','82.34','83.63','73.5','79.52','5.19','12.68'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/P300.eps')

# H1hesc - FOSL1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/FOSL1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/FOSL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / FOSL1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','99.14','99.12','99.17','98.43','96.93','96.8','96.66','96.71','95.7','94.24','95.71','85.98','90.95','15.12','22.09'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/FOSL1.eps')

# H1hesc - USF1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/USF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/USF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / USF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','58.56','57.24','55.25','52.47','49.52','49.43','49.35','49.35','49.93','48.36','48.19','50.56','41.89','25.42','22.77'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/USF1.eps')

# H1hesc - RXRA 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/RXRA_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/RXRA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / RXRA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','92.12','91.45','90.35','88.45','85.95','85.74','85.63','85.83','78.51','76.22','83.39','65.99','75.97','11.89','24.45'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/RXRA.eps')

# H1hesc - CTCF 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/CTCF_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / CTCF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','65.09','64.39','62.52','57.09','52.93','52.98','52.18','52.69','65.75','62.97','52.71','50.18','42.16','46.74','25.07'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/CTCF.eps')

# H1hesc - RFX5 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/RFX5_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/RFX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / RFX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','67.84','66.99','65.84','63.15','60.64','60.5','60.39','60.4','51.38','49.95','57.72','59.99','52.91','36.24','21.61'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/RFX5.eps')

# H1hesc - MAX 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/MAX_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / MAX',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','85.11','84.46','83.71','81.52','79.06','78.89','78.72','78.75','72.86','71.23','75.52','65.05','70.06','9.35','24.43'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/MAX.eps')

# H1hesc - SP1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/SP1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/SP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SP1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','88.44','87.61','84.9','81.26','78.11','77.82','76.9','77.5','75.47','72.75','75.06','65.95','66.43','21.57','13.38'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/SP1.eps')

# H1hesc - CEBPB 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/CEBPB_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / CEBPB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','53.03','51.9','49.96','48.04','45.97','45.87','45.8','45.8','40.91','39.99','44.81','42.01','39.84','16.16','19.07'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/CEBPB.eps')

# H1hesc - GABP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/GABP_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.515,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','96.91','96.76','95.65','92.81','89.99','89.71','88.55','89.41','89.66','86.37','87.06','76.64','77.54','32.12','22.85'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.49,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.488,'AUC(%)',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/H1hesc/GABP.eps')

