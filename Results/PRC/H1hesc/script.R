library(plotrix)
cols=rainbow(15)
styVec=c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
segLen=c(rep(2,15),rep(.3,15))
legendStyVec=c(styVec,rep(1,15))
legendCol=c(cols,rep('white',15))
mar.default <- c(5,5,4,2)
source('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend7.R')

# H1hesc - SRF 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/SRF_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/SRF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / SRF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','32.96','32.38','30.95','26.93','25.5','23.08','21.85','22.13','19.22','13.36','12.59','18.06','9.46','7.78','1.55'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/SRF.eps')

# H1hesc - YY1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/YY1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','67.08','66.69','65.45','61.5','60.01','57.66','55.65','56.07','55.67','44.45','39.74','37.7','34.39','7.91','3.47'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/YY1.eps')

# H1hesc - SP2 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/SP2_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/SP2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / SP2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','28.13','27.91','27.45','19.82','20.97','14.8','15.84','14.4','8.5','4.3','6.85','4.15','4.17','0.41','0.19'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/SP2.eps')

# H1hesc - MAFK 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/MAFK_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / MAFK',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','15.52','15.06','14.21','11.59','10.64','9.39','8.64','8.8','6.8','4.42','4.67','6.65','3.34','7.2','1.27'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/MAFK.eps')

# H1hesc - TBP 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/TBP_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / TBP',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','75.58','76.27','71.79','69.13','70.7','66.98','64.47','66.17','63.73','48.39','45.88','35.47','35.83','0.44','4.63'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/TBP.eps')

# H1hesc - CTCF 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/CTCF_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / CTCF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','71.13','70.65','68.33','62.8','58.83','59.03','56.96','57.89','67.68','60.84','55.54','47.93','42.76','49.39','24.75'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/CTCF.eps')

# H1hesc - RAD21 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/RAD21_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / RAD21',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','64.67','64.19','61.52','55.69','51.66','52.04','49.83','50.65','60.61','54.14','48.9','42.15','36.69','48.62','24.01'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/RAD21.eps')

# H1hesc - MAX 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/MAX_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / MAX',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','56.68','56.43','55.32','47.95','46.02','42.14','39.43','39.95','35.4','23.72','22.14','15.87','16.39','0.52','1.85'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/MAX.eps')

# H1hesc - SIX5 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/SIX5_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / SIX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','52.37','52.18','50.58','43.45','21.67','40.11','35.94','36.95','33.99','20.79','19.96','12.33','13.28','4.38','1.33'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/SIX5.eps')

# H1hesc - NRF1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/NRF1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / NRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','59.93','59.69','49.31','54.61','50.64','55.25','50.88','52.05','45.73','39.61','42.36','33.42','33.02','36.06','33.03'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/NRF1.eps')

# H1hesc - JUN 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/JUN_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','60.4','60.61','55.84','52.54','52.68','48.49','43.54','44.32','43.21','26.38','22.8','17.55','16.25','0.24','5.1'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/JUN.eps')

# H1hesc - ATF3 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/ATF3_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/ATF3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / ATF3',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','80.0','79.92','77.78','73.73','73.16','70.22','67.75','67.5','68.06','53.65','53.57','41.19','40.6','2.81','7.66'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/ATF3.eps')

# H1hesc - ZNF143 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/ZNF143_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/ZNF143.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / ZNF143',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','65.12','64.86','63.39','56.11','30.55','51.76','48.72','49.93','45.74','32.05','34.42','19.8','22.51','3.85','1.53'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/ZNF143.eps')

# H1hesc - BRCA1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/BRCA1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/BRCA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / BRCA1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','15.74','19.33','14.99','13.04','15.04','9.33','9.98','9.75','10.14','4.4','4.4','2.24','2.27','0.01','0.03'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/BRCA1.eps')

# H1hesc - USF2 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/USF2_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / USF2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','74.14','73.59','71.66','67.37','65.37','63.16','61.58','61.92','61.57','50.57','45.74','41.91','40.1','3.82','8.04'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/USF2.eps')

# H1hesc - CEBPB 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/CEBPB_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / CEBPB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','33.79','33.22','32.05','28.76','27.25','25.42','24.17','24.34','19.16','14.26','16.19','12.35','11.3','2.35','2.59'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/CEBPB.eps')

# H1hesc - RFX5 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/RFX5_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/RFX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / RFX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','20.74','20.67','18.98','14.73','15.67','12.73','11.94','11.68','11.04','5.53','5.09','3.5','3.26','1.44','0.4'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/RFX5.eps')

# H1hesc - BACH1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/BACH1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/BACH1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / BACH1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','26.36','25.9','24.34','20.1','24.95','16.61','15.06','15.42','14.81','8.96','8.92','6.2','5.53','1.09','1.21'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/BACH1.eps')

# H1hesc - SP4 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/SP4_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/SP4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / SP4',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','38.54','38.27','35.71','31.29','28.94','25.62','23.35','22.76','13.7','8.05','13.39','7.79','8.14','1.16','1.6'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/SP4.eps')

# H1hesc - GABP 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/GABP_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','82.34','82.17','80.0','72.63','68.71','65.98','63.46','64.07','65.01','49.56','50.53','34.16','34.26','7.77','5.08'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/GABP.eps')

# H1hesc - EGR1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/EGR1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/EGR1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / EGR1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','46.04','44.8','41.09','33.45','30.47','25.22','25.49','26.48','26.09','16.35','15.48','10.06','10.15','2.66','1.06'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/EGR1.eps')

# H1hesc - USF1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/USF1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/USF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / USF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','54.86','53.7','51.69','48.1','45.22','44.47','43.9','43.96','44.36','39.16','37.87','37.3','31.92','13.35','11.53'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/USF1.eps')

# H1hesc - FOSL1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/FOSL1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/FOSL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / FOSL1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','26.94','26.39','26.34','17.89','21.11','14.3','14.77','17.39','15.0','6.99','6.2','3.63','3.71','0.04','0.06'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/FOSL1.eps')

# H1hesc - JUND 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/JUND_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','60.34','59.37','56.91','50.22','46.98','44.22','42.27','43.01','39.14','29.0','26.71','24.64','20.95','2.43','2.92'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/JUND.eps')

# H1hesc - MYC 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/MYC_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / Myc',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','69.73','69.67','68.08','60.56','60.8','54.73','50.53','51.91','51.28','34.38','29.55','22.53','22.76','0.3','1.06'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/MYC.eps')

# H1hesc - RXRA 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/RXRA_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/RXRA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / RXRA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','13.07','13.14','12.68','9.31','8.89','6.66','5.97','6.43','5.89','2.57','2.53','1.24','1.39','0.04','0.11'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/RXRA.eps')

# H1hesc - TCF12 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/TCF12_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/TCF12.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / TCF12',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','36.87','36.79','35.19','26.01','25.53','21.02','20.04','20.89','19.47','10.62','9.88','6.47','6.58','0.32','0.4'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/TCF12.eps')

# H1hesc - P300 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/P300_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/P300.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / p300',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','29.0','29.11','29.36','21.19','22.33','17.67','16.58','16.6','16.38','7.85','6.66','3.85','4.23','0.06','0.1'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/P300.eps')

# H1hesc - SP1 
prc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/H1hesc/SP1_prc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/SP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
prc=prc[,c(41,42,43,44,45,46,25,26,53,54,61,62,47,48,3,4,67,68,9,10,33,34,19,20,59,60,1,2,39,40)]
plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='H1-hESC / SP1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))
myLegend(0.7,1.06,c('HINT-BC','HINT-BCN','HINT','DNase2TF','PIQ','Wellington','Neph','Boyle','BinDNase','Centipede','FLR','Cuellar','TC','PWM','FS','45.74','44.81','41.77','36.19','33.73','28.62','27.54','26.88','27.76','15.9','15.01','9.23','9.35','1.35','0.68'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.853,y=1.03,'METHOD',font=2,cex=0.8)
text(x=0.988,y=1.03,'AUPR',font=2,cex=0.8)
for (i in (1:15)*2){
  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/PRC/H1hesc/SP1.eps')

