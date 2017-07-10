library(plotrix)
cols=rep(rainbow(5),2)
styVec=c(2,2,2,2,2,1,1,1,1,1)
pointStyVec=c(22,22,22,22,22,21,21,21,21,21)
segLen=c(rep(2,10),rep(.3,10))
legendStyVec=c(styVec,rep(1,10))
legendCol=c(cols,rep('white',10))
mar.default <- c(5,5,4,2)
source('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/LabelTranslation/myLegend1.R')

# H1hesc - fdr_4 - YY1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/YY1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','81.55','88.11','88.06','69.81','86.88','85.82','85.27','84.27','90.84','91.28'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9591,0.7585,0.9599,0.7548,0.9927,0.4892,0.998,0.3021,0.9989,0.2359,0.9319,0.4154,0.8634,0.5088,0.9942,0.4867,0.991,0.5131)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/YY1.eps')

# H1hesc - fdr_4 - SP2 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/SP2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/SP2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SP2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','69.66','84.6','84.63','49.08','91.47','89.61','87.21','84.89','93.42','93.11'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.7053,0.972,0.7071,0.9709,0.9667,0.3778,0.9388,0.8229,0.946,0.7814,0.8329,0.8251,0.7072,0.8767,0.8808,0.9591,0.8591,0.9692)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/SP2.eps')

# H1hesc - fdr_4 - REST 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/REST_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/REST.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / REST',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','85.63','87.81','87.79','34.03','86.49','86.33','81.49','77.75','86.86','86.46'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8514,0.8099,0.8528,0.8078,0.9712,0.1242,0.9828,0.1528,0.9878,0.1236,0.902,0.1749,0.791,0.3305,0.9547,0.2842,0.9404,0.31)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/REST.eps')

# H1hesc - fdr_4 - SIX5 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/SIX5_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SIX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','80.19','86.69','86.59','74.3','91.31','90.76','87.57','86.96','95.07','95.33'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8872,0.7926,0.8891,0.7885,0.9795,0.6034,0.9906,0.5023,0.9932,0.4609,0.9176,0.493,0.8293,0.6418,0.9749,0.7268,0.9664,0.7584)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/SIX5.eps')

# H1hesc - fdr_4 - BRCA1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/BRCA1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/BRCA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / BRCA1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','52.64','95.77','95.69','99.67','75.69','79.65','82.81','83.92','86.62','89.82'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9704,0.9333,0.9687,0.9333,0.9854,0.9333,0.9975,0.4,0.9985,0.4667,0.9379,0.6667,0.8785,0.8,0.9911,0.6667,0.9866,0.8)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/BRCA1.eps')

# H1hesc - fdr_4 - TCF12 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/TCF12_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/TCF12.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / TCF12',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','51.11','78.47','78.43','80.42','64.55','66.19','67.67','69.17','78.09','79.94'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8272,0.8244,0.8304,0.8204,0.9579,0.4563,0.9879,0.2867,0.9902,0.3103,0.9074,0.4387,0.8042,0.5726,0.9587,0.6022,0.9437,0.6544)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/TCF12.eps')

# H1hesc - fdr_4 - EGR1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/EGR1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/EGR1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / EGR1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','71.93','81.86','81.89','87.0','85.12','85.0','84.54','82.6','89.53','89.58'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.6592,0.9523,0.6605,0.9518,0.9086,0.6901,0.9393,0.5861,0.9492,0.5716,0.8211,0.7333,0.6877,0.8166,0.8715,0.8357,0.8456,0.8676)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/EGR1.eps')

# H1hesc - fdr_4 - RAD21 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/RAD21_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / RAD21',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','82.81','87.87','87.84','90.13','85.04','85.15','78.52','76.24','87.76','87.6'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8337,0.8582,0.8355,0.8555,0.9819,0.3293,0.9722,0.2412,0.978,0.2155,0.8911,0.1892,0.7708,0.3861,0.9338,0.5024,0.9167,0.5375)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/RAD21.eps')

# H1hesc - fdr_4 - ATF3 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/ATF3_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/ATF3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / ATF3',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','82.35','92.64','92.66','91.31','91.56','90.71','87.6','87.36','95.02','95.58'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9196,0.9128,0.9208,0.911,0.9895,0.7141,0.9948,0.5761,0.996,0.5198,0.9227,0.5386,0.8382,0.6676,0.9836,0.7743,0.977,0.8107)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/ATF3.eps')

# H1hesc - fdr_4 - NRF1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/NRF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / NRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','83.24','88.95','88.99','61.37','91.03','90.98','87.2','85.84','92.45','92.2'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.5341,0.9933,0.5356,0.9931,0.9256,0.4453,0.8694,0.7745,0.8829,0.7374,0.6964,0.7914,0.5644,0.8496,0.7406,0.9108,0.7011,0.9276)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/NRF1.eps')

# H1hesc - fdr_4 - USF2 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/USF2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / USF2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','76.9','90.04','90.03','76.48','86.49','85.7','82.79','82.06','90.85','91.56'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9175,0.8734,0.9191,0.8706,0.9911,0.5064,0.996,0.3856,0.9971,0.3508,0.9226,0.398,0.8373,0.5366,0.9852,0.5785,0.9787,0.6213)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/USF2.eps')

# H1hesc - fdr_4 - JUND 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/JUND_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','70.31','84.26','84.24','73.97','74.68','74.81','75.79','76.36','81.26','82.27'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8814,0.7865,0.8839,0.7819,0.9863,0.3388,0.9967,0.1407,0.9979,0.1378,0.934,0.2945,0.8683,0.4207,0.988,0.353,0.9829,0.3958)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/JUND.eps')

# H1hesc - fdr_4 - ZNF143 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/ZNF143_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/ZNF143.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / ZNF143',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','70.16','80.55','80.52','79.04','84.76','84.05','78.37','80.73','91.62','92.23'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8885,0.7212,0.8903,0.7179,0.9808,0.5963,0.9916,0.4691,0.9941,0.4363,0.9183,0.4251,0.8302,0.6047,0.9764,0.7088,0.968,0.7417)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/ZNF143.eps')

# H1hesc - fdr_4 - BACH1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/BACH1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/BACH1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / BACH1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','64.63','69.86','69.8','61.38','67.85','67.36','67.91','68.24','71.87','72.67'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9068,0.4189,0.9088,0.4131,0.9861,0.2185,0.9961,0.0866,0.9974,0.0758,0.9323,0.2089,0.8627,0.3155,0.9877,0.2048,0.9829,0.2316)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/BACH1.eps')

# H1hesc - fdr_4 - SP4 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/SP4_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/SP4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SP4',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','64.85','80.26','80.28','45.9','88.4','86.39','84.47','82.78','90.22','89.91'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.7487,0.8948,0.7508,0.8932,0.9527,0.3467,0.9101,0.8034,0.923,0.7313,0.8016,0.8109,0.674,0.8893,0.8398,0.9267,0.8139,0.9433)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/SP4.eps')

# H1hesc - fdr_4 - MAFK 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/MAFK_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / MAFK',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','76.55','79.9','79.86','37.34','77.48','77.3','76.14','74.9','78.91','79.12'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9678,0.4187,0.9687,0.4126,0.9894,0.087,0.9984,0.0304,0.9991,0.0233,0.9447,0.1244,0.8918,0.2054,0.9942,0.0842,0.991,0.0986)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/MAFK.eps')

# H1hesc - fdr_4 - MYC 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/MYC_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / Myc',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','54.59','84.57','84.68','92.38','74.65','71.73','73.72','75.78','88.15','88.83'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8161,0.9423,0.8177,0.9423,0.9805,0.7372,0.9949,0.4287,0.9965,0.3835,0.933,0.4891,0.8564,0.6151,0.9815,0.7412,0.9732,0.7631)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/MYC.eps')

# H1hesc - fdr_4 - SRF 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/SRF_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/SRF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SRF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','80.54','90.34','90.31','46.92','83.08','82.58','82.03','80.71','84.94','85.39'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9338,0.8225,0.9351,0.8196,0.9886,0.2025,0.9982,0.0988,0.999,0.0783,0.9403,0.2362,0.885,0.3271,0.9942,0.1771,0.9909,0.2073)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/SRF.eps')

# H1hesc - fdr_4 - JUN 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/JUN_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','66.84','85.47','85.38','88.46','84.22','82.83','83.84','84.51','91.3','92.43'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9387,0.8024,0.9402,0.7984,0.9897,0.6802,0.9964,0.5092,0.9975,0.4623,0.9368,0.5703,0.8739,0.6752,0.9902,0.7261,0.9859,0.7617)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/JUN.eps')

# H1hesc - fdr_4 - TBP 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/TBP_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / TBP',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','69.39','89.24','89.37','94.64','87.22','83.81','88.16','89.3','92.88','93.89'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.987,0.7748,0.9873,0.773,0.994,0.8262,0.999,0.5691,0.9995,0.4805,0.9481,0.6933,0.9032,0.7801,0.9969,0.7642,0.9947,0.7943)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/TBP.eps')

# H1hesc - fdr_4 - P300 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/P300_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/P300.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / p300',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','48.24','82.69','82.64','92.83','72.03','70.14','76.71','76.6','87.02','88.29'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9109,0.8189,0.909,0.8215,0.972,0.7192,0.988,0.5039,0.9912,0.4619,0.9076,0.6404,0.8126,0.748,0.9666,0.8005,0.9558,0.832)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/P300.eps')

# H1hesc - fdr_4 - FOSL1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/FOSL1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/FOSL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / FOSL1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','65.14','86.21','86.38','97.6','81.55','82.22','86.43','86.22','92.39','94.52'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8578,0.9134,0.8613,0.9134,0.9833,0.8189,0.995,0.4724,0.9963,0.5039,0.9308,0.6772,0.8628,0.7638,0.9838,0.8189,0.9781,0.8898)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/FOSL1.eps')

# H1hesc - fdr_4 - USF1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/USF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/USF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / USF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','78.65','85.03','84.99','49.64','81.3','81.02','78.4','76.37','83.35','83.71'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.936,0.5915,0.937,0.5865,0.9937,0.1936,0.9973,0.131,0.9982,0.1159,0.926,0.2124,0.8426,0.3279,0.989,0.2385,0.9832,0.2682)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/USF1.eps')

# H1hesc - fdr_4 - RXRA 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/RXRA_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/RXRA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / RXRA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','51.75','75.95','76.05','88.34','64.7','64.23','64.34','68.57','77.69','80.14'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9063,0.7329,0.9082,0.7329,0.977,0.5638,0.994,0.2997,0.9959,0.276,0.9196,0.3561,0.8349,0.5223,0.981,0.5638,0.9729,0.6261)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/RXRA.eps')

# H1hesc - fdr_4 - CTCF 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/CTCF_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / CTCF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','82.87','88.66','88.64','91.84','86.12','86.16','79.54','77.92','89.38','89.34'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8357,0.8816,0.8375,0.8792,0.9847,0.3605,0.9751,0.2733,0.9807,0.2463,0.8935,0.216,0.7744,0.4268,0.9382,0.5517,0.9213,0.5894)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/CTCF.eps')

# H1hesc - fdr_4 - RFX5 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/RFX5_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/RFX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / RFX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','80.77','88.13','88.2','64.06','86.7','86.21','82.75','81.1','90.29','90.1'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8655,0.8513,0.8671,0.8513,0.9687,0.4077,0.9866,0.3449,0.9896,0.3103,0.9115,0.3564,0.8159,0.4872,0.9626,0.5359,0.9506,0.5667)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/RFX5.eps')

# H1hesc - fdr_4 - POU5F1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/POU5F1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/POU5F1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / POU5F1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','80.73','84.6','84.57','78.92','82.43','82.14','86.73','86.83','87.02','87.82'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.97,0.6908,0.9706,0.6884,0.9924,0.378,0.999,0.1127,0.9995,0.0816,0.9464,0.4502,0.9015,0.5522,0.9962,0.3637,0.9939,0.4066)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/POU5F1.eps')

# H1hesc - fdr_4 - MAX 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/MAX_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / MAX',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','51.59','76.92','76.82','80.85','62.44','60.67','63.01','66.7','73.26','75.66'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8696,0.7555,0.8712,0.7524,0.9871,0.4311,0.9973,0.1979,0.9983,0.1672,0.9415,0.2771,0.8786,0.4236,0.9898,0.4356,0.9846,0.4885)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/MAX.eps')

# H1hesc - fdr_4 - SP1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/SP1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/SP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SP1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','67.41','83.53','83.61','89.66','85.06','85.64','84.47','83.47','90.88','91.03'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.7188,0.9577,0.7213,0.9569,0.9319,0.7414,0.9551,0.6326,0.9589,0.6445,0.8524,0.7372,0.7359,0.8289,0.904,0.8625,0.8847,0.8885)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/SP1.eps')

# H1hesc - fdr_4 - CEBPB 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/CEBPB_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / CEBPB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','63.52','67.9','67.84','55.14','65.7','64.97','66.31','66.27','68.58','69.31'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9724,0.3039,0.9732,0.3007,0.9925,0.1618,0.9996,0.0447,0.9998,0.0298,0.9359,0.1652,0.876,0.2466,0.9982,0.1093,0.9965,0.129)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/CEBPB.eps')

# H1hesc - fdr_4 - GABP 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/GABP_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','74.33','90.57','90.55','96.57','90.38','90.12','89.37','87.7','95.73','95.65'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8457,0.9573,0.8477,0.9546,0.9291,0.9156,0.9625,0.7308,0.9699,0.6841,0.8721,0.7957,0.76,0.8493,0.9221,0.945,0.904,0.9576)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/H1hesc/fdr_4/GABP.eps')

