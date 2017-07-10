library(plotrix)
cols=rep(rainbow(5),2)
styVec=c(2,2,2,2,2,1,1,1,1,1)
pointStyVec=c(22,22,22,22,22,21,21,21,21,21)
segLen=c(rep(2,10),rep(.3,10))
legendStyVec=c(styVec,rep(1,10))
legendCol=c(cols,rep('white',10))
mar.default <- c(5,5,4,2)
source('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/LabelTranslation/myLegend1.R')

# K562 - fdr_4 - YY1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/YY1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','86.49','93.2','93.23','92.53','92.26','89.48','91.82','92.36','97.31','97.59'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.946,0.9091,0.9457,0.9094,0.9889,0.8489,0.9981,0.4702,0.9996,0.2192,0.841,0.7569,0.8144,0.8092,0.9834,0.8185,0.982,0.8457)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/YY1.eps')

# K562 - fdr_4 - ZBTB7A 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/ZBTB7A_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ZBTB7A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ZBTB7A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','43.68','70.67','70.67','97.61','59.74','52.26','73.85','75.66','84.82','86.55'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8185,0.7226,0.8178,0.7215,0.8389,0.9757,0.997,0.2859,0.9984,0.1656,0.8291,0.6878,0.7925,0.7584,0.9526,0.7679,0.9493,0.807)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ZBTB7A.eps')

# K562 - fdr_4 - SP2 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/SP2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/SP2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SP2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','71.31','85.9','85.81','97.35','93.53','86.22','88.38','88.05','92.45','92.59'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.7324,0.9782,0.7308,0.9771,0.9161,0.9914,0.9793,0.8363,0.9786,0.5984,0.7572,0.902,0.7109,0.9292,0.8806,0.9149,0.8752,0.923)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/SP2.eps')

# K562 - fdr_4 - PU1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/PU1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/PU1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / PU.1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','78.53','84.15','84.16','66.72','84.7','80.62','79.86','81.0','87.52','88.23'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.883,0.7382,0.8825,0.7374,0.9755,0.4097,0.9963,0.2013,0.999,0.0585,0.8636,0.3768,0.8315,0.4708,0.9689,0.385,0.9664,0.4162)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/PU1.eps')

# K562 - fdr_4 - NFYA 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/NFYA_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/NFYA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NF-YA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','72.65','85.01','85.01','89.41','90.36','84.24','85.94','86.8','95.16','95.66'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8397,0.9118,0.8396,0.9118,0.9557,0.8268,0.9927,0.5092,0.9963,0.3391,0.8702,0.6291,0.8401,0.6955,0.9572,0.7453,0.9542,0.764)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/NFYA.eps')

# K562 - fdr_4 - FOS 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/FOS_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/FOS.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / C-fos',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','63.95','85.0','85.0','98.19','86.5','70.62','77.24','80.36','92.35','94.09'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8479,0.8962,0.8474,0.8967,0.9285,0.9798,0.9897,0.6438,0.9979,0.1834,0.8687,0.541,0.8376,0.6569,0.9529,0.8429,0.9486,0.8962)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/FOS.eps')

# K562 - fdr_4 - TAL1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/TAL1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/TAL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / TAL1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','73.13','78.32','78.3','93.82','85.73','74.57','78.41','81.09','89.33','90.94'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9398,0.5982,0.9395,0.5961,0.9749,0.7749,0.9953,0.5023,0.9995,0.0698,0.8826,0.4381,0.856,0.5577,0.9736,0.6575,0.9713,0.7206)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/TAL1.eps')

# K562 - fdr_4 - REST 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/REST_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/REST.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / REST',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','84.43','87.79','87.75','54.93','88.41','86.62','79.21','78.92','89.51','89.74'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8452,0.8331,0.8447,0.8325,0.9588,0.3595,0.9943,0.1215,0.9967,0.1219,0.8245,0.2826,0.7779,0.3793,0.9426,0.4114,0.939,0.4325)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/REST.eps')

# K562 - fdr_4 - ELK1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/ELK1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ELK1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ELK1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','61.88','88.53','88.48','97.46','92.46','80.58','87.92','87.33','94.32','94.6'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8243,0.983,0.8225,0.9837,0.9213,0.995,0.9734,0.8128,0.984,0.5188,0.7832,0.9146,0.7359,0.9447,0.899,0.9617,0.893,0.9749)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ELK1.eps')

# K562 - fdr_4 - ARID3A 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/ARID3A_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ARID3A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ARID3A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','47.05','47.05','47.05','98.61','68.63','49.85','65.42','70.05','79.0','82.3'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,1.0,0.0,1.0,0.0,0.9792,0.9308,0.9992,0.3908,1.0,0.0508,0.8883,0.4523,0.8708,0.5585,0.9923,0.6062,0.9916,0.6738)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ARID3A.eps')

# K562 - fdr_4 - STAT5A 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/STAT5A_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/STAT5A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / STAT5A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','69.5','86.19','86.14','93.15','89.03','71.85','79.04','81.16','87.97','90.17'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8787,0.8395,0.8782,0.8374,0.9727,0.8088,0.9955,0.6171,0.9993,0.0718,0.8746,0.5093,0.8463,0.6065,0.9742,0.6049,0.9717,0.6811)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/STAT5A.eps')

# K562 - fdr_4 - SIX5 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/SIX5_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SIX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','79.91','87.11','87.16','93.9','93.15','88.33','88.03','88.35','96.34','96.59'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8714,0.8287,0.8711,0.8287,0.976,0.8644,0.9954,0.6138,0.9978,0.3761,0.8367,0.6708,0.8001,0.7366,0.9612,0.8421,0.9586,0.8644)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/SIX5.eps')

# K562 - fdr_4 - NFYB 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/NFYB_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/NFYB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NF-YB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','79.76','87.51','87.52','70.4','87.55','85.26','83.52','83.73','91.64','92.01'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.876,0.8374,0.8753,0.8376,0.9693,0.5557,0.9949,0.2281,0.9976,0.1638,0.8827,0.3952,0.8562,0.4558,0.9666,0.4675,0.9642,0.4868)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/NFYB.eps')

# K562 - fdr_4 - GATA1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/GATA1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/GATA1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / GATA1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','62.35','84.7','84.87','98.76','94.12','73.97','79.77','83.32','93.86','96.05'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9136,0.8614,0.9131,0.8641,0.9642,0.9843,0.9907,0.8558,0.9992,0.3073,0.8859,0.6095,0.8611,0.7255,0.9704,0.8729,0.9669,0.9339)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/GATA1.eps')

# K562 - fdr_4 - EJUNB 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/EJUNB_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/EJUNB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / eGFP-JunB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','68.68','82.91','82.89','95.55','82.46','71.91','74.89','77.69','88.72','90.25'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8332,0.8315,0.8324,0.8318,0.9422,0.8624,0.9919,0.448,0.9982,0.1088,0.866,0.4219,0.8348,0.5406,0.9564,0.7023,0.9524,0.7553)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/EJUNB.eps')

# K562 - fdr_4 - SMC3 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/SMC3_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/SMC3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SMC3',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','87.03','91.87','91.85','98.05','91.91','90.52','80.37','82.74','96.36','96.84'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8207,0.953,0.8196,0.9527,0.9352,0.9791,0.9894,0.4498,0.9919,0.3167,0.8162,0.3435,0.7661,0.5268,0.9142,0.8786,0.9095,0.9151)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/SMC3.eps')

# K562 - fdr_4 - EGR1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/EGR1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/EGR1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / EGR1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','69.35','77.32','77.3','87.02','77.28','75.4','78.3','79.19','86.18','86.84'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.6875,0.8523,0.6864,0.8522,0.92,0.7294,0.9886,0.2732,0.9864,0.2428,0.7642,0.6535,0.714,0.7259,0.8838,0.7359,0.8778,0.7625)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/EGR1.eps')

# K562 - fdr_4 - RAD21 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/RAD21_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/RAD21.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / RAD21',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','87.75','91.92','91.91','97.39','92.67','90.42','79.79','82.09','95.63','96.08'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8133,0.9529,0.8122,0.9527,0.9261,0.9687,0.9864,0.4887,0.9879,0.2846,0.8138,0.3143,0.7625,0.5027,0.9059,0.8568,0.9008,0.8929)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/RAD21.eps')

# K562 - fdr_4 - ATF3 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/ATF3_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ATF3.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ATF3',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','93.55','96.42','96.37','99.55','98.14','97.13','95.97','96.38','98.4','98.4'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8535,0.9856,0.8515,0.9819,0.9861,0.9928,0.9947,0.8989,0.9972,0.6715,0.7954,0.9061,0.7647,0.935,0.9785,0.9495,0.9766,0.9567)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ATF3.eps')

# K562 - fdr_4 - ETS1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/ETS1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ETS1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ETS1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','59.21','86.86','86.72','99.12','87.47','77.76','86.74','87.39','95.07','95.94'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8845,0.9262,0.8835,0.9227,0.9506,0.9969,0.9918,0.7167,0.9972,0.4563,0.8526,0.8266,0.8144,0.8815,0.9526,0.9303,0.9488,0.9537)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ETS1.eps')

# K562 - fdr_4 - GATA2 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/GATA2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/GATA2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / GATA2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','56.58','76.93','77.01','97.76','87.47','64.47','72.94','77.56','89.41','92.34'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9425,0.6751,0.9418,0.6764,0.9661,0.9325,0.9925,0.7178,0.9995,0.1752,0.8872,0.4988,0.8627,0.6313,0.9728,0.7689,0.9697,0.8441)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/GATA2.eps')

# K562 - fdr_4 - NRF1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/NRF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/NRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','82.36','88.94','88.92','84.8','94.4','91.38','90.09','90.03','93.16','93.55'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.6185,0.9927,0.6175,0.9925,0.9179,0.8266,0.9611,0.8127,0.9441,0.6986,0.6632,0.9042,0.6098,0.9332,0.7653,0.9269,0.7555,0.9474)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/NRF1.eps')

# K562 - fdr_4 - EJUND 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/EJUND_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/EJUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / eGFP-JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','67.78','81.5','81.53','95.24','81.55','71.48','73.9','77.02','88.53','90.22'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8304,0.8176,0.8299,0.8189,0.9388,0.8592,0.9945,0.4102,0.9988,0.1045,0.8683,0.3999,0.8375,0.522,0.96,0.6745,0.9564,0.7286)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/EJUND.eps')

# K562 - fdr_4 - USF2 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/USF2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/USF2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / USF2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','78.59','91.41','91.37','97.45','90.3','86.41','83.5','85.03','95.0','96.01'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9024,0.9417,0.9016,0.9411,0.9822,0.9135,0.9967,0.522,0.9986,0.3451,0.8322,0.5587,0.7975,0.6569,0.9689,0.8088,0.9664,0.8566)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/USF2.eps')

# K562 - fdr_4 - EFOS 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/EFOS_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/EFOS.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / eGFP-FOS',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','65.07','83.41','83.43','96.09','80.09','69.41','73.39','76.55','88.19','89.88'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8564,0.8403,0.8559,0.8411,0.9383,0.8858,0.993,0.4198,0.9988,0.1151,0.8714,0.4209,0.8414,0.5388,0.96,0.6933,0.9563,0.7439)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/EFOS.eps')

# K562 - fdr_4 - E2F6 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/E2F6_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/E2F6.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / E2F6',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','64.98','83.51','83.52','96.69','77.37','73.6','82.14','82.73','90.45','91.48'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8154,0.9056,0.8133,0.9076,0.9545,0.9152,0.9927,0.3666,0.9943,0.2692,0.8083,0.728,0.7652,0.7881,0.9348,0.8168,0.9308,0.8491)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/E2F6.eps')

# K562 - fdr_4 - JUND 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/JUND_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/JUND.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / JunD',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','66.84','80.25','80.26','94.76','78.54','69.99','73.64','76.72','86.89','88.62'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8365,0.7808,0.836,0.7815,0.9467,0.8231,0.9963,0.3398,0.9993,0.0865,0.8716,0.4036,0.8418,0.5223,0.9655,0.6293,0.9625,0.6819)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/JUND.eps')

# K562 - fdr_4 - ELF1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/ELF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ELF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ELF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','68.21','82.85','82.79','89.8','82.46','75.59','80.03','81.99','90.81','92.11'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9093,0.768,0.9086,0.7662,0.978,0.7468,0.9954,0.4009,0.9991,0.1874,0.8639,0.5386,0.8298,0.6394,0.965,0.6884,0.962,0.7332)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ELF1.eps')

# K562 - fdr_4 - ZNF143 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/ZNF143_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ZNF143.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ZNF143',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','70.65','79.99','79.99','91.71','84.32','80.02','78.67','81.4','93.13','93.8'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8725,0.7248,0.8722,0.7236,0.9777,0.8154,0.9961,0.479,0.9983,0.3058,0.8373,0.5454,0.8009,0.6566,0.9627,0.7888,0.9602,0.8169)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ZNF143.eps')

# K562 - fdr_4 - NFE2 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/NFE2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/NFE2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NF-E2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','89.54','93.07','93.08','96.67','96.57','91.48','88.06','89.01','97.32','97.86'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9142,0.9196,0.914,0.9192,0.957,0.912,0.9917,0.6383,0.9984,0.1633,0.8643,0.4638,0.8339,0.5874,0.9636,0.7427,0.9605,0.8055)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/NFE2.eps')

# K562 - fdr_4 - BACH1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/BACH1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/BACH1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / BACH1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','65.37','73.59','73.61','89.64','81.34','69.69','71.93','74.63','84.26','85.88'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8728,0.5874,0.8726,0.587,0.9306,0.7765,0.9887,0.4737,0.9975,0.1318,0.8556,0.407,0.8227,0.5164,0.9538,0.6034,0.9504,0.6559)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/BACH1.eps')

# K562 - fdr_4 - MAFK 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/MAFK_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/MAFK.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / MAFK',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','75.58','79.94','79.94','56.51','80.18','76.56','74.66','75.51','82.08','82.63'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9519,0.504,0.9515,0.5045,0.977,0.3114,0.9975,0.1432,0.9996,0.0254,0.8832,0.2368,0.8588,0.3078,0.9808,0.2427,0.9791,0.2673)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/MAFK.eps')

# K562 - fdr_4 - ATF1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/ATF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ATF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ATF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','71.53','78.55','78.64','93.44','84.6','76.1','79.09','81.42','89.49','90.89'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9408,0.618,0.9403,0.6203,0.97,0.7937,0.9959,0.5156,0.9976,0.2531,0.87,0.5214,0.835,0.6245,0.9629,0.7089,0.9598,0.7565)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ATF1.eps')

# K562 - fdr_4 - MYC 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/MYC_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/MYC.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / Myc',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','53.46','82.17','82.12','99.39','77.31','65.36','75.98','78.81','91.31','92.9'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.7738,0.9759,0.7727,0.9764,0.9835,0.977,0.9971,0.5147,0.9988,0.2473,0.8701,0.6042,0.8351,0.7049,0.9664,0.8428,0.9637,0.881)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/MYC.eps')

# K562 - fdr_4 - TR4 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/TR4_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/TR4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / TR4',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','84.39','91.07','91.06','93.16','93.71','90.59','91.56','91.52','94.95','95.27'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.846,0.9275,0.8452,0.9275,0.942,0.8626,0.993,0.5687,0.995,0.3397,0.8199,0.7977,0.7765,0.8511,0.9325,0.8053,0.9284,0.8359)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/TR4.eps')

# K562 - fdr_4 - SRF 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/SRF_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/SRF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SRF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','75.17','87.54','87.38','86.68','85.16','77.55','81.27','82.94','88.27','89.39'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.907,0.8269,0.9071,0.8212,0.9645,0.6867,0.9969,0.3508,0.9997,0.081,0.8763,0.4689,0.8503,0.5655,0.9796,0.5178,0.978,0.5636)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/SRF.eps')

# K562 - fdr_4 - JUN 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/JUN_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','63.97','80.07','80.02','96.04','86.59','75.13','78.79','81.37','92.27','93.51'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9124,0.7142,0.9121,0.7125,0.9841,0.8479,0.9968,0.5999,0.9992,0.2664,0.8687,0.5495,0.8402,0.6597,0.975,0.7884,0.9728,0.8306)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/JUN.eps')

# K562 - fdr_4 - TBP 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/TBP_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/TBP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / TBP',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','64.1','86.46','86.61','98.14','85.58','74.38','85.55','87.68','93.72','94.93'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9809,0.7563,0.9808,0.7583,0.9833,0.9479,0.9984,0.5979,0.9998,0.2896,0.8864,0.7375,0.8662,0.8,0.9865,0.8542,0.9855,0.8833)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/TBP.eps')

# K562 - fdr_4 - BHLHE40 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/BHLHE40_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/BHLHE40.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / BHLHE40',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','64.56','79.64','79.67','82.03','73.17','69.24','71.17','73.84','82.37','84.13'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9395,0.6535,0.9381,0.6556,0.9871,0.5186,0.998,0.2138,0.9991,0.1263,0.8497,0.4136,0.8126,0.5236,0.9717,0.509,0.9694,0.5569)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/BHLHE40.eps')

# K562 - fdr_4 - CTCFL 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/CTCFL_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/CTCFL.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / CTCFL',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','81.51','88.61','88.6','96.56','87.6','87.77','82.46','84.16','94.06','94.52'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8025,0.9373,0.8015,0.9368,0.9137,0.9582,0.9788,0.4254,0.9868,0.4301,0.8156,0.5104,0.7616,0.6506,0.8959,0.8805,0.8904,0.9131)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/CTCFL.eps')

# K562 - fdr_4 - IRF1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/IRF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/IRF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / IRF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','78.99','87.69','87.67','87.02','83.48','79.88','80.71','81.74','86.98','87.66'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.7319,0.9425,0.7319,0.9415,0.9363,0.6698,0.9986,0.1387,0.9999,0.0259,0.8441,0.4412,0.8183,0.5212,0.9904,0.3398,0.9897,0.3718)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/IRF1.eps')

# K562 - fdr_4 - FOSL1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/FOSL1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/FOSL1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / FOSL1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','67.62','83.21','83.18','96.91','82.49','72.12','76.27','79.33','90.68','92.29'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8088,0.8751,0.808,0.8749,0.9284,0.9353,0.9924,0.4519,0.9988,0.1358,0.8674,0.4718,0.835,0.5966,0.9558,0.755,0.9516,0.8072)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/FOSL1.eps')

# K562 - fdr_4 - NR2F2 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/NR2F2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/NR2F2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / NR2F2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','58.56','68.77','68.77','93.83','75.89','62.1','68.8','73.25','84.07','86.26'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9203,0.5169,0.9199,0.5151,0.9654,0.7866,0.9968,0.4062,0.9992,0.0809,0.8617,0.4092,0.8296,0.5464,0.9689,0.6449,0.9665,0.7011)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/NR2F2.eps')

# K562 - fdr_4 - USF1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/USF1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/USF1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / USF1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','79.32','87.2','87.21','85.95','83.61','81.33','77.08','79.1','89.39','90.4'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9184,0.7453,0.9177,0.7461,0.986,0.5732,0.9984,0.1945,0.9993,0.1066,0.8363,0.3574,0.8038,0.4738,0.976,0.5136,0.9741,0.5592)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/USF1.eps')

# K562 - fdr_4 - THAP1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/THAP1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/THAP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / THAP1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','54.54','87.22','86.77','97.19','80.63','70.09','83.56','84.03','88.5','89.39'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.885,0.9026,0.8842,0.8883,0.9689,0.9312,0.9947,0.5415,0.997,0.3295,0.8334,0.7908,0.7938,0.8539,0.9506,0.8052,0.9469,0.8309)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/THAP1.eps')

# K562 - fdr_4 - MAFF 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/MAFF_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/MAFF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / MAFF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','78.06','82.44','82.43','52.6','81.54','78.69','75.97','76.51','83.01','83.46'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9543,0.5274,0.9541,0.5267,0.9726,0.2659,0.9974,0.1134,0.9996,0.0176,0.8818,0.227,0.8569,0.292,0.9801,0.2045,0.9785,0.2245)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/MAFF.eps')

# K562 - fdr_4 - CTCF 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/CTCF_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/CTCF.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / CTCF',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','82.97','87.74','87.72','94.84','86.29','85.47','77.34','78.8','91.64','92.03'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8456,0.8596,0.8444,0.8592,0.958,0.792,0.9928,0.2738,0.9946,0.1967,0.8202,0.3103,0.7723,0.4544,0.9315,0.689,0.9274,0.7171)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/CTCF.eps')

# K562 - fdr_4 - MEF2A 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/MEF2A_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/MEF2A.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / MEF2A',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','73.65','81.59','81.53','94.19','85.33','75.31','79.46','81.89','88.72','90.49'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9712,0.6444,0.9712,0.6411,0.9409,0.8452,0.9991,0.3696,0.9999,0.0453,0.8631,0.4625,0.8438,0.5717,0.993,0.5186,0.9924,0.5818)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/MEF2A.eps')

# K562 - fdr_4 - RFX5 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/RFX5_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/RFX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / RFX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','76.32','85.4','85.45','82.74','87.8','81.07','82.03','82.76','90.38','91.13'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8571,0.8208,0.8563,0.8189,0.9635,0.6879,0.9936,0.4489,0.9964,0.2408,0.8371,0.5723,0.7958,0.6455,0.9481,0.6705,0.9445,0.6994)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/RFX5.eps')

# K562 - fdr_4 - STAT2 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/STAT2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/STAT2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / STAT2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','82.51','89.22','89.13','90.43','89.72','84.25','85.85','86.63','92.63','93.36'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9454,0.8177,0.9452,0.8165,0.9654,0.7698,0.9976,0.3033,0.9997,0.0816,0.8668,0.512,0.8418,0.5893,0.9849,0.5212,0.9835,0.574)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/STAT2.eps')

# K562 - fdr_4 - CCNT2 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/CCNT2_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/CCNT2.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / CCNT2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','74.14','83.55','83.5','98.31','93.83','78.0','84.59','86.63','94.41','95.97'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.933,0.7694,0.9326,0.7674,0.9651,0.9555,0.9895,0.8301,0.9989,0.1888,0.8788,0.6397,0.8509,0.7347,0.9655,0.8581,0.9624,0.9168)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/CCNT2.eps')

# K562 - fdr_4 - EGATA 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/EGATA_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/EGATA.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / eGFP-GATA2',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','57.01','72.59','72.61','95.81','83.54','62.19','70.47','75.29','84.94','88.5'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9419,0.5803,0.9412,0.5787,0.9655,0.8662,0.9919,0.6282,0.9992,0.1163,0.8869,0.4467,0.8623,0.5828,0.9722,0.6786,0.9691,0.7657)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/EGATA.eps')

# K562 - fdr_4 - ZNF263 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/ZNF263_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ZNF263.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ZNF263',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','70.48','71.57','71.57','71.42','74.9','73.39','76.22','77.08','81.6','82.32'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.1108,0.998,0.1111,0.9981,0.9756,0.3954,0.9945,0.1437,0.9958,0.0996,0.8315,0.4936,0.7932,0.5634,0.9503,0.4458,0.9475,0.4731)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ZNF263.eps')

# K562 - fdr_4 - MAX 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/MAX_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/MAX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / MAX',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','47.32','74.47','74.55','95.99','62.52','53.87','63.11','68.6','81.83','84.66'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8298,0.8011,0.829,0.804,0.9895,0.7793,0.9985,0.2823,0.9996,0.1151,0.8802,0.4048,0.8517,0.5387,0.9775,0.6728,0.9757,0.7272)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/MAX.eps')

# K562 - fdr_4 - E2F4 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/E2F4_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/E2F4.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / E2F4',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','57.1','79.51','79.38','52.76','82.93','75.52','82.37','81.4','88.69','89.12'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.6973,0.9719,0.6961,0.9699,0.9432,0.4825,0.9759,0.6112,0.9716,0.4614,0.7015,0.8935,0.6489,0.9239,0.8243,0.9126,0.8164,0.9281)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/E2F4.eps')

# K562 - fdr_4 - STAT1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/STAT1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/STAT1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / STAT1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','63.67','75.97','75.97','95.73','88.13','69.69','78.96','80.82','87.75','89.4'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8376,0.7458,0.8375,0.7458,0.9699,0.8746,0.9942,0.7153,0.9991,0.1831,0.8786,0.5932,0.852,0.661,0.9754,0.7119,0.9729,0.7763)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/STAT1.eps')

# K562 - fdr_4 - SP1 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/SP1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/SP1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SP1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','70.24','85.54','85.42','97.33','93.46','84.58','88.93','88.61','92.25','92.63'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.735,0.9828,0.7333,0.9813,0.9244,0.9848,0.9838,0.822,0.9838,0.5539,0.7755,0.9004,0.7318,0.9282,0.8971,0.8909,0.8918,0.9059)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/SP1.eps')

# K562 - fdr_4 - ZBTB33 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/ZBTB33_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ZBTB33.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / ZBTB33',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','72.03','81.81','81.72','75.87','84.01','78.87','81.96','82.43','84.41','84.95'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.7731,0.8289,0.7723,0.8282,0.8799,0.6358,0.9814,0.4675,0.9824,0.2905,0.8131,0.6375,0.7772,0.6906,0.8997,0.589,0.8954,0.6126)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/ZBTB33.eps')

# K562 - fdr_4 - CEBPB 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/CEBPB_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/CEBPB.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / CEBPB',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','70.34','73.15','73.14','58.19','73.2','70.86','72.07','73.42','77.06','77.81'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9651,0.3364,0.9647,0.3358,0.9923,0.1993,0.9995,0.0619,0.9999,0.0098,0.8483,0.3297,0.8267,0.3957,0.9928,0.1728,0.9922,0.1908)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/CEBPB.eps')

# K562 - fdr_4 - GABP 
roc=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/GABP_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','69.17','87.58','87.5','96.01','85.69','80.18','85.59','85.72','92.91','93.5'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8806,0.8979,0.8791,0.897,0.9406,0.9225,0.9853,0.5763,0.9928,0.3939,0.8073,0.7927,0.7585,0.8413,0.9172,0.8781,0.9112,0.902)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/ROC/K562/fdr_4/GABP.eps')

