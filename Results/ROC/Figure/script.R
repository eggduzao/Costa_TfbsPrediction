library(plotrix)
#cols=rep(rainbow(5),2) # Regular color figure
#styVec=c(2,2,2,2,2,1,1,1,1,1) # Regular color figure
#pointStyVec=c(22,22,22,22,22,21,21,21,21,21) # Regular color figure
#cols=rep(rainbow(2),5) # b&w1
#cols=rep(c("#FF7D7D","#003399"),5) # b&w2
cols=rep(c("#CC6464","#00246B"),5) # b&w3
styVec=c(1,1,2,2,3,3,4,4,5,5)
pointStyVec=c(NA,15,16,16,17,17,21,21,22,22)
segLen=c(rep(2,10),rep(.3,10))
legendStyVec=c(styVec,rep(1,10))
print(legendPchVec)
legendCol=c(cols,rep('white',10))
mar.default <- c(5,5,4,2)
source('/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/myLegend1.R')

# H1hesc - fdr_4 - GABP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/GABP_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Figure/H_GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='',xlab='',main='H1-hESC / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','74.33','90.57','90.55','96.57','90.38','90.12','89.37','87.7','95.73','95.65'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8457,0.9573,0.8477,0.9546,0.9291,0.9156,0.9625,0.7308,0.9699,0.6841,0.8721,0.7957,0.76,0.8493,0.9221,0.945,0.904,0.9576)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
  points(0.09,0.48-(0.0535*((i/2)-1)),pch=pointStyVec[i/2],col=cols[i/2],cex=2.0,lwd=3.0) # Legend points
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Figure/H_GABP.eps')

# H1hesc - fdr_4 - JUN 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/JUN_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Figure/H_JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='',xlab='',main='H1-hESC / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','66.84','85.47','85.38','88.46','84.22','82.83','83.84','84.51','91.3','92.43'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9387,0.8024,0.9402,0.7984,0.9897,0.6802,0.9964,0.5092,0.9975,0.4623,0.9368,0.5703,0.8739,0.6752,0.9902,0.7261,0.9859,0.7617)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
  points(0.09,0.48-(0.0535*((i/2)-1)),pch=pointStyVec[i/2],col=cols[i/2],cex=2.0,lwd=3.0) # Legend points
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Figure/H_JUN.eps')

# H1hesc - fdr_4 - SIX5 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/SIX5_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Figure/H_SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='',xlab='',main='H1-hESC / SIX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','80.19','86.69','86.59','74.3','91.31','90.76','87.57','86.96','95.07','95.33'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8872,0.7926,0.8891,0.7885,0.9795,0.6034,0.9906,0.5023,0.9932,0.4609,0.9176,0.493,0.8293,0.6418,0.9749,0.7268,0.9664,0.7584)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
  points(0.09,0.48-(0.0535*((i/2)-1)),pch=pointStyVec[i/2],col=cols[i/2],cex=2.0,lwd=3.0) # Legend points
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Figure/H_SIX5.eps')

# H1hesc - fdr_4 - YY1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC/H1hesc/fdr_4/YY1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Figure/H_YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,27,28,31,32,19,20,23,24)]
plot(roc[,1],roc[,2],type='n',ylab='',xlab='',main='H1-hESC / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','81.55','88.11','88.06','69.81','86.88','85.82','85.27','84.27','90.84','91.28'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9591,0.7585,0.9599,0.7548,0.9927,0.4892,0.998,0.3021,0.9989,0.2359,0.9319,0.4154,0.8634,0.5088,0.9942,0.4867,0.991,0.5131)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
  points(0.09,0.48-(0.0535*((i/2)-1)),pch=pointStyVec[i/2],col=cols[i/2],cex=2.0,lwd=3.0) # Legend points
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Figure/H_YY1.eps')

#####

# K562 - fdr_4 - GABP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/GABP_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Figure/K_GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='',xlab='',main='K562 / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','69.17','87.58','87.5','96.01','85.69','80.18','85.59','85.72','92.91','93.5'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8806,0.8979,0.8791,0.897,0.9406,0.9225,0.9853,0.5763,0.9928,0.3939,0.8073,0.7927,0.7585,0.8413,0.9172,0.8781,0.9112,0.902)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
  points(0.09,0.48-(0.0535*((i/2)-1)),pch=pointStyVec[i/2],col=cols[i/2],cex=2.0,lwd=3.0) # Legend points
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Figure/K_GABP.eps')

# K562 - fdr_4 - JUN 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/JUN_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Figure/K_JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='',xlab='',main='K562 / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','63.97','80.07','80.02','96.04','86.59','75.13','78.79','81.37','92.27','93.51'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.9124,0.7142,0.9121,0.7125,0.9841,0.8479,0.9968,0.5999,0.9992,0.2664,0.8687,0.5495,0.8402,0.6597,0.975,0.7884,0.9728,0.8306)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
  points(0.09,0.48-(0.0535*((i/2)-1)),pch=pointStyVec[i/2],col=cols[i/2],cex=2.0,lwd=3.0) # Legend points
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Figure/K_JUN.eps')

# K562 - fdr_4 - SIX5 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/SIX5_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Figure/K_SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='',xlab='',main='K562 / SIX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','79.91','87.11','87.16','93.9','93.15','88.33','88.03','88.35','96.34','96.59'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.8714,0.8287,0.8711,0.8287,0.976,0.8644,0.9954,0.6138,0.9978,0.3761,0.8367,0.6708,0.8001,0.7366,0.9612,0.8421,0.9586,0.8644)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
  points(0.09,0.48-(0.0535*((i/2)-1)),pch=pointStyVec[i/2],col=cols[i/2],cex=2.0,lwd=3.0) # Legend points
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Figure/K_SIX5.eps')

# K562 - fdr_4 - YY1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC/K562/fdr_4/YY1_roc.txt',header=TRUE)
postscript('/home/egg/Projects/TfbsPrediction/Results/ROC/Figure/K_YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,9,10,11,12,17,18,7,8,3,4,29,30,33,34,21,22,25,26)]
plot(roc[,1],roc[,2],type='n',ylab='',xlab='',main='K562 / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.59,c('PWM','Cuellar (2)','Cuellar (3)','Centipede','Neph','Boyle','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)  ','86.49','93.2','93.23','92.53','92.26','89.48','91.82','92.36','97.31','97.59'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)
text(x=0.208,y=0.55,'METHOD',font=2,cex=1.3)
text(x=0.494,y=0.55,'AUC(%)',font=2,cex=1.3)
pointList=c(NA,NA,0.946,0.9091,0.9457,0.9094,0.9889,0.8489,0.9981,0.4702,0.9996,0.2192,0.841,0.7569,0.8144,0.8092,0.9834,0.8185,0.982,0.8457)
for (i in (1:10)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)
  points(0.09,0.48-(0.0535*((i/2)-1)),pch=pointStyVec[i/2],col=cols[i/2],cex=2.0,lwd=3.0) # Legend points
}
dev.off()
system('epstopdf /home/egg/Projects/TfbsPrediction/Results/ROC/Figure/K_YY1.eps')


