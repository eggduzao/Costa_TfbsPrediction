library(plotrix)
cols=rainbow(14)
styVec=c(1,1,1,1,1,1,1,1,1,1,1,1,1,1)
segLen=c(rep(2,14),rep(.3,14))
legendStyVec=c(styVec,rep(1,14))
legendCol=c(cols,rep('white',14))
mar.default <- c(5,5,4,2)
source('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend5.R')

# H1hesc - GABP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/H1hesc/GABP_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/H_GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10,19,20,11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.495,c('PWM','TC','FS','HINT','HINT-BC','HINT-BCN','Boyle','Neph','Cuellar','Centipede','DNase2TF','FLR','PIQ','Wellington','32.12','77.54','22.85','95.65','96.91','96.76','89.41','88.55','76.64','86.37','92.81','87.06','89.99','89.71'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.47,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.468,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/H_GABP.eps')

# H1hesc - JUN 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/H1hesc/JUN_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/H_JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10,19,20,11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.495,c('PWM','TC','FS','HINT','HINT-BC','HINT-BCN','Boyle','Neph','Cuellar','Centipede','DNase2TF','FLR','PIQ','Wellington','13.51','87.33','44.43','95.23','96.4','96.03','92.81','92.79','82.67','90.83','94.23','89.91','93.09','92.94'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.47,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.468,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/H_JUN.eps')

# H1hesc - SIX5 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/H1hesc/SIX5_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/H_SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10,19,20,11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / SIX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.495,c('PWM','TC','FS','HINT','HINT-BC','HINT-BCN','Boyle','Neph','Cuellar','Centipede','DNase2TF','FLR','PIQ','Wellington','42.49','76.32','28.67','87.6','89.65','88.93','83.55','83.42','70.04','75.61','85.78','81.3','46.04','83.63'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.47,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.468,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/H_SIX5.eps')

# H1hesc - YY1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/H1hesc/YY1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/H_YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10,19,20,11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='H1-hESC / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.495,c('PWM','TC','FS','HINT','HINT-BC','HINT-BCN','Boyle','Neph','Cuellar','Centipede','DNase2TF','FLR','PIQ','Wellington','42.98','72.19','28.61','81.76','83.48','82.83','78.25','78.23','78.67','76.86','80.22','76.36','78.47','78.35'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.47,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.468,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/H_YY1.eps')

########################################################################################

# K562 - GABP 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/K562/GABP_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/K_GABP.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / GABPA',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.495,c('PWM','TC','FS','HINT','HINT-BC','HINT-BCN','Boyle','Neph','Cuellar','Centipede','DNase2TF','FLR','PIQ','Wellington','24.88','83.21','7.48','93.05','94.05','93.82','90.55','90.27','80.49','88.2','92.28','86.8','91.06','90.72'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.47,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.468,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/K_GABP.eps')

# K562 - JUN 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/K562/JUN_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/K_JUN.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / C-jun',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.495,c('PWM','TC','FS','HINT','HINT-BC','HINT-BCN','Boyle','Neph','Cuellar','Centipede','DNase2TF','FLR','PIQ','Wellington','12.76','90.7','28.23','96.53','97.21','96.99','95.06','95.24','71.74','91.57','95.99','92.93','95.34','95.18'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.47,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.468,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/K_JUN.eps')

# K562 - SIX5 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/K562/SIX5_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/K_SIX5.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / SIX5',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.495,c('PWM','TC','FS','HINT','HINT-BC','HINT-BCN','Boyle','Neph','Cuellar','Centipede','DNase2TF','FLR','PIQ','Wellington','41.16','89.78','8.55','95.81','96.64','96.38','94.11','94.12','79.39','90.22','95.2','91.88','50.15','94.24'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.47,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.468,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/K_SIX5.eps')

# K562 - YY1 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/K562/YY1_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/K_YY1.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,7,8,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='K562 / YY1',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.495,c('PWM','TC','FS','HINT','HINT-BC','HINT-BCN','Boyle','Neph','Cuellar','Centipede','DNase2TF','FLR','PIQ','Wellington','54.18','92.06','15.23','95.69','96.18','96.01','94.65','94.61','92.13','92.41','95.28','93.28','94.81','94.72'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.47,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.468,'AUC(%)',font=2,cex=0.8)
for (i in (1:14)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/K_YY1.eps')

########################################################################################

cols=rainbow(6)
styVec=c(1,1,1,1,1,1)
segLen=c(rep(2,6),rep(.3,6))
legendStyVec=c(styVec,rep(1,6))
legendCol=c(cols,rep('white',6))
mar.default <- c(5,5,4,2)
source('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend6.R')

# LnCaP - AR_R1881 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/LnCaP/AR_R1881_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/N_AR_R1881.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10,11,12)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='LNCaP / AR(R1881)',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.22,c('PWM','TC','FS','HINT','HINT-BC','HINT-BCN','17.98','20.96','9.34','30.5','32.58','31.66'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.19,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.188,'AUC(%)',font=2,cex=0.8)
for (i in (1:6)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/N_AR_R1881.eps')

# m3134_with_DEX - GR_withDEX 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/m3134_with_DEX/GR_withDEX_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/N_GR_withDEX.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10,11,12)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='m3134(with-DEX) / GR(withDEX)',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.22,c('PWM','TC','FS','HINT','HINT-BC','HINT-BCN','34.45','63.21','17.03','74.97','78.1','77.04'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.19,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.188,'AUC(%)',font=2,cex=0.8)
for (i in (1:6)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/N_GR_withDEX.eps')

# Mcf7 - ER_40min 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/Mcf7/ER_40min_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/N_ER_40min.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10,11,12)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='Mcf7 / ER(40m)',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.22,c('PWM','TC','FS','HINT','HINT-BC','HINT-BCN','27.04','31.47','9.67','43.95','48.19','46.66'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.19,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.188,'AUC(%)',font=2,cex=0.8)
for (i in (1:6)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/N_ER_40min.eps')

# Mcf7 - ER_160min 
roc=read.table('/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/Mcf7/ER_160min_roc.txt',header=TRUE)
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/N_ER_160min.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(mar = mar.default)
roc=roc[,c(1,2,3,4,5,6,7,8,9,10,11,12)]
plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='Mcf7 / ER(160m)',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)
myLegend(-0.035,0.22,c('PWM','TC','FS','HINT','HINT-BC','HINT-BCN','26.65','52.09','10.45','67.69','71.46','70.14'),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)
text(x=0.115,y=0.19,'METHOD',font=2,cex=0.8)
text(x=0.235,y=0.188,'AUC(%)',font=2,cex=0.8)
for (i in (1:6)*2){
  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])
}
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ROC/NatMet/Figure/N_ER_160min.eps')

