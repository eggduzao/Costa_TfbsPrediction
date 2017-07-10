
# Import
library(gplots)
library(scatterplot3d)
library(rpart)
library(Hmisc)

# Reading data
d=read.table("../table_changed.csv",header=TRUE,sep=",")
rownames(d)=paste0(d[,1],"_",d[,2],"_",d[,3])

# Tranforming TC in log scale
d$NORM_TC_AVG_CHIP_UNCORRECTED=log2(d$NORM_TC_AVG_CHIP_UNCORRECTED+1)
d$NORM_TC_AVG_DNASE_UNCORRECTED=log2(d$NORM_TC_AVG_DNASE_UNCORRECTED+1)
d$NORM_TC_AVG_CHIP_UNCORRECTED=log2(d$TC_AVG_CHIP_UNCORRECTED+1)
d$NORM_TC_AVG_DNASE_UNCORRECTED=log2(d$TC_AVG_DNASE_UNCORRECTED+1)
d$NORM_TC_AVG_FP_CORRECTED=log2(d$TC_AVG_FP_CORRECTED-min(d$TC_AVG_FP_CORRECTED,na.rm=TRUE)+1)

# Selection with on AUC, NORM TC, corrected FOS, corrected TC, 
sel=grep("BIAS|NUMBER|PERC|AUC_ALL|AUPR|_FP_CORR",colnames(d))

# Rank statistics
d$PROTECTION_FINAL_RANK = rank(d$"PROTECT_AVG_FP_CORRECTED") + rank(d$"FOS_AVG_FP_CORRECTED")
d$PROTECTION_FINAL = (d$"PROTECT_AVG_FP_CORRECTED") + (d$"FOS_AVG_FP_CORRECTED")

# Creating label dictionary
labelTable = read.table("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/TF.txt",header=FALSE,sep="\t")
labelDict = labelTable[,2]
names(labelDict) = labelTable[,1]

# Creating cell dictionary
cellTable = read.table("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/Cell.txt",header=FALSE,sep="\t")
cellDict = cellTable[,2]
names(cellDict) = cellTable[,1]

# Label fix
labels=sub("_"," ",rownames(d))
labels=sub("_"," ",labels)
labels=sub("_"," ",labels)
labels=sub("UW ","",labels)
labels=sub("DU ","",labels)
newlabels = c()
for(l in labels){
  if(l == "Mcf7 ER 160min"){ newlabels = c(newlabels,"ER-160m(MCF-7)")}
  else if(l == "Mcf7 ER 40min"){ newlabels = c(newlabels,"ER-40m(MCF-7)")}
  else if(l == "LnCaP AR R1881"){ newlabels = c(newlabels,"AR-R1881(LNCaP)")}
  else if(l == "LnCaP AR ethl"){ newlabels = c(newlabels,"AR-ethl(LNCaP)")}
  else if(l == "m3134 with DEX_GR_withDEX"){ newlabels = c(newlabels,"GR(m3134)")}
  else{ 
    nameVec = strsplit(l," ")[[1]]
    newlabels = c(newlabels, paste(labelDict[nameVec[2]],"(",cellDict[nameVec[1]],")",sep=""))
  }
}

# Selected color TFs
fast=grep("MCF|LNCaP|m3134",newlabels)
med=grep("C-jun\\(",newlabels)
low=grep("CTCF\\(",newlabels)

# Graph
pdf("hintbc_auc_prot.pdf")
#postscript("hintbc_auc_prot.eps", width=7.0, height=7.0, horizontal=FALSE, paper='special')
p=plot(d$MINSUM_PROT_AVG_FP_CORRECTED,d$HINT_BC_AUC_ALL,ylab="HINT-BC AUC",xlab="Protection score",pch = 20,cex=0.5)
abline(v=0,col='red')
sel=(d$HINT_BC_AUC_ALL<0.8)
points(d$MINSUM_PROT_AVG_FP_CORRECTED[sel],d$HINT_BC_AUC_ALL[sel],col="gray",pch = 20)
text(d$MINSUM_PROT_AVG_FP_CORRECTED[sel],d$HINT_BC_AUC_ALL[sel],newlabels[sel],pos=1,cex=0.9,col="gray")
sel=(d$MINSUM_PROT_AVG_FP_CORRECTED>6)
points(d$MINSUM_PROT_AVG_FP_CORRECTED[sel],d$HINT_BC_AUC_ALL[sel],col="gray",pch = 20)
text(d$MINSUM_PROT_AVG_FP_CORRECTED[sel],d$HINT_BC_AUC_ALL[sel],newlabels[sel],pos=1,cex=0.9,col="gray")
points(d$MINSUM_PROT_AVG_FP_CORRECTED[fast],d$HINT_BC_AUC_ALL[fast],col="red",pch = 20)
text(d$MINSUM_PROT_AVG_FP_CORRECTED[fast],d$HINT_BC_AUC_ALL[fast],newlabels[fast],pos=1,cex=0.9,col="red")
points(d$MINSUM_PROT_AVG_FP_CORRECTED[med],d$HINT_BC_AUC_ALL[med],col="blue",pch = 20)
text(d$MINSUM_PROT_AVG_FP_CORRECTED[med],d$HINT_BC_AUC_ALL[med],newlabels[med],pos=1,cex=0.9,col="blue")
points(d$MINSUM_PROT_AVG_FP_CORRECTED[low],d$HINT_BC_AUC_ALL[low],col="green",pch = 20)
text(d$MINSUM_PROT_AVG_FP_CORRECTED[low],d$HINT_BC_AUC_ALL[low],newlabels[low],pos=1,cex=0.9,col="green")
#print(p)
dev.off()

# Writing graph in tabular form
toWrite = c("FACTOR(CELL)","PROTECTION SCORE","HINT-BC AUC")
for(i in 1:length(d$MINSUM_PROT_AVG_FP_CORRECTED)){
  toWrite = c(toWrite,newlabels[i],d$MINSUM_PROT_AVG_FP_CORRECTED[i],d$HINT_BC_AUC_ALL[i])
}
write(toWrite,file="hintbc_auc_prot.txt",ncolumns=3,append=FALSE,sep='\t')


