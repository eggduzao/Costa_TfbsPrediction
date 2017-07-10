library(gplots)
library(scatterplot3d)
library(rpart)
library(Hmisc)

d=read.table("table_changed.csv",header=TRUE)

#d$log(d[,13]+min(d[,13])+1)
rownames(d)=paste0(d[,1],"_",d[,2],"_",d[,3])



#tranforming TC in log scale
d$NORM_TC_AVG_CHIP_UNCORRECTED=log2(d$NORM_TC_AVG_CHIP_UNCORRECTED+1)
d$NORM_TC_AVG_DNASE_UNCORRECTED=log2(d$NORM_TC_AVG_DNASE_UNCORRECTED+1)
d$NORM_TC_AVG_CHIP_UNCORRECTED=log2(d$TC_AVG_CHIP_UNCORRECTED+1)
d$NORM_TC_AVG_DNASE_UNCORRECTED=log2(d$TC_AVG_DNASE_UNCORRECTED+1)
d$NORM_TC_AVG_FP_CORRECTED=log2(d$TC_AVG_FP_CORRECTED-min(d$TC_AVG_FP_CORRECTED,na.rm=TRUE)+1)

#removing selected rows



# selection with on AUC, NORM TC, corrected FOS, corrected TC, 
sel=grep("BIAS|NUMBER|PERC|AUC_ALL|AUPR|_FP_CORR",colnames(d))

#correlation and scatter plots

pdf("heatmap.pdf")
corr=cor(d[,sel],use="pairwise.complete.obs")
p=rcorr(as.matrix(d[,sel]),type="pearson")
heatmap.2(corr,trace="none",col=redblue(256),margins=c(15,15),cexCol=0.6,cexRow=0.6)
dev.off()
pdf("scatter.pdf")
plot(d[,sel])
dev.off()

# PCA analysis

fit <- princomp(d[complete.cases(d),sel],cor=TRUE)
#biplot(fit)

pdf("PCA.pdf")
par(mfrow=c(1,2))
biplot(fit,choices = c(2,1))
biplot(fit,choices = c(3,1))
dev.off()

# decision tree

fit <- rpart(HINT_BC_AUPR~BIAS_PEARSON + NORM_TC_AVG_FP_CORRECTED + FOS_AVG_FP_CORRECTED + PROTECT_AVG_FP_CORRECTED + MDC_UNCORRECTED + MDC_CORRECTED, method="anova", data=d)


plot(fit, uniform=TRUE)
text(fit, use.n=TRUE, all=TRUE, cex=.8)

pfit<- prune(fit, cp=0.04)
plot(pfit, uniform=TRUE)
text(pfit, use.n=TRUE, all=TRUE, cex=.8)


sum(abs(residuals(fit)))/dim(d)[1]
sum(abs(residuals(pfit)))/dim(d)[1]


#final protection score

pdf("PROT_FOS_AUC.pdf")
sp=scatterplot3d(d[,c("PROTECT_AVG_FP_CORRECTED","FOS_AVG_FP_CORRECTED","HINT_BC_AUC_ALL")],highlight.3d=TRUE,  type="h")
fit <- lm(HINT_BC_AUC_ALL ~ PROTECT_AVG_FP_CORRECTED + FOS_AVG_FP_CORRECTED,d) 
sp$plane3d(fit)
dev.off()

pdf("PROT_FOS_AUPR.pdf")
sp=scatterplot3d(d[,c("PROTECT_AVG_FP_CORRECTED","FOS_AVG_FP_CORRECTED","HINT_BC_AUC_ALL")],highlight.3d=TRUE,  type="h")
fit <- lm(HINT_BC_AUC_ALL ~ PROTECT_AVG_FP_CORRECTED + FOS_AVG_FP_CORRECTED,d) 
sp$plane3d(fit)
dev.off()


#d$PROTECTION_FINAL = predict(fit)

d$PROTECTION_FINAL_RANK = rank(d$"PROTECT_AVG_FP_CORRECTED") + rank(d$"FOS_AVG_FP_CORRECTED")
d$PROTECTION_FINAL = (d$"PROTECT_AVG_FP_CORRECTED") + (d$"FOS_AVG_FP_CORRECTED")


rc=rcorr(as.matrix(d[,c("BIAS_PEARSON","MINSUM_PROT_AVG_FP_CORRECTED","PROTECT_AVG_FP_CORRECTED","FOS_AVG_FP_CORRECTED","PROTECTION_FINAL","HINT_BC_AUC_ALL","TC_AUC_ALL","HINT_BC_AUPR_ALL","TC_AUPR_ALL")]),type="spearman")

heatmap.2(rc$r,trace="none",col=redblue(256),margins=c(15,15),cexCol=0.6,cexRow=0.6)


pdf("auc_protection.pdf",paper="A4")

#factors discussed in sung et al. 
fast=grep("_ER_|_AR_|_GR_",rownames(d))
med=grep("JUN$",rownames(d))# Sung describes AP1 and indicaates a chip-seq from jun in the supplement
low=grep("CTCF$",rownames(d))
#med=grep("AP",rownames(d))
plot(d$PROTECTION_FINAL,d$HINT_BC_AUC_ALL)
#text(d$PROTECTION_FINAL,d$HINT_BC_AUC_ALL,rownames(d),cex=0.5)
text(d$PROTECTION_FINAL[fast],d$HINT_BC_AUC_ALL[fast],rownames(d[fast,]),cex=0.8,col="red")
text(d$PROTECTION_FINAL[med],d$HINT_BC_AUC_ALL[med],rownames(d[med,]),cex=0.8,col="blue")
text(d$PROTECTION_FINAL[low],d$HINT_BC_AUC_ALL[low],rownames(d[low,]),cex=0.8,col="green")
dev.off()

rc_auc=rcorr(as.matrix(d[,grep("MINSUM_PROT_AVG_FP_CORRECTED|_AUC_ALL",colnames(d))]),type="spearman")
p=rc_auc$P[7,]



pdf("hintbc_tc_auc_prot.pdf",paper="A4")
par(mfrow=c(2,1))

plot(d$MINSUM_PROT_AVG_FP_CORRECTED,d$HINT_BC_AUC_ALL)
text(d$MINSUM_PROT_AVG_FP_CORRECTED,d$HINT_BC_AUC_ALL,rownames(d),cex=0.3)
text(d$MINSUM_PROT_AVG_FP_CORRECTED[fast],d$HINT_BC_AUC_ALL[fast],rownames(d[fast,]),cex=0.5,col="red")
text(d$MINSUM_PROT_AVG_FP_CORRECTED[med],d$HINT_BC_AUC_ALL[med],rownames(d[med,]),cex=0.5,col="blue")
text(d$MINSUM_PROT_AVG_FP_CORRECTED[low],d$HINT_BC_AUC_ALL[low],rownames(d[low,]),cex=0.5,col="green")

plot(d$MINSUM_PROT_AVG_FP_CORRECTED,d$TC_AUC_ALL)
text(d$MINSUM_PROT_AVG_FP_CORRECTED,d$TC_AUC_ALL,rownames(d),cex=0.3)
text(d$MINSUM_PROT_AVG_FP_CORRECTED[fast],d$TC_AUC_ALL[fast],rownames(d[fast,]),cex=0.5,col="red")
text(d$MINSUM_PROT_AVG_FP_CORRECTED[med],d$TC_AUC_ALL[med],rownames(d[med,]),cex=0.5,col="blue")
text(d$MINSUM_PROT_AVG_FP_CORRECTED[low],d$TC_AUC_ALL[low],rownames(d[low,]),cex=0.5,col="green")
dev.off()


pdf("hintbc_auc_prot.pdf")
labels=sub("_"," ",rownames(d))
labels=sub("_"," ",labels)
labels=sub("_"," ",labels)
labels=sub("UW ","",labels)
labels=sub("DU ","",labels)
plot(d$MINSUM_PROT_AVG_FP_CORRECTED,d$HINT_BC_AUC_ALL,ylab="HINT-BC AUC",xlab="Protection score",pch = 20,cex=0.5)
abline(v=0,col='red')
sel=(d$HINT_BC_AUC_ALL<0.8)
text(d$MINSUM_PROT_AVG_FP_CORRECTED[sel],d$HINT_BC_AUC_ALL[sel],labels[sel],pos=1,cex=0.9,col="gray")
sel=(d$MINSUM_PROT_AVG_FP_CORRECTED>6)
text(d$MINSUM_PROT_AVG_FP_CORRECTED[sel],d$HINT_BC_AUC_ALL[sel],labels[sel],pos=1,cex=0.9,col="gray")
points(d$MINSUM_PROT_AVG_FP_CORRECTED[fast],d$HINT_BC_AUC_ALL[fast],col="red",pch = 20)
text(d$MINSUM_PROT_AVG_FP_CORRECTED[fast],d$HINT_BC_AUC_ALL[fast],labels[fast],pos=1,cex=0.9,col="red")
points(d$MINSUM_PROT_AVG_FP_CORRECTED[med],d$HINT_BC_AUC_ALL[med],col="blue",pch = 20)
text(d$MINSUM_PROT_AVG_FP_CORRECTED[med],d$HINT_BC_AUC_ALL[med],labels[med],pos=1,cex=0.9,col="blue")
points(d$MINSUM_PROT_AVG_FP_CORRECTED[low],d$HINT_BC_AUC_ALL[low],col="green",pch = 20)
text(d$MINSUM_PROT_AVG_FP_CORRECTED[low],d$HINT_BC_AUC_ALL[low],labels[low],pos=1,cex=0.9,col="green")
dev.off()

head(d[order(d$MINSUM_PROT_AVG_FP_CORRECTED),1:10],50)

d$DIFF=d$HINT_BC_AUC_10-d$HINT_AUC_10
d$DIFF_RANK=rank(d$HINT_BC_AUC_10-d$HINT_AUC_10)


lowest_fos_dnase=head(d[order(d_sel_dnase$FOS_RANKING),],26)
lowest_fos=head(d_sel[order(d_sel$FOS_RANKING),],26)

high_fos_dnase=tail(d_sel_dnase[order(d_sel_dnase$FOS_RANKING),],26)
high_fos=tail(d_sel[order(d_sel$FOS_RANKING),],26)

d$hint_diff=d$HINT_BC_AUPR-d$HINT_AUPR



plot((d$HINT_BC_AUPR),(d$HINT_AUPR))


#d_sel=d[abs(d$hint_diff)>0.1,]
d_sel=d[grep("EGR1|ER_4|UW_Motif_0500",as.character(d$FACTOR)),]
points((d_sel$HINT_BC_AUPR),(d_sel$HINT_AUPR),col="red",pch =19)
lines(c(0,1),c(0,1))
text(d_sel$HINT_BC_AUPR,-0.02+d_sel$HINT_AUPR,d_sel$FACTOR,col='red')




#pdf("aupr_protection.pdf",paper="A4")
#plot(d$PROTECTION_FINAL,d$HINT_BC_AUPR_ALL)
##text(d$PROTECTION_FINAL,d$HINT_BC_AUC_ALL,rownames(d),cex=0.5)
#text(d$PROTECTION_FINAL[fast],d$HINT_BC_AUPR_ALL[fast],rownames(d[fast,]),cex=0.8,col="red")
#text(d$PROTECTION_FINAL[med],d$HINT_BC_AUPR_ALL[med],rownames(d[med,]),cex=0.8,col="blue")
#text(d$PROTECTION_FINAL[low],d$HINT_BC_AUPR_ALL[low],rownames(d[low,]),cex=0.8,col="green")
#dev.off()

#plot(d$MINSUM_PROT_AVG_FP_CORRECTED,d$HINT_BC_AUPR_ALL)
#text(d$MINSUM_PROT_AVG_FP_CORRECTED,d$HINT_BC_AUPR_ALL,rownames(d),cex=0.5)
#text(d$MINSUM_PROT_AVG_FP_CORRECTED[fast],d$HINT_BC_AUPR_ALL[fast],rownames(d[fast,]),cex=0.8,col="red")
#text(d$MINSUM_PROT_AVG_FP_CORRECTED[med],d$HINT_BC_AUPR_ALL[med],rownames(d[med,]),cex=0.8,col="blue")
#text(d$MINSUM_PROT_AVG_FP_CORRECTED[low],d$HINT_BC_AUPR_ALL[low],rownames(d[low,]),cex=0.8,col="green")



