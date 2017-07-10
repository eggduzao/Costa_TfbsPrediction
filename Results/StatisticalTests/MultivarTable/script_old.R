library(gplots)
library(scatterplot3d)
library(rpart)


d=read.table("table_changed.csv",header=TRUE)

#d$log(d[,13]+min(d[,13])+1)
rownames(d)=paste0(d[,1],"_",d[,2],"_",d[,3])

#tranforming TC in log scale
d$NORM_TC_AVG_CHIP_UNCORRECTED=log2(d$NORM_TC_AVG_CHIP_UNCORRECTED+1)
d$NORM_TC_AVG_DNASE_UNCORRECTED=log2(d$NORM_TC_AVG_DNASE_UNCORRECTED+1)
d$NORM_TC_AVG_CHIP_UNCORRECTED=log2(d$TC_AVG_CHIP_UNCORRECTED+1)
d$NORM_TC_AVG_DNASE_UNCORRECTED=log2(d$TC_AVG_DNASE_UNCORRECTED+1)

# selection with on AUC, NORM TC
sel=c(4,5,6,7,8,9,10,16,17,18,19,20,22,28,35,37)

#correlation and scatter plots

pdf("heatmap.pdf")
corr=cor(d[,sel],use="pairwise.complete.obs")
heatmap.2(corr,trace="none",col=redblue(256),margins=c(15,15),cexCol=0.6,cexRow=0.6)
dev.off()
pdf("scatter.pdf")
plot(d[,sel])
dev.off()

# PCA analysis

fit <- princomp(d[complete.cases(d),sel],cor=TRUE)
biplot(fit)

pdf("PCA.pdf")
par(mfrow=c(1,2))
biplot(fit,choices = c(2,1))
biplot(fit,choices = c(3,1))
dev.off()

# Association betwenn AUC, AVE. TC and AVE. FOS

d_sel=d[,c("HINT_BC_AUC_10","TC_AVG_CHIP_UNCORRECTED","FOS_AVG_CHIP_CORRECTED","NORM_TC_AVG_CHIP_UNCORRECTED")]

pdf("AUC_FOS_TC_chip.pdf")
sp=scatterplot3d(d_sel[,c(2,3,1)],highlight.3d=TRUE,  type="h")
fit <- lm(HINT_BC_AUC_10 ~ TC_AVG_CHIP_UNCORRECTED + FOS_AVG_CHIP_UNCORRECTED,d) 
sp$plane3d(fit)
dev.off()

# joint TC and FOS statistic and rankings (chip evidence)
d_sel_n=data.frame(scale(d_sel))
d_sel$ranking =  d_sel$"NORM_TC_AVG_CHIP_UNCORRECTED" + d_sel$"FOS_AVG_CHIP_CORRECTED"

cor(d_sel,use="pairwise.complete.obs")
d_sel$FOS_RANKING=rank(d_sel$"FOS_AVG_CHIP_CORRECTED")
d_sel$TC_RANKING=rank(d_sel$"NORM_TC_AVG_CHIP_UNCORRECTED")
d_sel$RANKING_RANKING=rank(d_sel$"ranking")
d_sel$AUC_RANKING=rank(d_sel$"HINT_BC_AUC_10")


# joint TC and FOS statistic and rankings (DHS evidence)

d_sel_dnase=d[,c("HINT_BC_AUC","NORM_TC_AVG_FP_UNCORRECTED","FOS_AVG_DNASE_CORRECTED")]

pdf("AUC_FOS_TC_dnase.pdf")
sp=scatterplot3d(d_sel_dnase[,c(2,3,1)],highlight.3d=TRUE,  type="h")
fit <- lm(HINT_BC_AUC_10 ~ TC_AVG_DNASE_UNCORRECTED + FOS_AVG_DNASE_CORRECTED,d) 
sp$plane3d(fit)
dev.off()

d_sel_dnase_n=data.frame(scale(d_sel_dnase))
d_sel_dnase$ranking = d_sel_dnase_n$"NORM_TC_AVG_FP_UNCORRECTED" + d_sel_dnase_n$"FOS_AVG_DNASE_CORRECTED"
cor(d_sel_dnase,use="pairwise.complete.obs")


# rankings

d_sel_dnase$FOS_RANKING=rank(d_sel_dnase$"FOS_AVG_DNASE_CORRECTED")
d_sel_dnase$TC_RANKING=rank(d_sel_dnase$"NORM_TC_AVG_DNASE_UNCORRECTED")
d_sel_dnase$RANKING_RANKING=rank(d_sel_dnase$"ranking")
d_sel_dnase$AUC_RANKING=rank(d_sel_dnase$"HINT_BC_AUC")
d_sel_dnase$AUC_RANKING_TC=rank(d$"TC_AUC_10")

# evaluation of AUC for lowest vs. highest FOS

lowest_fos_dnase=head(d_sel_dnase[order(d_sel_dnase$FOS_RANKING),],26)
lowest_fos=head(d_sel[order(d_sel$FOS_RANKING),],26)

high_fos_dnase=tail(d_sel_dnase[order(d_sel_dnase$FOS_RANKING),],26)
high_fos=tail(d_sel[order(d_sel$FOS_RANKING),],26)


pdf("AUC_lowest_fos.pdf")
boxplot(lowest_fos_dnase$HINT_BC_AUC_10,high_fos_dnase$HINT_BC_AUC_10,names=c("Lowest 10 Perc.","Highest 10 Perc"),ylab="HINT-BC AUC (10% FDR)",xlab="Protection Score")
dev.off()

fast=grep("_ER_|_AR_|_GR_",rownames(d))
med=grep("FOS",rownames(d))
low=grep("CTCF",rownames(d))
#med=grep("AP",rownames(d))

#pdf("dist_fos.pdf",paper="A4")

#plot(d_sel_dnase$FOS_AVG_DNASE_CORRECTED,d_sel_dnase$HINT_BC_AUC)
#text(d_sel_dnase$FOS_AVG_DNASE_CORRECTED,d_sel_dnase$HINT_BC_AUC,rownames(d),cex=0.8)
#text(d_sel_dnase$FOS_AVG_DNASE_CORRECTED[fast],d_sel_dnase$HINT_BC_AUC[fast],rownames(d[fast,]),cex=0.8,col="red")
#text(d_sel_dnase$FOS_AVG_DNASE_CORRECTED[low],d_sel_dnase$HINT_BC_AUC[low],rownames(d[low,]),cex=0.8,col="green")

plot(d$PROTECT_AVG_DNASE_CORRECTED,d_sel_dnase$HINT_BC_AUC_10)
text(d$PROTECT_AVG_DNASE_CORRECTED,d_sel_dnase$HINT_BC_AUC_10,rownames(d),cex=0.8)
text(d$PROTECT_AVG_DNASE_CORRECTED[fast],d_sel_dnase$HINT_BC_AUC_10[fast],rownames(d[fast,]),cex=0.8,col="red")
text(d$PROTECT_AVG_DNASE_CORRECTED[med],d_sel_dnase$HINT_BC_AUC_10[med],rownames(d[med,]),cex=0.8,col="blue")
text(d$PROTECT_AVG_DNASE_CORRECTED[low],d_sel_dnase$HINT_BC_AUC_10[low],rownames(d[low,]),cex=0.8,col="green")



dev.off()



pdf("dist.pdf",paper="A4")

par(mfrow=c(3,2))

plot(d_sel$NORM_TC_AVG_CHIP_UNCORRECTED,d_sel$HINT_BC_AUC_10)
text(d_sel$NORM_TC_AVG_CHIP_UNCORRECTED,d_sel$HINT_BC_AUC_10,rownames(d),cex=0.5)

plot(d_sel_dnase$NORM_TC_AVG_DNASE_UNCORRECTED,d_sel_dnase$HINT_BC_AUC_10)
text(d_sel_dnase$NORM_TC_AVG_DNASE_UNCORRECTED,d_sel_dnase$HINT_BC_AUC_10,rownames(d),cex=0.5)


plot(d_sel$FOS_AVG_CHIP_UNCORRECTED,d_sel$HINT_BC_AUC_10)
text(d_sel$FOS_AVG_CHIP_UNCORRECTED,d_sel$HINT_BC_AUC_10,rownames(d),cex=0.5)

plot(d_sel_dnase$FOS_AVG_DNASE_UNCORRECTED,d_sel_dnase$HINT_BC_AUC_10)
text(d_sel_dnase$FOS_AVG_DNASE_UNCORRECTED,d_sel_dnase$HINT_BC_AUC_10,rownames(d),cex=0.5)

plot(d_sel$ranking,d_sel$HINT_BC_AUC_10)
text(d_sel$ranking,d_sel$HINT_BC_AUC_10,rownames(d),cex=0.5)

plot(d_sel_dnase$ranking,d_sel_dnase$HINT_BC_AUC_10)
text(d_sel_dnase$ranking,d_sel_dnase$HINT_BC_AUC_10,rownames(d),cex=0.5)

	

dev.off()

pdf("dist_final.pdf",paper="A4")

plot(d_sel$ranking,d_sel$HINT_BC_AUC)
text(d_sel$ranking,d_sel$HINT_BC_AUC,rownames(d),cex=0.5)

dev.off()




#random forest / exploratory analysis (not used

fit <- rpart(HINT_BC_AUC_10~BIAS_PEARSON + NUMBER_PEAKS_with_MOTIFS + PERC_PEAKS_MOTIFS + NORM_TC_AVG_CHIP_UNCORRECTED + FOS_AVG_CHIP_UNCORRECTED + PROTECT_AVG_CHIP_UNCORRECTED + NORM_TC_AVG_CHIP_CORRECTED + FOS_AVG_CHIP_CORRECTED + PROTECT_AVG_CHIP_CORRECTED + NORM_TC_AVG_DNASE_UNCORRECTED + FOS_AVG_DNASE_UNCORRECTED + PROTECT_AVG_DNASE_UNCORRECTED + NORM_TC_AVG_DNASE_CORRECTED + FOS_AVG_DNASE_CORRECTED + PROTECT_AVG_DNASE_CORRECTED + MDC_UNCORRECTED + MDC_CORRECTED, method="anova", data=d)

fit <- rpart(HINT_BC_AUC_10~BIAS_PEARSON +  NORM_TC_AVG_FP_UNCORRECTED + FOS_AVG_FP_UNCORRECTED + PROTECT_AVG_FP_UNCORRECTED + NORM_TC_AVG_FP_CORRECTED + FOS_AVG_FP_CORRECTED + PROTECT_AVG_FP_CORRECTED + MDC_UNCORRECTED + MDC_CORRECTED, method="anova", data=d)


plot(fit, uniform=TRUE)
text(fit, use.n=TRUE, all=TRUE, cex=.8)

pfit<- prune(fit, cp=0.04)
plot(pfit, uniform=TRUE)
text(pfit, use.n=TRUE, all=TRUE, cex=.8)


sum(abs(residuals(fit)))/dim(d)[1]
sum(abs(residuals(pfit)))/dim(d)[1]



sp=scatterplot3d(d[,c("PROTECT_AVG_FP_CORRECTED","FOS_AVG_FP_CORRECTED","HINT_BC_AUC_10")],highlight.3d=TRUE,  type="h")
fit <- lm(HINT_BC_AUC_10 ~ PROTECT_AVG_FP_CORRECTED + FOS_AVG_FP_CORRECTED,d) 
sp$plane3d(fit)


d_n=data.frame(scale(d[,c("PROTECT_AVG_FP_CORRECTED","FOS_AVG_FP_CORRECTED")]))
d$ranking_new = d_n$"PROTECT_AVG_FP_CORRECTED" + d_n$"FOS_AVG_FP_CORRECTED"
#d$ranking_new = predict(fit)


fast=grep("_ER_|_AR_|_GR_",rownames(d))
med=grep("FOS",rownames(d))
low=grep("CTCF",rownames(d))
#med=grep("AP",rownames(d))

#pdf("dist_fos.pdf",paper="A4")

#plot(d_sel_dnase$FOS_AVG_DNASE_CORRECTED,d_sel_dnase$HINT_BC_AUC)
#text(d_sel_dnase$FOS_AVG_DNASE_CORRECTED,d_sel_dnase$HINT_BC_AUC,rownames(d),cex=0.8)
#text(d_sel_dnase$FOS_AVG_DNASE_CORRECTED[fast],d_sel_dnase$HINT_BC_AUC[fast],rownames(d[fast,]),cex=0.8,col="red")
#text(d_sel_dnase$FOS_AVG_DNASE_CORRECTED[low],d_sel_dnase$HINT_BC_AUC[low],rownames(d[low,]),cex=0.8,col="green")


cor(d$ranking_new,d$HINT_BC_AUC_ALL,use="pairwise.complete.obs")
plot(d$ranking_new,d$HINT_BC_AUC_ALL)
text(d$ranking_new,d$HINT_BC_AUC_ALL,rownames(d),cex=0.5)
text(d$ranking_new[fast],d$HINT_BC_AUC_ALL[fast],rownames(d[fast,]),cex=0.8,col="red")
text(d$ranking_new[med],d$HINT_BC_AUC_ALL[med],rownames(d[med,]),cex=0.8,col="blue")
text(d$ranking_new[low],d$HINT_BC_AUC_ALL[low],rownames(d[low,]),cex=0.8,col="green")


