
library(Hmisc)
library(grDevices)


data_orig=read.table("multivar_table_all_methods.txt",header=TRUE)

row.names(data_orig)=data_orig[,1]

friedman_rank=read.table("rank_friedman.csv")

data=merge(data_orig,friedman_rank,by="row.names")

#data=data_orig[-c(1,2,4,5,6,8,9,10,12,13,14,15,16,18,19,21,22,23,24,26,27,33,34,35,37,38,40,43),]

par(mfrow=c(1,2))

plot(data$FLREXP_JOINT_FLR,data$AUC_10)
text(data$FLREXP_JOINT_FLR,data$AUC_10,data[,1])

plot(data$FLREXP_JOINT_FLR,data$AUPR)
text(data$FLREXP_JOINT_FLR,data$AUPR,data[,1])

#c=rcorr(data, type="spearman")

data=data[order(data$MEDIAN_AUC_10),]
data$RANK_DEC_JOINT_FLR=rank(-data$FLREXP_JOINT_FLR)
data$RANK_DEC_JOINT_FLR[7]=NA
data$RANK_AUC_100=rank(data$Rank_AUC_100)
data$RANK_AUC_10=rank(data$Rank_AUC_10)
data$RANK_AUC_1=rank(data$Rank_AUC_1)
data$RANK_AUPR_100=rank(data$Rank_AUPR)

data=data[order(data$Rank_AUC_10),]

x_size=dim(data)[1]
col=rainbow(x_size)



#c=rcorr(as.matrix(data[,sel]), type="spearman")

pdf("Raibow_dash_graph.pdf",width=16, height=8)

sel1=c(17,23,22,21,24)
sel=sel1

labels=c("FLR-Exp","AUC 100","AUC 10","AUC 1","AUPR")

par(mfrow=c(1,2))
plot(1:length(sel),data[1,sel],type='o',ylim=c(0,1),col=col[1], xaxt = "n",xlab="Metrics", ylab="Accuracy")
axis(1, at=1:length(sel), labels=labels)
for(i in 1:x_size){
 lines(1:length(sel),data[i,sel],type='o',col=col[i],lwd=2)
}


sel=29:33
#c=rcorr(as.matrix(data[,sel]), type="spearman")
plot(1:length(sel),data[1,sel],type='o',xlim=c(1,6),ylim=c(x_size,1),col=col[1], xaxt = "n",xlab="Metrics", ylab="Rankings")
axis(1, at=1:length(sel), labels=labels)
for(i in 1:x_size){
 lines(1:length(sel),data[i,sel],type='o',col=col[i],lwd=2)
}

text(rep(5.5,x_size),data[,33],labels=data[,1],col=col)

dev.off()
