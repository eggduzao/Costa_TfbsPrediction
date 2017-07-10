library(Hmisc)
library(grDevices)


data_orig=read.table("H1hesc_K562/multivar_table/multivar_table.txt",header=TRUE)


names=colnames(data)
data=data_orig[-c(1,2,4,5,6,9,10,12,13,15,16,18,19,21,22,23,24,26,27,33,34,35,37,38,43),]

par(mfrow=c(1,2))

plot(data$DEC_JOINT_FLR,data$AUC_10)
text(data$DEC_JOINT_FLR,data$AUC_10,data[,1])

plot(data$DEC_JOINT_FLR,data$AUPR)
text(data$DEC_JOINT_FLR,data$AUPR,data[,1])



plot(data$HINTBC_flr_ks_stat,data$EXPR_FOLDCHANGE)

c=rcorr(data, type="spearman")

rcorr(data$HINTBC_flr_ks_stat,data$EXPR_FOLDCHANGE, type="spearman")

data=data[order(data$AUC_10),]

data$RANK_AUC_100=rank(-data$AUC_100)
data$RANK_AUC_10=rank(-data$AUC_10)
data$RANK_AUC_1=rank(-data$AUC_1)
data$RANK_AUPR_100=rank(-data$AUPR_100)
data$RANK_DEC_JOINT_FLR=rank(-data$DEC_JOINT_FLR)


pdf("Raibow_dash_graph.pdf",width=16, height=8)
par(mfrow=c(1,2))

x_size=dim(data)[1]
col=rainbow(x_size)

sel1=c(16,22,21,20,23)
sel=sel1

plot(1:length(sel),data[1,sel],type='o',ylim=c(0,1),col=col[1], xaxt = "n",xlab="Metrics", ylab="Accuracy")
axis(1, at=1:length(sel), labels=colnames(data[,sel]))
for(i in 1:x_size){
 lines(1:length(sel),data[i,sel],type='o',col=col[i],lwd=2)
}


sel=c(28,24:27)
plot(1:length(sel),data[1,sel],type='o',xlim=c(1,6),ylim=c(x_size,1),col=col[1], xaxt = "n",xlab="Metrics", ylab="Rankings")
axis(1, at=1:length(sel), labels=colnames(data[,sel]))
for(i in 1:x_size){
 lines(1:length(sel),data[i,sel],type='o',col=col[i],lwd=2)
}

text(rep(5.5,x_size),data[,27],labels=data[,1],col=col)

dev.off()
