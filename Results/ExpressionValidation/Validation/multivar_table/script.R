library(Hmisc)

data=read.table("multivar_table.txt",header=T)

#olny hintbc data
sel=c(2,3,4,grep("PIQ",colnames(data)))
sel_data=as.matrix(data[,sel])
cor=rcorr(sel_data) 
pairs(sel_data[,c(1,grep("ks_stat",colnames(sel_data)))])

#all ks_stat data
colMeans(abs(data[,grep("ks_stat",colnames(data))]))

#sign errors
colSums(data[,grep("ks_stat",colnames(data))]*data[,2]<0)


cor_data=read.table("corr_final.csv",header=T)
cor=rcorr(as.matrix(cor_data[,1:5]),type="spearman") 


