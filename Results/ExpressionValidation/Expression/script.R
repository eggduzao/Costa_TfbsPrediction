library(GEOquery)
library(limma)


k562 = getGEO("GSE12760", GSEMatrix=TRUE)
h1 <- getGEO("GSE14863", GSEMatrix=TRUE)

pheno=pData(phenoData(h1[[1]]))
pheno$description[1]=pheno$description[2]

h1_data=exprs(h1[[1]])
annot=fData(h1[[1]])


pheno_k562=pData(phenoData(k562[[1]]))
data=exprs(k562[[1]])
k562_data=data[,grep("562",pheno_k562$title)]
GM12878_data=data[,grep("GM12878",pheno_k562$title)]


data=merge(h1_data,k562_data,by="row.names")
row.names(data)=data$Row.names
data=data[,-1]
data=merge(data,GM12878_data,by="row.names")
row.names(data)=data$Row.names
data=data[,-1]

data=normalizeBetweenArrays(as.matrix(data))


design <- model.matrix(~ 0+factor(c(1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)))
colnames(design) <- c("H1","K562","GM12878")

fit <- lmFit(data, design)
contrast.matrix <- makeContrasts(K562-H1, K562-GM12878, GM12878-H1, levels=design)
fit2 <- contrasts.fit(fit, contrast.matrix)
fit2 <- eBayes(fit2)

results_05 <- decideTests(fit2,lfc=1)

results=topTable(fit2, coef=1, number=dim(data)[1], adjust="BH",sort.by="none")

data_final=as.data.frame(data)
data_final$fold=results$logFC
data_final$Symbol=annot[,10]
data_final$de=results_05[,1]

data_final$h1=rowMeans(data_final[,c(1:4)])
data_final$k562=rowMeans(data_final[,c(5:25)])
data_final$symbol_clean=as.character(lapply(strsplit(as.character(data_final$Symbol), split="//"), "[", 2))

write.table(data_final[,c("symbol_clean","fold","de","h1","k562")],"de_k562_h1.txt",quote=FALSE, row.names = FALSE)

results=topTable(fit2, coef=2, number=dim(data)[1], adjust="BH",sort.by="none")

data_final=as.data.frame(data)
data_final$fold=results$logFC
data_final$Symbol=annot[,10]
data_final$de=results_05[,1]

data_final$GM12878=rowMeans(data_final[,c(26:45)])
data_final$k562=rowMeans(data_final[,c(5:25)])
data_final$symbol_clean=as.character(lapply(strsplit(as.character(data_final$Symbol), split="//"), "[", 2))

write.table(data_final[,c("symbol_clean","fold","de","GM12878","k562")],"de_k562_GM12878.txt",quote=FALSE, row.names = FALSE)

results=topTable(fit2, coef=3, number=dim(data)[1], adjust="BH",sort.by="none")

data_final=as.data.frame(data)
data_final$fold=results$logFC
data_final$Symbol=annot[,10]
data_final$de=results_05[,1]

data_final$h1=rowMeans(data_final[,c(1:4)])
data_final$GM12878=rowMeans(data_final[,c(26:45)])
data_final$symbol_clean=as.character(lapply(strsplit(as.character(data_final$Symbol), split="//"), "[", 2))

write.table(data_final[,c("symbol_clean","fold","de","GM12878","h1")],"de_GM12878_H1.txt",quote=FALSE, row.names = FALSE)

