
###################################################################################################
# INPUT
###################################################################################################
#library(colorBrewer)

# Import
library(lattice)
library(reshape)
library(plotrix)
w2l <- function(xx) melt(xx, measure.vars = colnames(xx))

###################################################################################################
# FUNCTIONS
###################################################################################################

# BOXPLOT
createBoxplot <- function(dataFrame, yLabel, abLineList, myCexX, myWidth, outFileName){

  postscript(outFileName,width=myWidth,height=4.0,horizontal=FALSE,paper='special')
  par(cex.axis=1.0)
  test_df2 <- w2l(dataFrame)
  p = bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=90,cex=myCexX), y=list(cex=1.3)), col = "black",
         main = "", xlab = "", ylab = list(yLabel,cex=1.8),
         par.settings = list( plot.symbol=list(col = "black"),box.rectangle = list(col = "black"),
         box.dot = list(col = "black"), box.umbrella= list(lty=1, col = "black")),
         panel=function(...) {
           panel.abline(h=abLineList,lty=2,lwd=1.0,col="gray") 
           panel.bwplot(...)
         })
  print(p)
  dev.off()
  system(paste("epstopdf",outFileName,sep=" "))
  #system(paste("rm",outFileName,sep=" "))

}

# PAIRED T-TEST
hypTest <- function(dataFrame, methodNameVec, outFileName){

  toWrite = c("XXXXX",methodNameVec)
  for (i in (1:ncol(dataFrame))){
    toWrite = c(toWrite,methodNameVec[i])
    for (j in (1:ncol(dataFrame))){
      if(i == j){
        toWrite = c(toWrite,"XXXXX")
      }
      else{
        teste = wilcox.test(dataFrame[,i],dataFrame[,j],paired=TRUE)
        toWrite = c(toWrite,format(teste$p.value,digits=5))
      }
    }
  }
  write(toWrite,file=outFileName,ncolumns=ncol(dataFrame)+1,append=FALSE,sep='\t')

}

###################################################################################################
# EXECUTION
###################################################################################################

aux=c()
# Input file loop
inList = c("AUC_1","AUC_10","AUC_100","AUPR_100")
for(inName in inList){

  # Input
  inFileName = paste("../",inName,".txt",sep="")
  outFileNameEps = paste("./",inName,".eps",sep="")
  outFileNameTest = paste("./",inName,"_Ttest.txt",sep="")

  # Reading table
  bx = read.table(inFileName, header=TRUE)
  #bx = bx[,c(3:ncol(bx))]
  #bx = bx[,c(3,9,12,17,20,24,27,28,29,30,31,34,37,39,40)]
  #bx = bx * 100

  # Friedman Nemenyi order
  bx = bx[,c(29,20,34,40,31,9,3,12,24,14,17,39,37,27)]
  bx = bx * 100

  # Sorting by median
  #medVec = apply(bx,2,median)
  #sortInd = sort(medVec, decreasing = TRUE, index.return = TRUE, na.last=NA)$ix
  #bx = bx[,sortInd]

  # Creating boxplot
  createBoxplot(bx,"AUC Distribution (%)",c(0,20,40,60,80,100),1.3,8,outFileNameEps)

  # Performing T-test
  hypTest(bx, colnames(bx), outFileNameTest)

}
#library(Hmisc)

#res=t(aux[c(4,1,2,3),])

#plot(replicate(dim(res)[1],c(1,2,3,4)),res,type="n")
#col=topo.colors(dim(res)[1])
#for(i in 1:dim(res)[1]){
#  lines(c(1,2,3,4),res[i,],col=col[i])
#}

#r=rcorr(as.matrix(res),type="spearman")




