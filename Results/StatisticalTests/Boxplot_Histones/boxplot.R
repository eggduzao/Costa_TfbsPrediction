
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

  postscript(outFileName,width=myWidth,height=5.0,horizontal=FALSE,paper='special')
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
  system(paste("rm",outFileName,sep=" "))

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

# Input
inFileName = "../MultivarTable/table/final_table.csv"
outFileNameEps = "boxplot.eps"
outFileNameTest = "boxplot_test.txt"

# Selected Methods and their labels on the boxplot
methodVec = c("HINT_BC_D13_DU_AUPR_ALL", "HINT_BC_AUPR_ALL", "HINT_AUPR_ALL","HINT_H13_DU_AUPR_ALL", "DNase2TF_AUPR_ALL", "PIQ_AUPR_ALL", "PWM_AUPR_ALL")
labelVec = c("HINT-BC D+H", "HINT BC D", "HINT D", "HINT H", "DNase2TF", "PIQ", "PWM")

# Reading table
bx = read.table(inFileName, sep=",", header=TRUE)
bx = bx[which(bx$LAB=="DU" & (bx$CELL=="H1hesc" | bx$CELL=="K562") & bx$FACTOR != "P300" & bx$FACTOR != "TBP"),] # Selecting TFs
bx = bx[,methodVec] # Selecting methods
colnames(bx) = labelVec
bx = bx * 100 # Changing to percentage

# Sorting by median
medVec = apply(bx,2,median)
sortInd = sort(medVec, decreasing = TRUE, index.return = TRUE, na.last=NA)$ix
bx = bx[,sortInd]

# Print to Friedman Nemenyi
bx2 = bx / 100
#write.table(bx2[,c(1:ncol(bx2))], file = "./FriedmanNemenyi/data.txt", sep = "\t", row.names = FALSE, quote = FALSE)
write(colnames(bx2), file = "./FriedmanNemenyi/header.txt",ncolumns = 1)

# Creating boxplot
createBoxplot(bx,"AUPR Distribution (%)",c(0,20,40,60,80,100),1.2,5,outFileNameEps)

# Performing T-test
hypTest(bx, colnames(bx), outFileNameTest)


