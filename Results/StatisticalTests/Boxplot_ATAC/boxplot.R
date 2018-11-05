
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
    
    postscript(outFileName,width=myWidth,height=6.0,horizontal=FALSE,paper='special')
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
inFileName = "../MultivarTable/table/DU_K562.txt"
outFileNameEps = "boxplot.eps"
outFileNameTest = "boxplot_test.txt"

# Selected Methods and their labels on the boxplot
methodVec = c("PWM_AUPR_ALL", "TC_AUPR_ALL", "HINT_AUPR_ALL", "HINT_BC_AUPR_ALL", "ATAC_TC_50_AUPR_ALL", "scATAC_TC_50_AUPR_ALL", "ATAC_HINT_twoside_1_AUPR_ALL", "sc_ATAC_HINT_twoside_1_AUPR_ALL", "ATAC_HINT_twoside_1_SHIFT_AUPR_ALL", "ATAC_HINT_twoside_1_SHIFT_BC_AUPR_ALL", "sc_ATAC_HINT_twoside_1_SHIFT_AUPR_ALL", "sc_ATAC_HINT_twoside_1_SHIFT_BC_AUPR_ALL")
labelVec = c("PWM", "TC_DNASE", "HINT_DNASE", "HINT-BC_DNASE", "TC_ATAC", "TC_scATAC", "HINT_ATAC", "HINT_scATAC", "HINT_ATAC_SHIFT4", "HINT-BC_ATAC_SHIFT4", "HINT_scATAC_SHIFT4", "HINT-BC_scATAC_SHIFT4")

sel=c(1,4,5,6,7,8)

methodVec=methodVec[sel]
labelVec=labelVec[sel]

# Reading table
bx = read.table(inFileName, sep="\t", header=TRUE)
rownames(bx)=bx[,"FACTOR"]
bx = bx[,methodVec]
bx = bx * 100 # Changing to percentage
colnames(bx) = labelVec

plot(bx[,5],bx[,8],ylim=c(0,100),xlim=c(0,100))
bx$ratio=bx[,5]-bx[,8]


# Sorting by median
medVec = apply(bx,2,median)
sortInd = sort(medVec, decreasing = TRUE, index.return = TRUE, na.last=NA)$ix
bx = bx[,sortInd]

# Print to Friedman Nemenyi
bx2 = bx / 100
#write.table(bx2[,c(1:ncol(bx2)-1)], file = "./FriedmanNemenyi/data.txt", sep = "\t", row.names = FALSE, quote = FALSE)
#write(colnames(bx), file = "./FriedmanNemenyi/header.txt",ncolumns = 1)

# Creating boxplot
createBoxplot(bx,"AUPR Distribution (%)",c(0,20,40,60,80,100),1.1,7,outFileNameEps)

# Performing T-test
hypTest(bx, colnames(bx), outFileNameTest)


