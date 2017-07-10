library(lattice)
library(reshape)
w2l <- function(xx) melt(xx, measure.vars = colnames(xx))

# Create Image
postscript('/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/Figure/Boxplot.eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')
par(cex.axis=0.8)

# Methods names
methodNameVec = c('Boyle','Neph','H-HMM (2)','H-HMM (3)','DH-HMM (2)','DH-HMM (3)')

# Read H1-hESC table
bx1=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/H1hesc/fdr_4/fdr_4_box.txt',header=TRUE)
bx1=replace(bx1, abs(bx1) > 100, NA)
bx1_2 = abs(bx1[,c(1,3,8,10,4,6)])
colnames(bx1_2) = methodNameVec

# Read K562 table
bx2=read.table('/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/K562/fdr_4/fdr_4_box.txt',header=TRUE)
bx2=replace(bx2, abs(bx2) > 100, NA)
bx2_2 = abs(bx2[,c(1,3,9,11,5,7)])
colnames(bx2_2) = methodNameVec

# Merge tables
bx = rbind(bx1_2,bx2_2)

# Reorder elements by median
orderVec = c(5,6,1,2,4,3)
bx = bx[,orderVec]
methodNameVec = methodNameVec[orderVec]

colnames(bx) = methodNameVec
test_df2 <- w2l(bx)
bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',
       main = list('',cex=1.8), xlab = '', ylab = list('Footprint distance to TFBSs (bp)',cex=1.8),
       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),
       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),
       panel=function(...) {
           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')
           panel.bwplot(...)
       })
dev.off()
system('epstopdf /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/Boxplot/Figure/Boxplot.eps')


