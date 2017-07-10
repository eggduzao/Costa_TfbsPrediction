
###########################################################################################################################################
# Correlation Function
###########################################################################################################################################

# Creating Correlation Function
# Formula are the name of the columns. It must be y~x
#createCorrelation <- function(vec1, vec2, vec3, vec4, outFileName, xlab, ylab, corrtype){
createCorrelation <- function(vec1, vec2, outFileName, xlab, ylab, corrtype){

  colVec = rainbow(3)
  colVec = c("black")

  # Plotting graph
  postscript(outFileName, width=7.0, height=6.0, horizontal=FALSE, paper='special')
  par(mar=c(5,5,4,2))
  plot(vec1, vec2, xlab=xlab, ylab=ylab, main="", col=colVec[1], pch = 1, xlim = c(-7,5), ylim = c(-0.8,0.8),
       cex.lab=2.0, cex.axis=1.5, cex.main=1.7, cex.sub=2.0)
  #points(vec1,vec3,col=colVec[2], pch = 1)
  #points(vec1,vec4,col=colVec[3], pch = 1)
  grid()
  #legend(-7,0.8,c("GM12878_K562","H1hesc_GM12878","H1hesc_K562"),col=colVec,pch = 1)
  dev.off()
  system(paste('epstopdf',outFileName,sep=' '))
  system(paste('rm',outFileName,sep=' '))

}

###########################################################################################################################################
# Execution
###########################################################################################################################################

# Parameters
inFile1 = "../Results/GM12878_K562/multivar_table/multivar_table.txt"
inFile2 = "../Results/H1hesc_GM12878/multivar_table/multivar_table.txt"
inFile3 = "../Results/H1hesc_K562/multivar_table/multivar_table.txt"
outLoc = "./"
corrtype = "spearman"
method = "HINTBC"
metric = "flr"

# Data
data1 = read.table(inFile1, sep='\t', header=T, row.names = 1)
data2 = read.table(inFile2, sep='\t', header=T, row.names = 1)
data3 = read.table(inFile3, sep='\t', header=T, row.names = 1)

# Vectors
name1 = "EXPR_FOLDCHANGE"
name2 = paste(method,metric,"ks_stat",sep="_")
vec1 = data1[,name1]
vec2 = data1[,name2]
vec3 = data2[,name2]
vec4 = data3[,name2]

# Correlation graph
outFileName = paste(outLoc,paste(method,metric,sep="_"),'.eps',sep='')
#createCorrelation(vec1,vec2,vec3,vec4,outFileName,name1,name2,corrtype)
createCorrelation(vec1,vec4,outFileName,name1,name2,corrtype)


