
###########################################################################################################################################
# Import
###########################################################################################################################################

# Import
#library(lattice)
#library(reshape)
#library(plotrix)
#library(ggplot2)

###########################################################################################################################################
# Input
###########################################################################################################################################

# Input
loc = '/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBias'
inLoc = paste(loc,'KmerTable',sep='/')

# Parameters
flag_All_HS = FALSE
flag_Dataset = TRUE
flag_Forward_Reverse = FALSE
flag_Sampled_Artificial = FALSE
flag_SeqDepth_DatasetCorr = FALSE

###########################################################################################################################################
# Correlation Function
###########################################################################################################################################

# Creating Correlation Function
# Formula are the name of the columns. It must be y~x
createCorrelation <- function(vec1, vec2, formula, data, outFileName, xlab, ylab, corrtype){

  # Calculating correlation
  regLine = lm(formula, data=data) # Regression line (y,x)
  corrTest = cor.test(vec1, vec2, alternative = "two.sided", method = corrtype, conf.level = 0.95) # Correlation
  pValue = corrTest$p.value
  corr = corrTest$estimate

  # Plotting graph
  postscript(outFileName, width=7.0, height=6.0, horizontal=FALSE, paper='special')
  par(mar=c(5,5,4,2))
  plot(vec1, vec2, xlab=xlab, ylab=ylab,
       main=paste('Correlation = ',round(corr,digits = 4),' / p-value = ',round(pValue,digits = 4),sep=''),
       cex.lab=2.0, cex.axis=1.5, cex.main=1.7, cex.sub=2.0)
  #text(vec1, vec2, cex= 0.3, pos = 3)
  abline(regLine,lty=1,lwd=3.0,col="black")
  grid()
  dev.off()
  system(paste('epstopdf',outFileName,sep=' '))

  # Return correlation
  return(round(corr,digits = 6))

}

# Function to standardize values
standardize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

###########################################################################################################################################
# All vs HS
###########################################################################################################################################
# Bias evaluated using All reads as foreground vs only reads found at HS sites. In both cases
# the background consists of all k-mers in HS regions. Only same strands are compared.

if(flag_All_HS){

# Strand Parameters
allLoc = paste(inLoc,"All",sep="/")
hsLoc = paste(inLoc,"HS",sep="/")
strandList = c("All", "F", "R")

# Strand Loop
for (str in 1:length(strandList)) {

  # Dataset Parameters
  strand = strandList[str]
  inList = c(
paste("DNase_H1hesc_6_",strand,sep=""),
paste("DNase_HeLaS3_6_",strand,sep=""),
paste("DNase_HepG2_6_",strand,sep=""),
paste("DNase_Huvec_6_",strand,sep=""),
paste("DNase_K562_6_",strand,sep=""),
paste("DNase_Mcf7_6_",strand,sep=""),
paste("DNaseUW_HepG2_6_",strand,sep=""),
paste("DNaseUW_Huvec_6_",strand,sep=""),
paste("DNaseUW_IMR90_6_",strand,sep=""),
paste("DNaseUW_K562_6_",strand,sep=""),
paste("DNaseUW_LnCaP_6_",strand,sep=""),
paste("DNaseUW_m3134_with_DEX_6_",strand,sep=""),
paste("DNaseUW_m3134_wo_DEX_6_",strand,sep="")
)
  labelList = c("DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","DU_Mcf7","UW_HepG2","UW_Huvec","UW_IMR90","UW_K562","UW_LnCaP","UW_m3134(with)","UW_m3134(wo)")
  outLoc = paste(loc,'CorrelationGraphs','All_HS',sep='/')

  # Initializing correlation table
  toWriteSam = c()
  toWriteArt = c()

  for (i in 1:length(inList)) {

    # Reading input
    dataAll = read.table(paste(allLoc,'/',inList[i],'.txt',sep=''), sep='\t', header=T)
    dataAll = dataAll[order(dataAll[,1]),]
    dataHS = read.table(paste(hsLoc,'/',inList[i],'.txt',sep=''), sep='\t', header=T)
    dataHS = dataHS[order(dataHS[,1]),]

    # Creating data table
    va_1 = standardize(dataAll[,2])
    va_2 = standardize(dataAll[,3])
    vh_1 = standardize(dataHS[,2])
    vh_2 = standardize(dataHS[,3])
    data.sampled = data.frame(X=va_1, Y=vh_1)
    data.artificial = data.frame(X=va_2, Y=vh_2)

    # Correlation graphs
    corr_sam = createCorrelation(data.sampled[,1], data.sampled[,2], Y~X, data.sampled,
                                 paste(outLoc,'/Sampled_',strand,'/',labelList[i],'.eps',sep=''), 
                                 'Bias Considering All Reads', 'Bias Considering HS Reads',"pearson")
    corr_art = createCorrelation(data.artificial[,1], data.artificial[,2], Y~X, data.artificial,
                                 paste(outLoc,'/Artificial_',strand,'/',labelList[i],'.eps',sep=''),
                                 'Bias Considering All Reads', 'Bias Considering HS Reads',"pearson")
    
    # Writing correlations
    toWriteSam = c(toWriteSam,labelList[i],corr_sam)
    toWriteArt = c(toWriteArt,labelList[i],corr_art)

  }

  # Writing correlation to file
  write(toWriteSam,file=paste(outLoc,'/Sampled_',strand,'_corr.txt',sep=''),ncolumns=2,append=FALSE,sep='\t')
  write(toWriteArt,file=paste(outLoc,'/Artificial_',strand,'_corr.txt',sep=''),ncolumns=2,append=FALSE,sep='\t')

}

}

###########################################################################################################################################
# Correlation between datasets
###########################################################################################################################################
# All datasets are compared with each other. Only same-strand datasets are compared.
# Only kmer tables generated using HS method were used in this test.

corrtype = "spearman" # "pearson" # 

if(flag_Dataset){

# Strand Parameters
#strandList = c("All", "F", "R")
strandList = c("All")

# Strand Loop
for (str in 1:length(strandList)) {

  # Dataset Parameters
  strand = strandList[str]
  inList = c(
paste("DNase_H1hesc_6_",strand,sep=""),
paste("DNase_HeLaS3_6_",strand,sep=""),
paste("DNase_HepG2_6_",strand,sep=""),
paste("DNase_Huvec_6_",strand,sep=""),
paste("DNase_K562_6_",strand,sep=""),
paste("DNase_Mcf7_6_",strand,sep=""),
paste("DNase_NakedK562_6_",strand,sep=""),
paste("DNase_NakedMcf7_6_",strand,sep=""),
paste("DNaseUW_H7hesc_6_",strand,sep=""),
paste("DNaseUW_HepG2_6_",strand,sep=""),
paste("DNase_UW_HepG2_HS_6_",strand,sep=""),
paste("DNaseUW_Huvec_6_",strand,sep=""),
paste("DNaseUW_IMR90_6_",strand,sep=""),
paste("DNaseUW_K562_6_",strand,sep=""),
paste("DNase_UW_K562_HS_6_",strand,sep=""),
paste("DNase_UW_K562_RAW_6_",strand,sep=""),
paste("DNaseUW_LnCaP_6_",strand,sep=""),
paste("DNaseUW_m3134_with_DEX_6_",strand,sep="")
)
  labelList = c("DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","DU_Mcf7",
         "DU_NakedK562","DU_NakedMcf7","UW_H7hesc","UW_HepG2","UW_HepG2_HS","UW_Huvec","UW_IMR90","UW_K562","UW_K562_HS","UW_K562_RAW","UW_LnCaP","UW_m3134")
  outLoc = paste(loc,'CorrelationGraphs','Dataset',sep='/')

  # Initializing correlation table
  toWriteSam = c('',labelList)
  toWriteArt = c('',labelList)

  # Correlating sets two-by-two
  for (s1 in 1:length(inList)) {

    # Reading input
    name1 = inList[s1]
    data1 = read.table(paste(inLoc,'/HS/',name1,'.txt',sep=''), sep='\t', header=T)
    data1 = data1[order(data1[,1]),]
    v1_1 = standardize(data1[,2])
    v1_2 = standardize(data1[,3])

    # Writing correlation heading
    toWriteSam = c(toWriteSam,labelList[s1])
    toWriteArt = c(toWriteArt,labelList[s1])

    for (s2 in 1:length(inList)) {
    
      # Reading input
      name2 = inList[s2]
      data2 = read.table(paste(inLoc,'/HS/',name2,'.txt',sep=''), sep='\t', header=T)
      data2 = data2[order(data2[,1]),]
      v2_1 = standardize(data2[,2])
      v2_2 = standardize(data2[,3])

      # Creating data table
      data.sampled = data.frame(X=v1_1, Y=v2_1)
      data.artificial = data.frame(X=v1_2, Y=v2_2)

      # Correlation graphs
      corr_sam = createCorrelation(data.sampled[,1], data.sampled[,2], Y~X, data.sampled,
                                paste(outLoc,'_',corrtype,'/Sampled_',strand,'/',name1,'+',name2,'.eps',sep=''),
                                   name1, name2, corrtype)
      #corr_art = createCorrelation(data.artificial[,1], data.artificial[,2], Y~X, data.artificial,
      #                          paste(outLoc,'_',corrtype,'/Artificial_',strand,'/',name1,'+',name2,'.eps',sep=''), 
      #                             name1, name2, corrtype)
    
      # Writing correlations
      toWriteSam = c(toWriteSam,corr_sam)
      #toWriteArt = c(toWriteArt,corr_art)

    }
  }

  ## Initializing correlation table
  #toWriteSam = c('',labelList[2:length(labelList)])
  #toWriteArt = c('',labelList[2:length(labelList)])

  ## Correlating sets two-by-two
  #for (s1 in 1:(length(inList)-1)) {

  #  # Reading input
  #  name1 = inList[s1]
  #  data1 = read.table(paste(inLoc,'/HS/',name1,'.txt',sep=''), sep='\t', header=T)
  #  data1 = data1[order(data1[,1]),]
  #  v1_1 = standardize(data1[,2])
  #  v1_2 = standardize(data1[,3])

  #  # Writing correlation heading
  #  toWriteSam = c(toWriteSam,labelList[s1])
  #  toWriteArt = c(toWriteArt,labelList[s1])
  #  for (i in 1:s1){
  #    toWriteSam = c(toWriteSam,'NA')
  #    toWriteArt = c(toWriteArt,'NA')
  #  }
  #  toWriteSam = toWriteSam[1:(length(toWriteSam)-1)]
  #  toWriteArt = toWriteArt[1:(length(toWriteArt)-1)]

  #  for (s2 in (s1+1):length(inList)) {
    
  #    # Reading input
  #    name2 = inList[s2]
  #    data2 = read.table(paste(inLoc,'/HS/',name2,'.txt',sep=''), sep='\t', header=T)
  #    data2 = data2[order(data2[,1]),]
  #    v2_1 = standardize(data2[,2])
  #    v2_2 = standardize(data2[,3])

  #    # Creating data table
  #    data.sampled = data.frame(X=v1_1, Y=v2_1)
  #    data.artificial = data.frame(X=v1_2, Y=v2_2)

  #    # Correlation graphs
  #    corr_sam = createCorrelation(data.sampled[,1], data.sampled[,2], Y~X, data.sampled,
  #                                 paste(outLoc,'/Sampled_',strand,'/',name1,'+',name2,'.eps',sep=''), 
  #                                 name1, name2, "pearson")
  #    corr_art = createCorrelation(data.artificial[,1], data.artificial[,2], Y~X, data.artificial,
  #                                 paste(outLoc,'/Artificial_',strand,'/',name1,'+',name2,'.eps',sep=''),
  #                                 name1, name2, "pearson")
    
  #    # Writing correlations
  #    toWriteSam = c(toWriteSam,corr_sam)
  #    toWriteArt = c(toWriteArt,corr_art)

  #  }
  #}

  # Writing correlation to file
  write(toWriteSam,file=paste(outLoc,'_',corrtype,'/Sampled_',strand,'_corr.txt',sep=''),ncolumns=length(inList)+1,append=FALSE,sep='\t')
  #write(toWriteArt,file=paste(outLoc,'_',corrtype,'/Artificial_',strand,'_corr.txt',sep=''),ncolumns=length(inList)+1,append=FALSE,sep='\t')

}
}

###########################################################################################################################################
# Forward vs Reverse
###########################################################################################################################################
# Correlation between the bias in the forward vs in the reverse strand. 

if(flag_Forward_Reverse){

# Type Parameters
typeList = c("All", "HS")

# Type Loop
for (t in 1:length(typeList)) {

  # Dataset Parameters
  type = typeList[t]
  inList = c(
"DNase_H1hesc_6_",
"DNase_HeLaS3_6_",
"DNase_HepG2_6_",
"DNase_Huvec_6_",
"DNase_K562_6_",
"DNase_Mcf7_6_",
"DNaseUW_HepG2_6_",
"DNaseUW_Huvec_6_",
"DNaseUW_IMR90_6_",
"DNaseUW_K562_6_",
"DNaseUW_LnCaP_6_",
"DNaseUW_m3134_with_DEX_6_",
"DNaseUW_m3134_wo_DEX_6_"
)
  labelList = c("DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","DU_Mcf7","UW_HepG2","UW_Huvec","UW_IMR90","UW_K562","UW_LnCaP","UW_m3134(with)","UW_m3134(wo)")
  outLoc = paste(loc,'CorrelationGraphs','Forward_Reverse',sep='/')

  # Initializing correlation table
  toWriteSam = c()
  toWriteArt = c()

  for (i in 1:length(inList)) {

    # Reading input
    dataF = read.table(paste(inLoc,'/',type,'/',inList[i],'F.txt',sep=''), sep='\t', header=T)
    dataF = dataF[order(dataF[,1]),]
    rownames(dataF) <- dataF[,1]
    dataR = read.table(paste(inLoc,'/',type,'/',inList[i],'R.txt',sep=''), sep='\t', header=T)
    dataR = dataR[order(dataR[,1]),]
    rownames(dataR) <- dataR[,1]

    # Fetching only the top k-mers
    top = 2048 # 205 410 819 1229 2048
    dataF = dataF[order(dataF[,2]),]
    dataF = dataF[1:top,]
    dataR = dataR[dataF[,1],]
    dataF = dataF[order(dataF[,1]),]
    dataR = dataR[order(dataR[,1]),]

    # Creating data table
    vf_1 = standardize(dataF[,2])
    vf_2 = standardize(dataF[,3])
    vr_1 = standardize(dataR[,2])
    vr_2 = standardize(dataR[,3])
    data.sampled = data.frame(X=vf_1, Y=vr_1)
    data.artificial = data.frame(X=vf_2, Y=vr_2)

    # Correlation graphs
    corr_sam = createCorrelation(data.sampled[,1], data.sampled[,2], Y~X, data.sampled,
                                 paste(outLoc,'/Sampled_',type,'/',labelList[i],'.eps',sep=''), 
                                 'Forward Bias', 'Reverse Bias',"pearson")
    corr_art = createCorrelation(data.artificial[,1], data.artificial[,2], Y~X, data.artificial,
                                 paste(outLoc,'/Artificial_',type,'/',labelList[i],'.eps',sep=''),
                                 'Forward Bias', 'Reverse Bias',"pearson")
    
    # Writing correlations
    toWriteSam = c(toWriteSam,labelList[i],corr_sam)
    toWriteArt = c(toWriteArt,labelList[i],corr_art)

  }

  # Writing correlation to file
  write(toWriteSam,file=paste(outLoc,'/Sampled_',type,'_corr.txt',sep=''),ncolumns=2,append=FALSE,sep='\t')
  write(toWriteArt,file=paste(outLoc,'/Artificial_',type,'_corr.txt',sep=''),ncolumns=2,append=FALSE,sep='\t')

}

}

###########################################################################################################################################
# Sampled vs Artificial
###########################################################################################################################################
# Correlation between the sampled and artificial bias.

if(flag_Sampled_Artificial){

# Type Parameterscorrtype
typeList = c("All", "HS")

# Type Loop
for (t in 1:length(typeList)) {

  type = typeList[t]

  # Strand Parameters
  strandList = c("All","F","R")
  
  # Strand Loop
  for (str in 1:length(strandList)) {

    strand = strandList[str]

    # Initializing correlation table
    toWrite = c()

    # Input Parameters
    inList = c(
paste("DNase_H1hesc_6_",strand,sep=""),
paste("DNase_HeLaS3_6_",strand,sep=""),
paste("DNase_HepG2_6_",strand,sep=""),
paste("DNase_Huvec_6_",strand,sep=""),
paste("DNase_K562_6_",strand,sep=""),
paste("DNase_Mcf7_6_",strand,sep=""),
paste("DNaseUW_HepG2_6_",strand,sep=""),
paste("DNaseUW_Huvec_6_",strand,sep=""),
paste("DNaseUW_IMR90_6_",strand,sep=""),
paste("DNaseUW_K562_6_",strand,sep=""),
paste("DNaseUW_LnCaP_6_",strand,sep=""),
paste("DNaseUW_m3134_with_DEX_6_",strand,sep=""),
paste("DNaseUW_m3134_wo_DEX_6_",strand,sep="")
)
    labelList = c("DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","DU_Mcf7","UW_HepG2","UW_Huvec","UW_IMR90","UW_K562","UW_LnCaP","UW_m3134(with)","UW_m3134(wo)")
    outLoc = paste(loc,'CorrelationGraphs','Sampled_Artificial',sep='/')

    # Input Loop
    for (i in 1:length(inList)) {

      # Reading input
      data = read.table(paste(inLoc,'/',type,'/',inList[i],'.txt',sep=''), sep='\t', header=T)
      data = data[order(data[,1]),]

      # Creating data table
      v1 = standardize(data[,2])
      v2 = standardize(data[,3])
      dataFr = data.frame(X=v1, Y=v2)

      # Correlation graphs
      curr_corr = createCorrelation(dataFr[,1], dataFr[,2], Y~X, dataFr,
                                   paste(outLoc,'/',type,'_',strand,'/',labelList[i],'.eps',sep=''), 
                                   'Sampled Bias', 'Artificial Bias',"pearson")
    
      # Writing correlations
      toWrite = c(toWrite,labelList[i],curr_corr)

    }

    # Writing correlation to file
    write(toWrite,file=paste(outLoc,'/',type,'_',strand,'_corr.txt',sep=''),ncolumns=2,append=FALSE,sep='\t')

  }

}

}


###########################################################################################################################################
# Correlation between sequencing depth and the dataset correlation
###########################################################################################################################################
# The correlation between datasets (must be run first, check if tabs are ok) is tested
# against the sequencing depth (total number of reads in each bam file).

if(flag_SeqDepth_DatasetCorr){

# Sequencing depth list
datasetNames = c("DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","DU_Mcf7","UW_HepG2",               
                 "UW_Huvec","UW_IMR90","UW_K562","UW_LnCaP","UW_m3134(with)","UW_m3134(wo)")
depthList = c(110303078,54267867,50838536,31848532,365820647,89113893,168883956,
              429088276,138604440,179970820,163625945,127594903,130489281)

# Input Parameters
inputLocation = paste(loc,'CorrelationGraphs','Dataset',sep='/')
inList =  c("Artificial_All", "Artificial_F", "Artificial_R", "Sampled_All", "Sampled_F", "Sampled_R")

# Output Parameters
outLoc = paste(loc,'CorrelationGraphs','SeqDepth_DatasetCorr',sep='/')
toWrite = c()

# Input Loop
for (i in 1:length(inList)){

  # Reading input table
  inName = inList[i]
  inFileName = paste(inputLocation,'/',inName,'_corr.txt',sep='')
  data = read.table(inFileName, sep='\t', header=T)
  row.names(data) <- datasetNames[1:length(datasetNames)]
  data = data[,2:ncol(data)]
  colnames(data) <- datasetNames[1:length(datasetNames)]

  # Creating input vectors
  corrVec = c()
  depthVec = c()
  for (s1 in 1:(length(depthList)-1)){
    name1 = datasetNames[s1]
    for (s2 in (s1+1):length(depthList)){
      name2 = datasetNames[s2]
      depthVec = c(depthVec,max(depthList[s1],depthList[s2])/min(depthList[s1],depthList[s2]))
      corrVec = c(corrVec,as.numeric(as.character(data[name1,name2])))
    }
  }

  # Evaluating correlation
  currData = data.frame(X=corrVec, Y=depthVec)
  curr_corr = createCorrelation(corrVec, depthVec, Y~X, currData,
                               paste(outLoc,'/',inName,'.eps',sep=''), 
                               'Correlation between Datasets', 'Delta Sequencing Depth',"pearson")
  toWrite = c(toWrite,inName,curr_corr)

}

# Writing correlation to file
write(toWrite,file=paste(outLoc,'corr.txt',sep='/'),ncolumns=2,append=FALSE,sep='\t')

}


