
loc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson/"

# Iterating on input files
inList = c( "DU_H1hesc", "DU_HeLaS3", "DU_HepG2", "DU_Huvec", "DU_K562", "DU_Mcf7", "DU_Saos2", 
"UW_denovo", "UW_H7hesc", "UW_HepG2", "UW_Huvec", "UW_K562", "UW_LnCaP", "UW_m3134_with_DEX", "UW_m3134_wo_DEX")
for (inName in inList) {

  toWrite = c("FACTOR","SPEARMAN","PVALUE")
  dataO = read.table(paste(loc,inName,"/signal_original.txt",sep=''), sep='\t', header=T)
  dataB = read.table(paste(loc,inName,"/signal_bias.txt",sep=''), sep='\t', header=T)
  cNames = colnames(dataO)

  for(i in 0:((ncol(dataO)/2)-1)){

    originalF = dataO[,(i*2)+1]
    originalR = dataO[,(i*2)+2]
    vecO = c(originalF,originalR)

    biasF = dataB[,(i*2)+1]
    biasR = dataB[,(i*2)+2]
    vecB = c(biasF,biasR)
    
    nameVec = strsplit(cNames[(i*2)+1],"_")[[1]]
    name = paste(nameVec[1:length(nameVec)-1],collapse="_")
    if(nameVec[1] == "UW"){title = paste("De novo motif #",nameVec[3],sep="")}
    else if(nameVec[1] == "ER" || nameVec[1] == "AR" || nameVec[1] == "GR"){
      title = paste(nameVec[1]," (",nameVec[2],")",sep="")
    }
    else{title = nameVec[1]}

    corrTest = cor.test(vecO, vecB, alternative = "two.sided", method = "spearman", conf.level = 0.95)
    pvalue = corrTest$p.value
    corr = corrTest$estimate

    toWrite = c(toWrite,name,corr,pvalue)

  }

  write(toWrite,file=paste(loc,inName,"/spearman.txt",sep=''),ncolumns=3,append=FALSE,sep='\t')

}

warnings()
