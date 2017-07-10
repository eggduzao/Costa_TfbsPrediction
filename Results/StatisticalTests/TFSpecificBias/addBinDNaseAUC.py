
import os
import sys

inLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson_AUC_Table/"
rocLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM2/"
inFileList = ["DU_H1hesc_10AUC", "DU_K562_10AUC"]

for inName in inFileList:

  cell = inName.split("_")[1]
  inFileName = inLoc+inName+".txt"
  outFileName = inLoc+inName+"_BD.txt"
  inFile = open(inFileName,"r")
  outFile = open(outFileName,"w")
  outFile.write("\t".join(inFile.readline().strip().split("\t")+["BINDNASE"])+"\n")
  for line in inFile:
    ll = line.strip().split("\t")
    rocFile = open(rocLoc+cell+"/"+ll[0]+"_stats.txt","r")
    rocFile.readline()
    myAUC = "NA"
    for jine in rocFile:
      jj = jine.strip().split("\t")
      if(jj[0] == "BinDNase_85"):
        myAUC = jj[2]
        break
    outFile.write("\t".join(ll+[myAUC])+"\n")
    rocFile.close()
  inFile.close()
  outFile.close()


