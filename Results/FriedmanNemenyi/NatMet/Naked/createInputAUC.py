
# Import
import os
import sys
from glob import glob

# Input
rocLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/"
outLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/FriedmanNemenyi/NatMet/Naked/Input/"

# Parameters
#nonFactor = ["BRCA1","FOSL1","P300","TBP","ARID3A","THAP1","ZBTB7A"]
nonFactor = ["P300","TBP","ER_0min","ER_2min","ER_5min","ER_10min","P53","P53_K562","AR_1nmR","AR_10nmR","GR_woDEX"]
tfCount = 0

# Lab Parameters
#labList = ["DU","UW"]
labList = ["UW"]
resDict = dict()

# Iterating on lab
for lab in labList:

  resDict[lab] = dict()

  # Cell Parameters
  if(lab == "DU"): cellList = ["H1hesc","HeLaS3","HepG2","Huvec","K562","Mcf7","Saos2"]
  elif(lab == "UW"): cellList = ["HepG2","Huvec","K562","LnCaP","m3134_with_DEX"]

  # Iterating on cell type
  for cell in cellList:

    resDict[lab][cell] = dict()

    # Creating cell-specific auc selection
    if(lab == "DU"): aucSelection = ["HMMD_DU","HMMD_DU_BIAS","HMMD_DU_NAKED"]
    if(lab == "UW"): aucSelection = ["HMMD_UW","HMMD_UW_BIAS","HMMD_UW_NAKED"]

    # Iterating on each TF
    tfList = sorted(["_".join(e.split("/")[-1].split("_")[:-1]) for e in glob(rocLoc+cell+"/*_stats.txt") if "_".join(e.split("/")[-1].split("_")[:-1]) not in nonFactor])
    for tf in tfList:

      # Fetching all AUC
      aucFileName = rocLoc+cell+"/"+tf+"_stats.txt"
      aucFile = open(aucFileName,"r")
      curr_aucDict = dict()
      for line in aucFile:
        ll = line.strip().split()
        vec = [ll[1],ll[2]]
        curr_aucDict[ll[0]] = vec # ORDER = AUC, AUC_10
      aucFile.close()

      # Selecting AUC vector
      resDict[lab][cell][tf] = [curr_aucDict[e] for e in aucSelection]

      tfCount += 1

# Writing file
headerList = ["HINT","HINT-BC","HINT-BCN"]
fileTypeList = ["ALLAUC","10AUC"]
for i in range(0,len(fileTypeList)):

  # Writing AUC
  outputFileName = outLoc+fileTypeList[i]
  outputFile = open(outputFileName+".txt","w")
  outputFile.write("\t".join([str(tfCount),str(len(headerList)),"asc"])+"\n")
  for lab in labList:
    if(lab == "DU"): cellList = ["H1hesc","HeLaS3","HepG2","Huvec","K562","Mcf7","Saos2"]
    elif(lab == "UW"): cellList = ["HepG2","Huvec","K562","LnCaP","m3134_with_DEX"]
    for cell in cellList:
      tfList = sorted(["_".join(e.split("/")[-1].split("_")[:-1]) for e in glob(rocLoc+cell+"/*_stats.txt") if "_".join(e.split("/")[-1].split("_")[:-1]) not in nonFactor])
      for tf in tfList: outputFile.write("\t".join([e[i] for e in resDict[lab][cell][tf]])+"\n")
  outputFile.close()

# Writing header
headerFile = open(outLoc+"header.txt","w")
for h in headerList: headerFile.write(h+"\n")
headerFile.close()


