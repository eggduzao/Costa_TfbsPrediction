
# Import
import os
import sys
from glob import glob

# Input
rocLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_TC/"
outLoc = "/home/egg/Projects/TfbsPrediction/Results/FriedmanNemenyi/BiasCorrection/All/Input/"

# Parameters
#nonFactor = ["BRCA1","FOSL1","P300","TBP","ARID3A","THAP1","ZBTB7A"]
nonFactor = ["P300","TBP"]
tfCount = 0

# Lab Parameters
labList = ["DU","UW"]
resDict = dict()

# Iterating on lab
for lab in labList:

  resDict[lab] = dict()

  # Cell Parameters
  if(lab == "DU"): cellList = ["H1hesc","HeLaS3","HepG2","Huvec","K562"]
  elif(lab == "UW"): cellList = ["HepG2","Huvec","K562"]

  # Iterating on cell type
  for cell in cellList:

    resDict[lab][cell] = dict()

    # Creating cell-specific auc selection
    #if(lab == "DU"): aucSelection = ["TC_DU","FOS_DU","PWM","HMM_DU","HMM_DU_BIAS","HMMD_DU","HMMD_DU_BIAS"]
    #elif(lab == "UW"): aucSelection = ["TC_UW","FOS_UW","PWM","HMM_UW","HMM_UW_BIAS","HMMD_UW","HMMD_UW_BIAS"]
    if(lab == "DU"): aucSelection = ["TC_DU","FOS_DU","PWM","HMMD_DU","HMMD_DU_BIAS"]
    elif(lab == "UW"): aucSelection = ["TC_UW","FOS_UW","PWM","HMMD_UW","HMMD_UW_BIAS"]

    # Iterating on each TF
    tfList = sorted([e.split("/")[-1].split("_")[0] for e in glob(rocLoc+cell+"/*_stats.txt") if e.split("/")[-1].split("_")[0] not in nonFactor])
    for tf in tfList:

      # Fetching all AUC
      aucFileName = rocLoc+cell+"/"+tf+"_stats.txt"
      aucFile = open(aucFileName,"r")
      curr_aucDict = dict()
      for line in aucFile:
        ll = line.strip().split()
        if(lab == "DU"): vec = [ll[1],ll[2],ll[5],ll[8]]
        if(lab == "UW"): vec = [ll[1],ll[2],ll[11],ll[14]]
        curr_aucDict[ll[0]] = vec # ORDER = AUC, AUC_10, AUC_HMM, AUC_BIAS
      aucFile.close()

      # Selecting AUC vector
      resDict[lab][cell][tf] = [curr_aucDict[e] for e in aucSelection]

      tfCount += 1

# Writing file
#headerList = ["TC","FOS","PWM","HMM","HMM(BIAS)","HMMD","HMMD(BIAS)"]
headerList = ["TC","FOS","PWM","HINT","HINT(BC)"]
fileTypeList = ["ALLAUC","10AUC","HMMAUC","BIASAUC"]
for i in range(0,len(fileTypeList)):

  # Writing AUC
  outputFileName = outLoc+fileTypeList[i]
  outputFile = open(outputFileName+".txt","w")
  outputFile.write("\t".join([str(tfCount),str(len(headerList)),"asc"])+"\n")
  for lab in labList:
    if(lab == "DU"): cellList = ["H1hesc","HeLaS3","Huvec","K562"]
    elif(lab == "UW"): cellList = ["HepG2","Huvec","K562"]
    for cell in cellList:
      tfList = sorted([e.split("/")[-1].split("_")[0] for e in glob(rocLoc+cell+"/*_stats.txt") if e.split("/")[-1].split("_")[0] not in nonFactor])
      for tf in tfList: outputFile.write("\t".join([e[i] for e in resDict[lab][cell][tf]])+"\n")
  outputFile.close()

# Writing header
headerFile = open(outLoc+"header.txt","w")
for h in headerList: headerFile.write(h+"\n")
headerFile.close()


