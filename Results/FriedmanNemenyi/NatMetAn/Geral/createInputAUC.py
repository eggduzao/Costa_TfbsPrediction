
# Import
import os
import sys
from glob import glob

# Input
rocLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM2/"
outLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/FriedmanNemenyi/NatMetAn/Geral/Input/"

# Parameters
nonFactor = ["P300","TBP","ER_0min","ER_2min","ER_5min","ER_10min","P53","P53_K562","AR_1nmR","AR_10nmR","GR_woDEX"]
aucSelection = {
"PWM": "PWM", 
"Boyle_DU": "Boyle",
#"Centipede_80": "Centipede(80)",
#"Centipede_85": "Centipede(85)",  
"Centipede_90": "Centipede(90)",
#"Centipede_95": "Centipede(95)", 
#"Centipede_99": "Centipede(99)",
#"Cuellar_80": "Cuellar(80)",
#"Cuellar_85": "Cuellar(85)",  
"Cuellar_90": "Cuellar(90)", 
#"Cuellar_95": "Cuellar(95)",
#"Cuellar_99": "Cuellar(99)", 
"Dnase2Tf_DU": "DNase2TF(TC)",
#"Dnase2Tf_rank": "DNase2TF(rank)",
#"FLR_80": "FLR(80)",
#"FLR_85": "FLR(85)",  
"FLR_90": "FLR(90)", 
#"FLR_95": "FLR(95)", 
#"FLR_99": "FLR(99)", 
"FS_DU": "FS",
"HINT-BC_D_DU": "HINTBC", 
"HINT-BCN_D_DU": "HINTBCN", 
"HINT_D_DU": "HINT",
"Neph_DU": "Neph", 
#"PIQ_80": "PIQ(80)",
#"PIQ_85": "PIQ(85)",  
"PIQ_90": "PIQ(90)", 
#"PIQ_95": "PIQ(95)", 
#"PIQ_99": "PIQ(99)",
"TC_DU": "TC", 
"Wellington_DU": "Wellington(TC)",
#"Wellington_rank": "Wellington(rank)",
#"Protection_DU": "Protection",
"BinDNase_80": "BinDNase(80)",
#"BinDNase_85": "BinDNase(85)",
#"BinDNase_90": "BinDNase(90)",
#"BinDNase_95": "BinDNase(95)",
#"BinDNase_99": "BinDNase(99)",
#"BinDNase_rank": "BinDNase(rank)"
}
fileTypeList = ["AUC_100","AUC_10","AUC_1","AUPR_100"]
headerList = sorted(aucSelection.keys())

# Cell Loop
resDict = dict()
tfCount = 0
cellList = ["H1hesc","K562"]
for cell in cellList:

  resDict[cell] = dict()

  # TF Loop
  tfList = sorted(["_".join(e.split("/")[-1].split("_")[:-1]) for e in glob(rocLoc+cell+"/*_stats.txt") if "_".join(e.split("/")[-1].split("_")[:-1]) not in nonFactor])
  for tf in tfList:

    # Fetching all AUC
    aucFileName = rocLoc+cell+"/"+tf+"_stats.txt"
    aucFile = open(aucFileName,"r")
    curr_aucDict = dict()
    for line in aucFile:
      ll = line.strip().split()
      curr_aucDict[ll[0]] = [ll[1],ll[2],ll[5]]
    aucFile.close()

    # Fetching all AUPR
    auprFileName = rocLoc+cell+"/"+tf+"_statsPR.txt"
    auprFile = open(auprFileName,"r")
    curr_auprDict = dict()
    for line in auprFile:
      ll = line.strip().split()
      curr_auprDict[ll[0]] = [ll[1]]
    auprFile.close()

    # Selecting AUC + AUPR vector
    resDict[cell][tf] = []
    for k in headerList:
      try: resDict[cell][tf].append(curr_aucDict[k]+curr_auprDict[k])
      except Exception: resDict[cell][tf].append(curr_aucDict["TC_DU"]+curr_auprDict["TC_DU"])
    tfCount += 1

# Writing file
for i in range(0,len(fileTypeList)):

  # Writing AUC
  outputFileName = outLoc+fileTypeList[i]
  outputFile = open(outputFileName+".txt","w")
  outputFile.write("\t".join([str(tfCount),str(len(headerList)),"asc"])+"\n")
  for cell in cellList:
    tfList = sorted(["_".join(e.split("/")[-1].split("_")[:-1]) for e in glob(rocLoc+cell+"/*_stats.txt") if "_".join(e.split("/")[-1].split("_")[:-1]) not in nonFactor])
    for tf in tfList: outputFile.write("\t".join([e[i] for e in resDict[cell][tf]])+"\n")
  outputFile.close()

# Writing header
headerFile = open(outLoc+"header.txt","w")
for h in headerList: headerFile.write(aucSelection[h]+"\n")
headerFile.close()


