
###################################################################################################
# IMPORT
###################################################################################################

import os
import sys

###################################################################################################
# INPUT
###################################################################################################

# Input
lab = sys.argv[1]
cell = sys.argv[2]

# Input files
wl = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/"
bl = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/Boxplot/"
pearsonLoc = wl+"StatisticalTests/TFSpecificBias/Pearson/"
outputFileName = wl+"StatisticalTests/MultivarTable/table6/"+lab+"_"+cell+".txt"

binDNaseNonFactor = ["H1hesc_P300", "H1hesc_POU5F1", "H1hesc_REST", "H1hesc_RFX5", 
                     "H1hesc_SP1", "H1hesc_SP2", "H1hesc_SRF", "H1hesc_TCF12", "H1hesc_ZNF143", 
                     "K562_ARID3A", "K562_CTCF_Y", "K562_IRF1", "K562_MEF2A", "K562_PU1",
                     "K562_REST", "K562_RFX5", "K562_SP1", "K562_SP2", "K562_STAT2", "K562_ZNF263"]

###################################################################################################
# Pearson Bias estimate
###################################################################################################

# Fetching Pearson bias estimate
pearsonDict = dict()
pearsonFileName = pearsonLoc+lab+"_"+cell+"/pearson.txt"
pearsonFile = open(pearsonFileName,"r")
pearsonFile.readline()
for line in pearsonFile:
  ll = line.strip().split("\t") 
  pearsonDict[ll[0]] = ll[1]
pearsonFile.close()

# Factor List
factorList = sorted(pearsonDict.keys())
  
###################################################################################################
# AUC / AUPR
###################################################################################################

# Fetching auc/aupr
aucDict = dict()
typeList = ["AUC_1", "AUC_10", "AUC_100", "AUPR_100"]
for t in typeList:
  
  inFile = open(bl+t+".txt","r")
  methodVec = inFile.readline().strip().split("\t")
  for line in inFile:
    ll = line.strip().split("\t")
    if(ll[0] != cell or lab == "UW"): continue
    if(ll[0]+"_"+ll[1] in binDNaseNonFactor): ll = ll[:2]+["NA","NA","NA","NA","NA","NA"]+ll[8:]
    for i in range(2,len(ll)): aucDict["_".join([ll[0],ll[1],t,methodVec[i]])] = ll[i]
  inFile.close()

###################################################################################################
# Writing results
###################################################################################################

# Header
header = []
for t in typeList:
  if(t in ["AUC_10","AUC_100","AUPR_100"]):
    methodVec = ["BinDNase_80","BinDNase_85","BinDNase_90","BinDNase_95","BinDNase_99",
                 "BinDNase_rank","Centipede_80","Centipede_85",
                 "Centipede_95","Centipede_99","Cuellar_80","Cuellar_85",
                 "Cuellar_95","Cuellar_99","DNase2TF_rank","FLR_80","FLR_85","FLR_95","FLR_99",
                 "PIQ_80","PIQ_85","PIQ_95","PIQ_99","Protection","Wellington_rank"]
  else: 
    methodVec = ["BinDNase_80","BinDNase_85","BinDNase_90","BinDNase_95","BinDNase_99",
                 "BinDNase_rank","Boyle","Centipede_80","Centipede_85","Centipede_90",
                 "Centipede_95","Centipede_99","Cuellar_80","Cuellar_85","Cuellar_90",
                 "Cuellar_95","Cuellar_99","DNase2TF_TC","DNase2TF_rank",
                 "FLR_80","FLR_85","FLR_90","FLR_95","FLR_99","FS","HINTBCN","HINTBC","HINT",
                 "Neph","PIQ_80","PIQ_85","PIQ_90","PIQ_95","PIQ_99","PWM","Protection","TC",
                 "Wellington_TC","Wellington_rank"]
  if(t == "AUC_100"): t = "AUC_ALL"
  elif(t == "AUPR_100"): t = "AUPR_ALL"
  for m in methodVec: header.append(m+"_"+t)

# Writing results
outputFile = open(outputFileName,"w")
outputFile.write("\t".join(header)+"\n")
for factor in factorList:
  vec = []
  for t in typeList:
    if(t in ["AUC_10","AUC_100","AUPR_100"]):
      methodVec = ["BinDNase_80","BinDNase_85","BinDNase_90","BinDNase_95","BinDNase_99",
                   "BinDNase_rank","Centipede_80","Centipede_85",
                   "Centipede_95","Centipede_99","Cuellar_80","Cuellar_85",
                   "Cuellar_95","Cuellar_99","DNase2TF_rank","FLR_80","FLR_85","FLR_95","FLR_99",
                   "PIQ_80","PIQ_85","PIQ_95","PIQ_99","Protection","Wellington_rank"]
    else: 
      methodVec = ["BinDNase_80","BinDNase_85","BinDNase_90","BinDNase_95","BinDNase_99",
                   "BinDNase_rank","Boyle","Centipede_80","Centipede_85","Centipede_90",
                   "Centipede_95","Centipede_99","Cuellar_80","Cuellar_85","Cuellar_90",
                   "Cuellar_95","Cuellar_99","DNase2TF_TC","DNase2TF_rank",
                   "FLR_80","FLR_85","FLR_90","FLR_95","FLR_99","FS","HINTBCN","HINTBC","HINT",
                   "Neph","PIQ_80","PIQ_85","PIQ_90","PIQ_95","PIQ_99","PWM","Protection","TC",
                   "Wellington_TC","Wellington_rank"]
    for m in methodVec:
      try: vec.append(aucDict["_".join([cell,factor,t,m])])
      except Exception: vec.append("NA")
  outputFile.write("\t".join(vec)+"\n")

outputFile.close()


