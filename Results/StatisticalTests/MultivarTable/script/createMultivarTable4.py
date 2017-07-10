
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
hl = "/hpcwork/izkf/projects/TfbsPrediction/"
pearsonLoc = wl+"StatisticalTests/TFSpecificBias/Pearson/"
rocLoc = hl+"Results/Footprints/ROC_NM_ALLTC/"
outputFileName = wl+"StatisticalTests/MultivarTable/table4/"+lab+"_"+cell+".txt"
tempLoc = wl+"StatisticalTests/MultivarTable/table4/"+lab+"_"+cell+"/"
os.system("mkdir -p "+tempLoc)

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
aucDictAll = dict()
aucDict10 = dict()
auprDictAll = dict()
metVec = ["Boyle_"+lab,"Neph_"+lab,"Cuellar_"+lab,"Pique_"+lab,"DNase2TF","FLR","PIQ","Wellington"]
for factor in factorList:
  aucFileName = rocLoc+cell+"/"+factor+"_stats.txt"
  auprFileName = rocLoc+cell+"/"+factor+"_statsPR.txt"
  try:
    aucFile = open(aucFileName,"r")
    auprFile = open(auprFileName,"r")
    if(lab == "UW"): 1/0
  except Exception: 
    aucDictAll[factor] = ["NA" for e in metVec]
    aucDict10[factor] = ["NA" for e in metVec]
    auprDictAll[factor] = ["NA" for e in metVec]
    continue
  aucFile.readline()
  auprFile.readline()
  aucDict = dict()
  for line in aucFile:
    ll = line.strip().split("\t")
    aucDict[ll[0]] = [ll[1],ll[2]]
  auprDict = dict()
  for line in auprFile:
    ll = line.strip().split("\t")
    auprDict[ll[0]] = [ll[1]]
  aucVec = []; auc10Vec = []; auprVec = []
  for e in metVec:
    try: aucVec.append(str(aucDict[e][0]))
    except Exception: aucVec.append("NA")
    try: auc10Vec.append(str(aucDict[e][1]))
    except Exception: auc10Vec.append("NA")
    try: auprVec.append(str(auprDict[e][0]))
    except Exception: auprVec.append("NA")
  aucDictAll[factor] = aucVec
  aucDict10[factor] = auc10Vec
  auprDictAll[factor] = auprVec
  aucFile.close()
  auprFile.close()

###################################################################################################
# Writing results
###################################################################################################

# Writing results
outputFile = open(outputFileName,"w")
outputFile.write("\t".join(["Boyle_AUC_ALL","Neph_AUC_ALL","Cuellar_AUC_ALL","Centipede_AUC_ALL",
                            "DNase2TF_AUC_ALL","FLR_AUC_ALL","PIQ_AUC_ALL","Wellington_AUC_ALL",
                            "Boyle_AUC_10","Neph_AUC_10","Cuellar_AUC_10","Centipede_AUC_10",
                            "DNase2TF_AUC_10","FLR_AUC_10","PIQ_AUC_10","Wellington_AUC_10",
                            "Boyle_AUPR_ALL","Neph_AUPR_ALL","Cuellar_AUPR_ALL","Centipede_AUPR_ALL",
                            "DNase2TF_AUPR_ALL","FLR_AUPR_ALL","PIQ_AUPR_ALL","Wellington_AUPR_ALL"])+"\n")
for factor in factorList:
  outputFile.write("\t".join([str(e) for e in [
                   aucDictAll[factor][0], aucDictAll[factor][1], aucDictAll[factor][2], aucDictAll[factor][3],
                   aucDictAll[factor][4], aucDictAll[factor][5], aucDictAll[factor][6], aucDictAll[factor][7],
                   aucDict10[factor][0], aucDict10[factor][1], aucDict10[factor][2], aucDict10[factor][3],
                   aucDict10[factor][4], aucDict10[factor][5], aucDict10[factor][6], aucDict10[factor][7],
                   auprDictAll[factor][0], auprDictAll[factor][1], auprDictAll[factor][2], auprDictAll[factor][3],
                   auprDictAll[factor][4], auprDictAll[factor][5], auprDictAll[factor][6], auprDictAll[factor][7],
                   ]])+"\n")
outputFile.close()

# Termination
os.system("rm -rf "+tempLoc)


