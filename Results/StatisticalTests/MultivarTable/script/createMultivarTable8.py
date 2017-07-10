
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
rocLoc = hl+"Results_DNase/Footprints/ROC/"
outputFileName = wl+"StatisticalTests/MultivarTable/table8/"+lab+"_"+cell+".txt"
tempLoc = wl+"StatisticalTests/MultivarTable/table8/"+lab+"_"+cell+"/"
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
aucDict1 = dict()
auprDictAll = dict()
metVec = ["TCH_DU","HINT_H13_DU","HINT_D13_DU","HINT-BC_D13_DU"]
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
    aucDict1[factor] = ["NA" for e in metVec]
    auprDictAll[factor] = ["NA" for e in metVec]
    continue
  aucFile.readline()
  auprFile.readline()
  aucDict = dict()
  for line in aucFile:
    ll = line.strip().split("\t")
    aucDict[ll[0]] = [ll[1],ll[2],ll[5]]
  auprDict = dict()
  for line in auprFile:
    ll = line.strip().split("\t")
    auprDict[ll[0]] = [ll[1]]
  aucVec = []; auc10Vec = []; auc1Vec = []; auprVec = []
  for e in metVec:
    try: aucVec.append(str(aucDict[e][0]))
    except Exception: aucVec.append("NA")
    try: auc10Vec.append(str(aucDict[e][1]))
    except Exception: auc10Vec.append("NA")
    try: auc1Vec.append(str(aucDict[e][2]))
    except Exception: auc1Vec.append("NA")
    try: auprVec.append(str(auprDict[e][0]))
    except Exception: auprVec.append("NA")
  aucDictAll[factor] = aucVec
  aucDict10[factor] = auc10Vec
  aucDict1[factor] = auc1Vec
  auprDictAll[factor] = auprVec
  aucFile.close()
  auprFile.close()

###################################################################################################
# Writing results
###################################################################################################

# Writing results
outputFile = open(outputFileName,"w")
outputFile.write("\t".join([b+a for a in ["_AUC_ALL","_AUC_10","_AUC_1","_AUPR_ALL"] for b in ["TCH_DU","HINT_H13_DU","HINT_D13_DU","HINT_BC_D13_DU"]])+"\n")

for factor in factorList:
  outputFile.write("\t".join([str(e) for e in [
                   aucDictAll[factor][0], aucDictAll[factor][1], aucDictAll[factor][2], aucDictAll[factor][3],
                   aucDict10[factor][0], aucDict10[factor][1], aucDict10[factor][2], aucDict10[factor][3],
                   aucDict1[factor][0], aucDict1[factor][1], aucDict1[factor][2], aucDict1[factor][3],
                   auprDictAll[factor][0], auprDictAll[factor][1], auprDictAll[factor][2], auprDictAll[factor][3]
                   ]])+"\n")
outputFile.close()

# Termination
os.system("rm -rf "+tempLoc)


