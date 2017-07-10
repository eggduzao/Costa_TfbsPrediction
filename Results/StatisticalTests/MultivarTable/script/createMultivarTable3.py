
###################################################################################################
# IMPORT
###################################################################################################

import os
import sys
import math
sys.path.append("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code")
from util import *

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
auprLoc = hl+"Results/Footprints/ROC_NM/"
fpLoc = wl+"StatisticalTests/MultivarTable/predictions/"
outputFileName = wl+"StatisticalTests/MultivarTable/table3/"+lab+"_"+cell+".txt"
tempLoc = wl+"StatisticalTests/MultivarTable/table3/"+lab+"_"+cell+"/"
os.system("mkdir -p "+tempLoc)

# Parameters
peakExtHalf = 50
motifExt = 20

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
# aupr
###################################################################################################

# Fetching aupr HINT(BC)
auprDictAll = dict()
#auprDict10 = dict()
metVec = ["PWM","TC_"+lab,"FOS_"+lab,"HMMD_"+lab,"HMMD_"+lab+"_BIAS","HMMD_"+lab+"_NAKED"]
for factor in factorList:
  auprFileName = auprLoc+cell+"/"+factor+"_statsPR.txt"
  try: auprFile = open(auprFileName,"r")
  except Exception: 
    auprDictAll[factor] = ["NA" for e in metVec]
    #auprDict10[factor] = ["NA" for e in metVec]
    continue
  auprFile.readline()
  auprDict = dict()
  for line in auprFile:
    ll = line.strip().split("\t")
    #auprDict[ll[0]] = [ll[1],ll[2]]
    auprDict[ll[0]] = [ll[1]]
  auprDictAll[factor] = [auprDict[e][0] for e in metVec]
  #auprDict10[factor] = [auprDict[e][1] for e in metVec]
  auprFile.close()

###################################################################################################
# Writing results
###################################################################################################

# Writing results
outputFile = open(outputFileName,"w")
outputFile.write("\t".join(["PWM_AUPR_ALL", "TC_AUPR_ALL", "FOS_AUPR_ALL", 
                            "HINT_AUPR_ALL", "HINT_BC_AUPR_ALL", "HINT_BCN_AUPR_ALL"])+"\n")
for factor in factorList:
  outputFile.write("\t".join([str(e) for e in [
                   auprDictAll[factor][0], auprDictAll[factor][1], auprDictAll[factor][2], 
                   auprDictAll[factor][3], auprDictAll[factor][4], auprDictAll[factor][5]
                   ]])+"\n")
outputFile.close()

# Termination
os.system("rm -rf "+tempLoc)


