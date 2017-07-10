
###################################################################################################
# IMPORT
###################################################################################################

import os
import sys
from glob import glob

###################################################################################################
# INPUT
###################################################################################################

# Input
lab = sys.argv[1]
cell = sys.argv[2]

# Input files
wl = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/"
pwml = "/hpcwork/izkf/projects/TfbsPrediction/Data/PWM/All/"
nrl = "/hpcwork/izkf/projects/TfbsPrediction/Data/PWM/NuclearReceptors/"
pearsonLoc = wl+"StatisticalTests/TFSpecificBias/Pearson/"
sinFileName = wl+"StatisticalTests/MultivarTable/script/pwm_chip.txt"
outputFileName = wl+"StatisticalTests/MultivarTable/table7/"+lab+"_"+cell+".txt"

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

# Reading synonimous
sinFile = open(sinFileName,"r")
sinDict = dict()
for f in factorList: sinDict[f] = [f]
for line in sinFile:
  ll = line.strip().split("\t")
  try: sinDict[ll[0]].append(ll[1])
  except Exception: sinDict[ll[0]] = [ll[1]]
sinFile.close()

###################################################################################################
# Fetching CG
###################################################################################################

pwmcgres = dict()
nrList = [nrl+"AR_MA0007_2.pwm",nrl+"ESR1_MA0112_2.pwm",nrl+"NR3C1_MA0113_2.pwm",nrl+"TP53_MA0106_2.pwm"]
for pwmFileName in glob(pwml+"*.pfm")+nrList:

  pwmName = pwmFileName.strip().split("/")[-1].split("_")[0]
  
  pwmFile = open(pwmFileName,"r")
  sumVec = []
  for line in pwmFile:
    ll = line.strip().split(" ")
    sumVec.append(sum([float(e) for e in ll]))
  pwmFile.close()

  v = str((sumVec[1]+sumVec[2])/(sum(sumVec)))
  pwmcgres[pwmName] = v
  try:
    for e in sinDict[pwmName]: pwmcgres[e] = v
  except Exception: pass

###################################################################################################
# Writing results
###################################################################################################

outFile = open(outputFileName,"w")
outFile.write("MOTIF_CG_CONTENT\n")
for factor in factorList:
  try: outFile.write(pwmcgres[factor]+"\n")
  except Exception: outFile.write("NA\n")
outFile.close()


