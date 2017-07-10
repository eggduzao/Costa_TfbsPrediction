
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
tfbsFileName = hl+"Data/TFBS/TFBS_table.txt"
outputFileName = wl+"StatisticalTests/MultivarTable/table5/"+lab+"_"+cell+".txt"
tempLoc = wl+"StatisticalTests/MultivarTable/table5/"+lab+"_"+cell+"/"
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
# ChIP and PWM information
###################################################################################################

# Create dictionary
tfbsDict = dict()
tfbsFile = open(tfbsFileName,"r")
tfbsFile.readline()
for line in tfbsFile:
  ll = line.strip().split("\t")
  if(cell != ll[0]): continue
  tfbsDict[ll[1]] = ll[2:]
tfbsFile.close()

# Writing to factor dict
infoDict = dict()
for factor in factorList:
  try: infoDict[factor] = tfbsDict[factor]
  except Exception:
    print cell, factor
    infoDict[factor] = ["NA" for e in range(0,4)]

###################################################################################################
# Writing results
###################################################################################################

# Writing results
outputFile = open(outputFileName,"w")
outputFile.write("\t".join(["CHIPSEQ_LAB","CHIPSEQ_ID","PWM_REPOSITORY","PWM_ID"])+"\n")
for factor in factorList: outputFile.write("\t".join(infoDict[factor])+"\n")
outputFile.close()

# Termination
os.system("rm -rf "+tempLoc)


