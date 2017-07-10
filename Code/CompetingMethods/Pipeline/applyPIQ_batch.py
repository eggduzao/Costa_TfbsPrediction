
# Import
import os
import sys
from glob import glob

# Fetch TF information
tfFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/PWM/PIQ/Expression_Analysis/PIQjaspar_translate.txt"
tfFile = open(tfFileName,"r")
tfDict = dict()
for line in tfFile:
  ll = line.strip().split("\t")
  tfDict[ll[0]] = ll[1]
tfFile.close()

tfDict = {"107": "PRRX2"}

# Cell Loop
#cellList = ["H1hesc","K562","GM12878"]
cellList = ["GM12878"]
for cell in cellList:
    
  # TF Loop
  keyList = sorted(tfDict.keys())
  for k in keyList:
  
    # Parameters
    pwnN = k
    factor = tfDict[k]
    pwmLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/PIQ_batch/"
    cutFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/DU_"+cell+"/PIQ/DNase.RData"
    outputLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/PIQ_batch/"
    os.system("mkdir -p "+outputLocation)

    # Execution
    myL = "_".join([cell,factor,"APQ"])
    clusterCommand = "bsub "
    clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    clusterCommand += "-W 100:00 -M 24000 -S 100 -P izkf -R \"select[hpcwork]\" ./applyPIQ_pipeline.zsh "
    clusterCommand += pwnN+" "+factor+" "+pwmLoc+" "+cutFileName+" "+outputLocation
    os.system(clusterCommand) 
# 


