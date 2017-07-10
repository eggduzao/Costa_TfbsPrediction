
# Import
import os
import sys
from glob import glob
 
# Cell Loop
#cellList = ["H1hesc", "K562", "GM12878"]
cellList = ["GM12878"]
for cell in cellList:
    
  # TF Loop
  tfFileList = glob("/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/fdr_4_inside_HS/*.bed")
  for tfFileName in tfFileList:

    # Parameters
    mpbsName = tfFileName.split("/")[-1].split(".")[0]
    mpbsFileName = tfFileName
    priorFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/DU_"+cell+"/Centipede_batch/"+mpbsName+"_PRIOR.txt"
    dnaseFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/DU_"+cell+"/Centipede_batch/"+mpbsName+"_DNASE.txt"
    outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/Centipede_batch/"
    outputFileName = outLoc+mpbsName+".bed"
    os.system("mkdir -p "+outLoc)

    # Execution
    myL = "_".join([cell,mpbsName,"ACE"])
    clusterCommand = "bsub "
    clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    clusterCommand += "-W 1:00 -M 8192 -S 100 -P izkf -R \"select[hpcwork]\" ./applyCentipede_pipeline.zsh "
    clusterCommand += mpbsName+" "+mpbsFileName+" "+priorFileName+" "+dnaseFileName+" "+outputFileName
    os.system(clusterCommand)
# 


