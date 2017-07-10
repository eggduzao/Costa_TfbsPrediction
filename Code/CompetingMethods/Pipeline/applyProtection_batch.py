
# Import
import os
import sys
from glob import glob

# Cell Loop
cellList = ["H1hesc", "K562", "GM12878"]
for cell in cellList:
    
  # TF Loop
  tfList = glob("/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/fdr_4_inside_HS/*.bed")
  for tfFile in tfList:
  
    # Parameters
    tfName = tfFile.split("/")[-1].split(".")[0]
    mpbsFileName = tfFile
    signalFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/DU_"+cell+"/DNase.bw"
    outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/Protection_batch/"
    outputFileName = outputLocation+tfName+".bed"
    os.system("mkdir -p "+outputLocation)

    # Execution
    myL = "_".join([cell,tfName,"FSB"])
    clusterCommand = "bsub "
    clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    clusterCommand += "-W 100:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./applyProtection_pipeline.zsh "
    clusterCommand += mpbsFileName+" "+signalFileName+" "+outputFileName
    os.system(clusterCommand) 
# 


