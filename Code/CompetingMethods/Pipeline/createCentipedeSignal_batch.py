
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
    consFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/PhastCons/PhastCons.bw"
    tssFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/DistanceTSS/distanceTSS.bw"
    dnaseTotExt = "200"
    dnasePosFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/DU_"+cell+"/DNasePos.bw"
    dnaseNegFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/DU_"+cell+"/DNaseNeg.bw"
    outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/DU_"+cell+"/Centipede_batch/"
    os.system("mkdir -p "+outputLocation)

    # Execution
    myL = "_".join([cell,mpbsName,"CCSB"])
    clusterCommand = "bsub "
    clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt -W 100:00 "
    clusterCommand += "-M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./createCentipedeSignal_pipeline.zsh "
    clusterCommand += mpbsName+" "+mpbsFileName+" "+consFileName+" "+tssFileName+" "+dnaseTotExt+" "
    clusterCommand += dnasePosFileName+" "+dnaseNegFileName+" "+outputLocation
    os.system(clusterCommand)


