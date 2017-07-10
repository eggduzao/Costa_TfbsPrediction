
# Import
import os
import sys
from glob import glob

# Cell Paramters
cellList = ["H1hesc", "K562"]

# Cell Loop
for cell in cellList:

  # Factor Loop
  inFileList = glob("/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/Cuellar5/DNase_priorH3K4me1_priorH3K4me3_priorH3K9ac_prior_*.bed")
  for inFileName in inFileList:

    # Auxiliary
    factor = inFileName.split("/")[-1].split(".")[0].split("_")[-1]
    
    # Parameters
    ext = "0"
    mpbsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/TagCount_DU/"+factor+".bed"
    predFileName = inFileName
    outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/Cuellar_DU/"
    outputFileName = outLoc+factor+"_"+ext+".bed"
    os.system("mkdir -p "+outLoc)

    # Execution
    myL = "_".join(["EIMS_CUE"])
    clusterCommand = "bsub "
    clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt -W 24:00 -M 12000 -S 100 "
    clusterCommand+= "-P izkf -R \"select[hpcwork]\" ./extendAndIntersectMaxScore_pipeline.zsh "
    clusterCommand += ext+" "+ext+" "+mpbsFileName+" "+predFileName+" "+outputFileName
    os.system(clusterCommand)


