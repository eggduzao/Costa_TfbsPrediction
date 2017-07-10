#!/bin/zsh

# Import
import os
import sys
from glob import glob

# Cell Paramters
il = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions_NM/"
cellList = ["H1hesc", "K562"]

# Cell Loop
for cell in cellList:

  # Factor Loop
  inFileList = glob(il+cell+"/PIQ_DU/PIQ_*.bed")
  for inFileName in inFileList:

    # Auxiliary
    factor = "_".join(inFileName.split("/")[-1].split(".")[0].split("_")[1:])

    # Parameters
    ext = "0"
    mpbsFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/"+cell+"_Evidence/fdr_4/"+factor+".bed"
    predFileName = inFileName
    outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/PIQ_DU/"
    outputFileName = outLoc+factor+"_"+ext+".bed"
    os.system("mkdir -p "+outLoc)

    # Execution
    myL = "_".join([cell,factor,"EIKFM"])
    clusterCommand = "bsub "
    clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt -W 24:00 -M 12000 -S 100 "
    clusterCommand+= "-P izkf -R \"select[hpcwork]\" ./extendAndIntersectKeepScore_pipeline.zsh "
    clusterCommand += ext+" "+ext+" "+mpbsFileName+" "+predFileName+" "+outputFileName
    os.system(clusterCommand)


