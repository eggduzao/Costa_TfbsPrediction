#!/bin/zsh

# Import
import os
import sys
from glob import glob

# Cell Paramters
il = "/hpcwork/izkf/projects/TfbsPrediction/Results_ATAC/Footprints/Predictions/K562/"
inFileList = [
"ATAC_HINT_oneside_1",
"ATAC_HINT_oneside_3",
"ATAC_HINT_oneside_5",
"ATAC_HINT_oneside_7",
"ATAC_HINT_twoside_1",
"ATAC_HINT_twoside_3",
"ATAC_HINT_twoside_5",
"ATAC_HINT_twoside_7",
"sc_ATAC_HINT_twoside_1"
]

# Cell Loop
for inName in inFileList:

  # Parameters
  ext = "5"
  mpbsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results_ATAC/Footprints/Results/K562/ATAC_TC_50.bed"
  predFileName = il+inName+".bed"
  outputFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results_ATAC/Footprints/Results/K562/"+inName+".bed"

  # Execution
  myL = "_".join([inName,"EIMA"])
  clusterCommand = "bsub "
  clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
  clusterCommand+= "-W 24:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./extendAndIntersectMaxScore_pipeline.zsh "
  clusterCommand += ext+" "+ext+" "+mpbsFileName+" "+predFileName+" "+outputFileName
  os.system(clusterCommand)


