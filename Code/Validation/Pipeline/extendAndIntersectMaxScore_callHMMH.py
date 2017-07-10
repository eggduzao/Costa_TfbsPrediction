
# Import
import os
import sys
from glob import glob

# Cell Paramters
il = "/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/Predictions/"
inFileList = [
il+"H1hesc/H-HMM_D13_DU.bed",
il+"K562/H-HMM_D13_DU.bed"
]

# Cell Loop
for inFileName in inFileList:

  # Auxiliary
  cell = inFileName.split("/")[-2]

  # Parameters
  ext = "0"
  mpbsFileName="/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/Results/"+cell+"/TCH_DU.bed"
  predFileName = inFileName
  outputFileName="/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/Results/"+cell+"/HINT_H13_DU.bed"

  # Execution
  myL = "_".join([cell,"EIMH"])
  clusterCommand = "bsub "
  clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
  clusterCommand+= "-W 24:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./extendAndIntersectMaxScore_pipeline.zsh "
  clusterCommand += ext+" "+ext+" "+mpbsFileName+" "+predFileName+" "+outputFileName
  os.system(clusterCommand)


