# This script fixes Centipede calls by selecting only the instances
# above a certain threshold so this method can be considered segmentation-based

# Import
import os
import sys
from glob import glob

# Parameters
threshold = "0.90"

# Cell Loop
cellList = ["H1hesc", "K562"]
for cell in cellList:

  inLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/"+cell+"/fdr_4/Pique/"
  outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/Pique/"
  os.system("mkdir -p "+outLoc)

  # Factor loop
  for inFileName in glob(inLoc+"*_ces.bed"):
  
    fn = inFileName.split("/")[-1]
    outFileName = outLoc+fn

    os.system("awk '$5>="+threshold+"' "+inFileName+" > "+outFileName)


