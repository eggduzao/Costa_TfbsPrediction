# This script fixes FLR calls by selecting only the instances
# above a certain threshold so this method can be considered segmentation-based

# Import
import os
import sys
from glob import glob

# Parameters
threshold = "9"
# This score represents the likelihood ratio of a sensitivity of 90%

# Cell Loop
cellList = ["H1hesc", "K562"]
for cell in cellList:

  inLoc1 = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/FootprintMixture2_DU/"
  inLoc2 = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/FootprintMixture_DU/"
  outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/FootprintMixture_DU_Filtered2/"
  os.system("mkdir -p "+outLoc)

  # Factor loop 1
  factorList = []
  for inFileName in glob(inLoc1+"*.bed"):
    fn = inFileName.split("/")[-1].split(".")[0]
    factorList.append(fn)
    outFileName = outLoc+fn+".bed"
    os.system("awk '$5>="+threshold+"' "+inFileName+" > "+outFileName)
    
  # Factor loop 2
  for inFileName in glob(inLoc2+"*.bed"):
    fn = inFileName.split("/")[-1].split(".")[0]
    if(fn in factorList): continue
    outFileName = outLoc+fn+".bed"
    os.system("awk '$5>="+threshold+"' "+inFileName+" > "+outFileName)


