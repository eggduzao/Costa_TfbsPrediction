
# Import
import os
import sys
from glob import glob

# Cell Paramters
cellList = ["H1hesc", "K562"]

# Cell Loop
for cell in cellList:

  # Factor Loop
  inFileList = glob("/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/"+cell+"_Evidence/fdr_4/*.bed")
  for inFileName in inFileList:

    # Auxiliary
    factor = inFileName.split("/")[-1].split(".")[0]
    
    # Fetching TC subset
    tcFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/TagCount_DU.bed"
    tcFactorFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/TagCount_DU/"+factor+".bed"
    os.system("mkdir -p /hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/TagCount_DU/")
    os.system("grep \"\t\""+factor+": "+tcFileName+" > "+tcFactorFileName)
    
    
