
# Import
import os
import sys
from glob import glob

# Cell Paramters
cellList = ["H1hesc", "K562"]

# Cell Loop
for cell in cellList:

  # Factor Loop
  inFileList = glob("/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/FootprintMixture_DU_Filtered/*.bed")
  for inFileName in inFileList:

    # Auxiliary
    factor = inFileName.split("/")[-1].split(".")[0]
    
    # Parameters
    ext = "0"
    mpbsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/TagCount_DU/"+factor+".bed"
    predFileName = inFileName
    outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/FLR_DU2/"
    outputFileName = outLoc+factor+"_"+ext+".bed"
    os.system("mkdir -p "+outLoc)

    # Execution
    #myL = "_".join(["EIMS_FLR"])
    #clusterCommand = "bsub "
    #clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt -W 24:00 -M 12000 -S 100 "
    #clusterCommand+= "-P izkf -R \"select[hpcwork]\" ./extendAndIntersectMaxScore_pipeline.zsh "
    #clusterCommand += ext+" "+ext+" "+mpbsFileName+" "+predFileName+" "+outputFileName
    #os.system(clusterCommand)

    clusterCommand = "extendAndIntersectMaxScore "
    clusterCommand += ext+" "+ext+" "+mpbsFileName+" "+predFileName+" "+outputFileName
    os.system(clusterCommand)


