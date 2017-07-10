
# Import
import os
import sys
from glob import glob

# Parameters
outputLocation = "./"
mpbsName = "JUN"
bedList = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/H1hesc/HINT-BC_D_DU.bed,/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/H1hesc/HINT_D_DU.bed"
typeList = "SB,SB"
labelList = "HINT-BC,HINT"

# Execution
clusterCommand = "rocFromBedSBSC "+mpbsName+" "+typeList+" "+labelList+" "+bedList+" "+outputLocation
os.system(clusterCommand)


