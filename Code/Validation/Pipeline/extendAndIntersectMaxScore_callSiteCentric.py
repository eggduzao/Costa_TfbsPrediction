
# Import
import os
import sys
from glob import glob

# Cell Loop
cellList = ["H1hesc", "K562"]
for c in cellList:

  # Method Loop
  methodList = ["Centipede", "Cuellar", "FLR", "PIQ"]
  for m in methodList:

    # Percentile Loop
    pList = ["95", "99"]
    for p in pList:
    
      # Execution
      ext = "0"
      mpbsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/"+c+"/TC_DU.bed"
      predFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+c+"/"+m+"_"+p+".bed"
      outputFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/"+c+"/"+m+"_"+p+".bed"
      clusterCommand = "extendAndIntersectMaxScore "+ext+" "+ext+" "+mpbsFileName+" "+predFileName+" "+outputFileName
      os.system(clusterCommand)


