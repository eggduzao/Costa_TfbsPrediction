
# Import
import os
import sys
from glob import glob

# Cell Loop
cellList = ["GM12878", "H1hesc", "K562"]
for cell in cellList:

  # Parameters
  mpbsFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Validation/benchmark/MPBS_FLREXP.bed"
  flrFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/FLR_batch/all.bed"
  outputFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Validation/benchmark/res/DU_"+cell+"_FLR.bed"

  # Execution
  os.system("intersectBed -a "+flrFileName+" -b "+mpbsFileName+" -wa -u > "+outputFileName)


