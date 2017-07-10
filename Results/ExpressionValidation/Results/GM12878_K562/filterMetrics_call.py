
# Import
import os
import sys
from glob import glob

# TF Loop
inFileList = glob("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Validation/distribution/*.txt")

inFileList = ["/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Validation/distribution/ZNF263.txt"]

for inFileName in inFileList:

  # Execution
  tf = inFileName.split("/")[-1].split(".")[0]
  myL = "FLM_"+tf
  clusterCommand = "bsub -J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt -W 1:00 -M 24000 "
  clusterCommand += "-S 100 -R \"select[hpcwork]\" ./filterMetrics_pipeline.zsh "+inFileName
  os.system(clusterCommand)
  # -P izkf


