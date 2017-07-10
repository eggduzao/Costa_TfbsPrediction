
# Import
import os
import sys
from glob import glob

# TF Loop
tfList = [e.split("/")[-1].split(".")[0] for e in glob("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/MPBS/PWM/*.pwm")]
for tf in tfList:

  # Execution
  myL = "EVD_"+tf
  clusterCommand = "bsub -J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
  clusterCommand += "-W 1:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./evaluateDistribution_pipeline.zsh "+tf
  os.system(clusterCommand)
  # 


