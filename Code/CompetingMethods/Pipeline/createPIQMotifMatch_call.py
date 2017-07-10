
# Import
import os
import sys

# PWM Loop
pwmNbList = range(1,59)
for pwmNb in pwmNbList:
  
  # Parameters
  pwmN = str(pwmNb)
  pwmFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/PWM/PIQ/PIQjaspar.txt"
  outputLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/PIQ/"

  # Execution
  myL = "_".join([pwmN,"CPMM"])
  clusterCommand = "bsub "
  clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
  clusterCommand += "-W 24:00 -M 24000 -S 100 -P izkf -R \"select[hpcwork]\" ./createPIQMotifMatch_pipeline.zsh "
  clusterCommand += pwmN+" "+pwmFileName+" "+outputLocation
  os.system(clusterCommand)


