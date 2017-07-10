
# Import
import os
import sys
from glob import glob

# Cell Loop
cellList = ["GM12878", "H1hesc", "K562"]
for cell in cellList:

  # Parameters
  mpbsFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Validation/benchmark/MPBS_FLREXP.bed"
  signalFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Bias/DU_"+cell+"/DNaseBiasCorrected.bw"
  outputFileNameFos = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Validation/benchmark/res/DU_"+cell+"_FS.bed"
  
  # Execution
  myL = "ETPB_"+cell
  clusterCommand = "bsub -J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
  clusterCommand += "-W 100:00 -M 48000 -S 100 -P izkf -R \"select[hpcwork]\" ./evaluateTcProtFos_pipeline.zsh "
  clusterCommand += mpbsFileName+" "+signalFileName+" "+outputFileNameFos
  os.system(clusterCommand)


