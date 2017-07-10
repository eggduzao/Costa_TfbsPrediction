
# Import
import os
import sys

# Cell Line Parameters
cellList = ["GM12878"]

# Cell Loop
for cell in cellList:

  # Parameters
  fosThresh="0.95"
  coordFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/HMM/InputRegions/DU_"+cell+"/hd.bed"
  dnaseFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/DU_"+cell+"/DNase.bw"
  outputFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/Neph.bed"

  # Execution
  myL = "_".join([cell,"ANE"])
  clusterCommand = "bsub "
  clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
  clusterCommand += "-W 100:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./applyNeph_pipeline.zsh "
  clusterCommand += fosThresh+" "+coordFileName+" "+dnaseFileName+" "+outputFileName
  os.system(clusterCommand)


