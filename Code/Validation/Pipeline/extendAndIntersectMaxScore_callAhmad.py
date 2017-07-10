
# Import
import os
import sys
from glob import glob

# Cell Paramters
il = "/hpcwork/izkf/projects/TfbsPrediction/Results/SignalProcessing/Predictions/"
ol = "/hpcwork/izkf/projects/TfbsPrediction/Results/SignalProcessing/Results/"
inList = [ "butterlow", "elliphigh", "endstd_1", "order_1.2", "Order_1", "Order_4", "Order_8", "Startstd_4",
           "Startstd_5", "window_20", "window_30", "wp_0.2", "wp_0.6", "wp_0.8"]

# Cell Loop
for inName in inList:
    
  # Parameters
  ext = "5"
  mpbsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/K562/TC_DU.bed"
  predFileName = il+inName+".bed"
  outputFileName = ol+inName+".bed"

  # Execution
  myL = "_".join(["EIMS_AHM"])
  clusterCommand = "bsub "
  clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt -W 24:00 -M 12000 -S 100 "
  clusterCommand+= "-P izkf -R \"select[hpcwork]\" ./extendAndIntersectMaxScore_pipeline.zsh "
  clusterCommand += ext+" "+ext+" "+mpbsFileName+" "+predFileName+" "+outputFileName
  os.system(clusterCommand)


