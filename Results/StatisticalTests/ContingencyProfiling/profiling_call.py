
# Import
import os
import sys
from glob import glob

# TF Loop
tfDict = dict([(e.split("/")[-1].split("_")[0],None) for e in glob("./contingency_files/*.*")])
tfList = sorted(tfDict.keys())
for tf in tfList:

  # Parameters
  totalWindow="200"
  bedFileNameList="./contingency_files/"+tf+"_TP.bed,./contingency_files/"+tf+"_FP.bed,./contingency_files/"+tf+"_TN.bed,./contingency_files/"+tf+"_FN.bed"
  bamFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/DNase_DU/K562/DNase.bam"
  outFileName="./contingency_profiles/"+tf+".txt"
  signalExt="1"
  signalExtBoth="N"

  # Execution
  #myL = "_".join(["PG",tf])
  #clusterCommand = "bsub "
  #clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
  #clusterCommand += "-W 100:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./profiling_pipeline.zsh "
  #clusterCommand += totalWindow+" "+bedFileNameList+" "+bamFileName+" "+outFileName+" "+signalExt+" "+signalExtBoth
  #os.system(clusterCommand)

  clusterCommand = "profileGraph "
  clusterCommand += totalWindow+" "+bedFileNameList+" "+bamFileName+" "+outFileName+" "+signalExt+" "+signalExtBoth
  os.system(clusterCommand)


