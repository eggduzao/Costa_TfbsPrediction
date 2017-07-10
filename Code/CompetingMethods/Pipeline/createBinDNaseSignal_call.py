
# Import
import os
import sys
from glob import glob

# Lab Loop
labList = ["DU"]
for lab in labList:

  # Cell Loop
  cellList = ["H1hesc","K562"]
  for cell in cellList:
  
    # Factor Loop
    factorList = glob("/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/fdr_4_inside_HS/*.bed")
    for factorFileName in factorList:
 
      factorName = factorFileName.split("/")[-1].split(".")[0]
      organism = "hg19"

      # Parameters
      mpbsFileName = factorFileName
      bamFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNase_"+lab+"/"+cell+"/DNase.bam"
      outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"+lab+"_"+cell+"/BinDNase_batch/"
      os.system("mkdir -p "+outputLocation)

      # Execution
      myL = "_".join([lab,cell,factorName,"CBDS"])
      clusterCommand = "bsub "
      clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt -W 120:00 -M 12000 -S 100 "
      clusterCommand += "-P izkf -R \"select[hpcwork]\" ./createBinDNaseSignal_pipeline.zsh "
      clusterCommand += mpbsFileName+" "+bamFileName+" "+outputLocation
      os.system(clusterCommand)


