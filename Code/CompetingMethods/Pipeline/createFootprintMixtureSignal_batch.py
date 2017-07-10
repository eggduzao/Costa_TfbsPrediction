
# Import
import os
import sys
from glob import glob

# Lab Loop
labList = ["DU"]
for lab in labList:

  # Cell Loop
  #cellList = ["H1hesc", "K562", "GM12878"]
  cellList = ["GM12878"]
  for cell in cellList:

    # Factor Loop
    factorList = glob("/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/fdr_4_inside_HS/*.bed")
    for factorFileName in factorList:
 
      factorName = factorFileName.split("/")[-1].split(".")[0]
      organism = "hg19"

      # Parameters
      mpbsFileName = factorFileName
      bamFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNase_"+lab+"/"+cell+"/DNase.bam"
      genomeFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/"+organism.upper()+"/"+organism+".fa"
      outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"+lab+"_"+cell+"/FootprintMixture_batch/"
      outputFileName = outLoc+factorName
      os.system("mkdir -p "+outLoc)

      # Execution
      myL = "_".join([lab,cell,factorName,"CFMSB"])
      clusterCommand = "bsub "
      clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt -W 100:00 -M 12000 -S 100 "
      clusterCommand += "-R \"select[hpcwork]\" ./createFootprintMixtureSignal_pipeline.zsh "
      clusterCommand += mpbsFileName+" "+bamFileName+" "+genomeFileName+" "+outputFileName
      os.system(clusterCommand)
      # -P izkf

