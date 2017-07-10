
# Import
import os
import sys
from glob import glob

# Cell Loop
cellList = ["K562"]
for cell in cellList:

  # Signal Loop
  signalList = ["ATAC","DNase"]
  for signal in signalList:

    # TF Loop
    tfList = glob("/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/MPBSAWG/"+cell+"_Evidence/fdr_4/*.bed")
    for tfFile in tfList:

      # Parameters
      tfName = tfFile.split("/")[-1].split(".")[0]
      mpbsFileName = tfFile
      if(signal == "ATAC"):
        bamFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/ATAC/"+cell+"/ATAC.bam"
        outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results_ATAC/Footprints/Results/"+cell+"/ATAC_FS_/"
      elif(signal == "DNase"):
        bamFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNase_DU/"+cell+"/DNase.bam"
        outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results_ATAC/Footprints/Results/"+cell+"/DNase_FS_/"
      outputFileName = outputLocation+tfName+".bed"
      os.system("mkdir -p "+outputLocation)

      # Execution
      myL = "_".join([cell,signal,"FSBAM"])
      clusterCommand = "bsub "
      clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
      clusterCommand += "-W 200:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./applyFOSBam_pipeline.zsh "
      clusterCommand += mpbsFileName+" "+bamFileName+" "+outputFileName
      os.system(clusterCommand)


